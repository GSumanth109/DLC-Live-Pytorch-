"""
gui/main_window.py

Defines the `App` class, the main QMainWindow for the application.

This class is responsible for:
- Building the entire user interface (widgets, layouts, menus).
- Loading and saving configuration settings (`rt_config.json`).
- Handling all user interactions (button clicks, slider moves).
- Managing the application's state (Preview vs. Analysis).
- Orchestrating all backend threads and processes:
    - `PreviewWorker` (for lightweight camera preview).
    - `VideoProcessingThread` (for reading frames during analysis).
    - `InferenceProcessManager` (for managing the inference process).
    - `GuiUpdateWorker` (for displaying results at a target FPS).
"""

import sys
import os
import time
import cv2
import numpy as np
import multiprocessing as mp
from collections import deque
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy, QMessageBox, QSlider,
    QGroupBox, QFormLayout, QRadioButton, QLineEdit, QCheckBox, QToolBar, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from utils.helpers import ActiveQueueReference
from utils.config import (
    gather_settings, gather_preview_settings,
    load_settings_from_file, save_settings_to_file
)
from utils.drawing import draw_skeleton, draw_latency_overlay
from processing.manager import InferenceProcessManager
from processing.video import VideoProcessingThread
from gui.gui_update_worker import GuiUpdateWorker
from gui.preview_worker import PreviewWorker


class App(QMainWindow):
    """
    The main application window.
    
    Manages UI, state, and coordinates all background threads/processes
    for video capture, inference, and display.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Live - (Preview Mode)")
        
        # --- Application State ---
        self.settings = {}
        self.latest_original_frame = None
        self.latest_processed_frame = None
        self.video_writer = None
        self._slider_busy = False # Flag to prevent slider/textbox update loops

        # --- Threads & Processes ---
        self.video_processing_thread = None
        self.inference_manager_thread = None
        self.gui_update_worker = None
        self.preview_worker = None
        
        # --- Inter-process Communication ---
        self.shutdown_event = mp.Event()
        self.results_queue = None
        self.camera_command_queue = mp.Queue()
        self.active_queue_ref = ActiveQueueReference()
        
        # --- Shared Statistics Counters ---
        self.frames_captured_counter = mp.Value('i', 0)
        self.frames_enqueued_counter = mp.Value('i', 0)
        self.frames_processed_counter = mp.Value('i', 0)
        self.csv_write_counter = mp.Value('i', 0)
        
        # --- Stats Display Timer ---
        self.last_csv_count = 0
        self.last_stats_time = time.monotonic()
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(500)
        self.stats_timer.timeout.connect(self.update_stats_display)

        # --- Initialization ---
        self.create_widgets()
        self.load_app_config()
        self.update_camera_source_ui() 
        
        # Start the camera preview shortly after the app loads
        QTimer.singleShot(100, self.start_preview)

    # ==========================================================================
    # UI CREATION
    # ==========================================================================

    def create_widgets(self):
        """Builds all widgets and layouts for the main window."""
        self.create_toolbar()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Left Control Panel ---
        ctrl_layout = QVBoxLayout()
        main_layout.addLayout(ctrl_layout)

        # --- Right Video Panel ---
        vid_layout = QVBoxLayout()
        main_layout.addLayout(vid_layout, 1) # Give video panel more space

        # Group 1: Model
        g1 = QGroupBox("1. Input Model")
        ctrl_layout.addWidget(g1)
        f1 = QFormLayout(g1)
        self.cfg_edit = self._create_file_input(f1, "Config:", file_filter="YAML Config (*.yaml)")
        self.snap_edit = self._create_file_input(f1, "Snapshot:", file_filter="Model Files (*.pt *.pb)")

        # Group 2: Source
        g2 = QGroupBox("2. Source")
        ctrl_layout.addWidget(g2)
        v2 = QVBoxLayout(g2)
        self.usb_rb = QRadioButton("USB Webcam")
        self.usb_rb.setChecked(True)
        self.file_rb = QRadioButton("Video File")
        v2.addWidget(self.usb_rb)
        v2.addWidget(self.file_rb)
        
        self.usb_rb.toggled.connect(self.update_camera_source_ui)
        self.file_rb.toggled.connect(self.update_camera_source_ui)
        
        h2 = QHBoxLayout()
        self.src_lbl = QLabel("Index:")
        self.src_edit = QLineEdit("0")
        self.src_file_edit = QLineEdit()
        self.src_browse_btn = QPushButton("...")
        self.src_browse_btn.clicked.connect(lambda: self.browse_file(self.src_file_edit, is_open=True, file_filter="Video Files (*.mp4 *.avi *.mov *.mkv)"))
        
        h2.addWidget(self.src_lbl)
        h2.addWidget(self.src_edit)
        h2.addWidget(self.src_file_edit)
        h2.addWidget(self.src_browse_btn)
        v2.addLayout(h2)

        # Group 3: Camera Parameters
        self.cam_grp = QGroupBox("3. Cam Params")
        ctrl_layout.addWidget(self.cam_grp)
        f3 = QFormLayout(self.cam_grp)
        self.w_edit = QLineEdit("640")
        self.h_edit = QLineEdit("480")
        self.fps_edit = QLineEdit("60")
        f3.addRow("W:", self.w_edit)
        f3.addRow("H:", self.h_edit)
        f3.addRow("FPS:", self.fps_edit)
        
        self.apply_source_btn = QPushButton("Reload Source / Apply Cam Settings")
        self.apply_source_btn.clicked.connect(self.on_reload_source)
        f3.addRow(self.apply_source_btn)
        
        self.w_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.h_edit.textChanged.connect(self._update_crop_slider_ranges)

        # Group 4: Realtime Camera Control
        self.rt_grp = QGroupBox("4. Realtime")
        ctrl_layout.addWidget(self.rt_grp)
        f4 = QFormLayout(self.rt_grp)
        self.exp_sld = self._create_slider(f4, "Exp:", -13, -1, -11, self.update_exposure)
        self.gain_sld = self._create_slider(f4, "Gain:", 0, 128, 0, self.update_gain)
        self.wb_sld = self._create_slider(f4, "WB:", 2000, 8000, 4000, self.update_white_balance)

        # Group 5: Cropping
        g5 = QGroupBox("5. Cropping")
        ctrl_layout.addWidget(g5)
        f5 = QFormLayout(g5)
        self.crop_cb = QCheckBox("Enable Crop")
        f5.addRow(self.crop_cb)
        self.crop_x_edit = QLineEdit("0")
        self.crop_x_sld = QSlider(Qt.Horizontal)
        f5.addRow("Crop X:", self.crop_x_edit)
        f5.addRow(self.crop_x_sld)
        self.crop_y_edit = QLineEdit("0")
        self.crop_y_sld = QSlider(Qt.Horizontal)
        f5.addRow("Crop Y:", self.crop_y_edit)
        f5.addRow(self.crop_y_sld)
        self.crop_w_edit = QLineEdit("640")
        self.crop_w_sld = QSlider(Qt.Horizontal)
        f5.addRow("Crop W:", self.crop_w_edit)
        f5.addRow(self.crop_w_sld)
        self.crop_h_edit = QLineEdit("480")
        self.crop_h_sld = QSlider(Qt.Horizontal)
        f5.addRow("Crop H:", self.crop_h_edit)
        f5.addRow(self.crop_h_sld)
        self._connect_crop_widgets()
        
        # Group 6: Pre-processing
        g6 = QGroupBox("6. Pre-processing")
        ctrl_layout.addWidget(g6)
        f6 = QFormLayout(g6)
        self.pre_cmb = QComboBox()
        self.pre_cmb.addItems(["None", "Flat-field", "Morphological Opening"])
        self.pre_cmb.currentTextChanged.connect(self.update_preprocessing_ui)
        f6.addRow("Method:", self.pre_cmb)
        self.flat_edit = self._create_file_input(f6, "Flat Img:", file_filter="Image Files (*.png *.jpg *.bmp *.tif)")
        self.flat_edit.parentWidget().setVisible(False) # Hide by default

        # Group 7: Performance & Display
        g7 = QGroupBox("7. Performance & Display")
        ctrl_layout.addWidget(g7)
        f7 = QFormLayout(g7)
        self.ram_sld = self._create_slider(f7, "RAM Rst:", 4, 32, 16)
        if not PSUTIL_AVAILABLE:
            self.ram_sld.setEnabled(False)
            f7.addRow(QLabel("<i>psutil not installed</i>"))
        self.bs_sld = self._create_slider(f7, "Batch(1):", 1, 1, 1)
        self.bs_sld.setEnabled(False) # Batch size > 1 not implemented
        self.fp16_cb = QCheckBox("Use FP16")
        self.fp16_cb.setChecked(True)
        f7.addRow(self.fp16_cb)
        self.disp_fps_sld = self._create_slider(f7, "Disp FPS:", 1, 120, 60)
        self.skel_sld = self._create_slider(f7, "Skel Conf:", 0, 100, 10)
        self.pt_sld = self._create_slider(f7, "Pt Conf:", 0, 100, 60)
        self.show_skel_cb = QCheckBox("Show Skel")
        self.show_skel_cb.setChecked(True)
        f7.addRow(self.show_skel_cb)

        # Group 8: Output
        g8 = QGroupBox("8. Output")
        ctrl_layout.addWidget(g8)
        f8 = QFormLayout(g8)
        self.csv_cb = QCheckBox("Save CSV")
        self.csv_edit = self._create_file_input(f8, self.csv_cb, is_open=False, file_filter="CSV files (*.csv)")
        self.vid_cb = QCheckBox("Save Vid")
        self.vid_edit = self._create_file_input(f8, self.vid_cb, is_open=False, file_filter="Video Files (*.mp4)")

        # Group 9: Capture
        g9 = QGroupBox("9. Capture")
        ctrl_layout.addWidget(g9)
        h9 = QHBoxLayout(g9)
        self.pht_btn = QPushButton("Photo")
        self.pht_btn.clicked.connect(self.take_photo)
        self.rec_btn = QPushButton("Record")
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self.toggle_recording)
        h9.addWidget(self.pht_btn)
        h9.addWidget(self.rec_btn)

        # Group 10: Stats
        g10 = QGroupBox("10. Stats")
        ctrl_layout.addWidget(g10)
        v10 = QVBoxLayout(g10)
        self.st_lbl = QLabel("Cap: 0 | Enq: 0 | Proc: 0 | Drop: 0 | CSV FPS: 0.0")
        self.st_lbl.setAlignment(Qt.AlignLeft)
        v10.addWidget(self.st_lbl)
        
        # --- Start/Stop Controls ---
        h_start = QHBoxLayout()
        self.start_btn = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Analysis")
        self.stop_btn.clicked.connect(self.on_stop_button_clicked)
        self.start_btn.setEnabled(False) # Enabled after preview starts
        self.stop_btn.setEnabled(False) # Enabled after analysis starts
        h_start.addWidget(self.start_btn)
        h_start.addWidget(self.stop_btn)
        ctrl_layout.addLayout(h_start)
        
        ctrl_layout.addStretch() # Pushes all controls to the top

        # --- Video Display Labels ---
        vid_group = QGroupBox('Preview')
        vid_layout.addWidget(vid_group, 1)
        vid_box = QHBoxLayout(vid_group)
        self.orig_lbl = QLabel('Original (Live Preview)')
        self.proc_lbl = QLabel('Processed (Awaiting Analysis)')
        for lbl in [self.orig_lbl, self.proc_lbl]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 240)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setScaledContents(False) # We handle scaling manually
            vid_box.addWidget(lbl)
            
        # --- Status Bar ---
        self.lat_lbl = QLabel("Inf: -- | Cap-CSV: -- | E2E: -- | RAM: --")
        self.statusBar().addPermanentWidget(self.lat_lbl)
        self.statusBar().showMessage("Initializing preview...")

    def create_toolbar(self):
        """Creates the top application toolbar."""
        tb = self.addToolBar('File')
        save_action = QAction(QIcon.fromTheme("document-save"), 'Save Settings', self)
        save_action.triggered.connect(self.save_app_config)
        tb.addAction(save_action)
        tb.addSeparator()
        o = QAction('Set Flat-field Image', self)
        o.triggered.connect(lambda: self.browse_file(self.flat_edit, file_filter="Image Files (*.png *.jpg *.bmp *.tif)"))
        tb.addAction(o)
        e = QAction(QIcon.fromTheme("application-exit"), 'Exit', self)
        e.triggered.connect(self.close)
        tb.addAction(e)

    def _create_file_input(self, layout, label_widget, is_open=True, file_filter="All Files (*)"):
        """
        Helper function to create a text box + "Browse" button row.

        Args:
            layout (QFormLayout): The layout to add the row to.
            label_widget (str or QWidget): The label for the form row.
            is_open (bool): True for "Open" dialog, False for "Save As" dialog.
            file_filter (str): The file filter for the dialog.

        Returns:
            QLineEdit: The text edit widget.
        """
        edit = QLineEdit()
        btn_txt = "..." if is_open else "Save As"
        btn = QPushButton(btn_txt)
        btn.clicked.connect(lambda: self.browse_file(edit, is_open, file_filter))
        h_layout = QHBoxLayout()
        h_layout.addWidget(edit)
        h_layout.addWidget(btn)
        if isinstance(label_widget, str): layout.addRow(label_widget, h_layout)
        else: layout.addRow(label_widget, h_layout)
        return edit

    def _create_slider(self, layout, label, min_v, max_v, def_v, func=None):
        """
        Helper function to create a slider + value label row.

        Args:
            layout (QFormLayout): The layout to add the row to.
            label (str): The text label for the row.
            min_v (int): Slider minimum value.
            max_v (int): Slider maximum value.
            def_v (int): Slider default value.
            func (callable, optional): A function to connect to 
                                       `valueChanged`.

        Returns:
            QSlider: The slider widget.
        """
        sld = QSlider(Qt.Horizontal)
        sld.setRange(min_v, max_v)
        sld.setValue(def_v)
        v_lbl = QLabel(f"{def_v}")
        sld.valueChanged.connect(lambda v, lbl=v_lbl: lbl.setText(f"{v}"))
        if func: sld.valueChanged.connect(func)
        h_layout = QHBoxLayout(); h_layout.addWidget(sld); h_layout.addWidget(v_lbl)
        layout.addRow(label, h_layout); return sld

    # ==========================================================================
    # CONFIG FILE HANDLING
    # ==========================================================================

    def load_app_config(self):
        """Loads settings from `rt_config.json` and populates the UI."""
        settings = load_settings_from_file()
        if settings:
            self.populate_ui_from_settings(settings)
        else:
            # If no config exists, save the current defaults
            self.save_app_config()

    @pyqtSlot()
    def save_app_config(self):
        """Saves the current UI settings to `rt_config.json`."""
        settings_dict = self.get_ui_settings_as_dict()
        save_settings_to_file(settings_dict)
        self.statusBar().showMessage("Settings saved to rt_config.json.", 3000)

    def get_ui_settings_as_dict(self):
        """
        Gathers all values from the UI widgets into a dictionary.

        Returns:
            dict: A dictionary of all UI settings.
        """
        settings = {
            'source_is_usb': self.usb_rb.isChecked(),
            'camera_path': self.src_edit.text(),
            'video_file_path': self.src_file_edit.text(),
            'config_path': self.cfg_edit.text(),
            'snapshot_path': self.snap_edit.text(),
            'cam_width': self.w_edit.text(),
            'cam_height': self.h_edit.text(),
            'cam_fps': self.fps_edit.text(),
            'exposure': self.exp_sld.value(),
            'gain': self.gain_sld.value(),
            'white_balance': self.wb_sld.value(),
            'crop_enabled': self.crop_cb.isChecked(),
            'crop_x': self.crop_x_edit.text(),
            'crop_y': self.crop_y_edit.text(),
            'crop_w': self.crop_w_edit.text(),
            'crop_h': self.crop_h_edit.text(),
            'method': self.pre_cmb.currentText(),
            'flat_image_path': self.flat_edit.text(),
            'ram_threshold_gb': self.ram_sld.value(),
            'use_fp16': self.fp16_cb.isChecked(),
            'target_fps': self.disp_fps_sld.value(),
            'skeleton_confidence': self.skel_sld.value() / 100.0,
            'point_confidence': self.pt_sld.value() / 100.0,
            'show_skeleton': self.show_skel_cb.isChecked(),
            'save_csv': self.csv_cb.isChecked(),
            'csv_output_path': self.csv_edit.text(),
            'save_video': self.vid_cb.isChecked(),
            'video_output_path': self.vid_edit.text()
        }
        return settings

    def populate_ui_from_settings(self, settings):
        """
        Sets all UI fields from a loaded settings dictionary.

        Args:
            settings (dict): The dictionary loaded from the config file.
        """
        if settings.get('source_is_usb', True):
            self.usb_rb.setChecked(True)
        else:
            self.file_rb.setChecked(True)
        
        self.src_edit.setText(str(settings.get('camera_path', '0')))
        self.src_file_edit.setText(str(settings.get('video_file_path', '')))
        
        self.cfg_edit.setText(str(settings.get('config_path', '')))
        self.snap_edit.setText(str(settings.get('snapshot_path', '')))
        self.w_edit.setText(str(settings.get('cam_width', '640')))
        self.h_edit.setText(str(settings.get('cam_height', '480')))
        self.fps_edit.setText(str(settings.get('cam_fps', '60')))
        
        self.exp_sld.setValue(settings.get('exposure', -11))
        self.gain_sld.setValue(settings.get('gain', 0))
        self.wb_sld.setValue(settings.get('white_balance', 4000))
        
        self.crop_cb.setChecked(settings.get('crop_enabled', False))
        self.crop_x_edit.setText(str(settings.get('crop_x', '0')))
        self.crop_y_edit.setText(str(settings.get('crop_y', '0')))
        self.crop_w_edit.setText(str(settings.get('crop_w', '640')))
        self.crop_h_edit.setText(str(settings.get('crop_h', '480')))
        
        self.pre_cmb.setCurrentText(str(settings.get('method', 'None')))
        self.flat_edit.setText(str(settings.get('flat_image_path', '')))
        
        self.ram_sld.setValue(settings.get('ram_threshold_gb', 16))
        self.fp16_cb.setChecked(settings.get('use_fp16', True))
        self.disp_fps_sld.setValue(settings.get('target_fps', 60))
        
        self.skel_sld.setValue(int(settings.get('skeleton_confidence', 0.1) * 100))
        self.pt_sld.setValue(int(settings.get('point_confidence', 0.6) * 100))
        self.show_skel_cb.setChecked(settings.get('show_skeleton', True))
        
        self.csv_cb.setChecked(settings.get('save_csv', False))
        self.csv_edit.setText(str(settings.get('csv_output_path', '')))
        self.vid_cb.setChecked(settings.get('save_video', False))
        self.vid_edit.setText(str(settings.get('video_output_path', '')))
        
    # ==========================================================================
    # UI SLOTS & CALLBACKS
    # ==========================================================================
    
    def _connect_crop_widgets(self):
        """Connects all crop sliders and text boxes for two-way updates."""
        self.crop_x_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_x_edit, v))
        self.crop_y_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_y_edit, v))
        self.crop_w_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_w_edit, v))
        self.crop_h_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_h_edit, v))
        
        self.crop_x_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_x_sld, t))
        self.crop_y_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_y_sld, t))
        self.crop_w_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_w_sld, t))
        self.crop_h_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_h_sld, t))
        
        # Update slider ranges when text changes
        self.crop_x_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.crop_y_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.crop_w_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.crop_h_edit.textChanged.connect(self._update_crop_slider_ranges)

    def _set_text_from_slider(self, text_edit, value):
        """
        Updates a QLineEdit from its corresponding QSlider.
        Uses `_slider_busy` to prevent feedback loops.
        """
        if self._slider_busy: return
        self._slider_busy = True
        text_edit.setText(str(value))
        self._slider_busy = False

    def _set_slider_from_text(self, slider, text):
        """
        Updates a QSlider from its corresponding QLineEdit.
        Uses `_slider_busy` to prevent feedback loops.
        """
        if self._slider_busy: return
        self._slider_busy = True
        try:
            slider.setValue(int(text))
        except ValueError:
            pass # Ignore invalid (e.g., empty) text
        self._slider_busy = False

    @pyqtSlot()
    def _update_crop_slider_ranges(self):
        """
        Dynamically adjusts the crop slider ranges based on the source
        dimensions and the current crop settings.
        
        Ensures that:
        - X cannot be > (Width - CropWidth)
        - W cannot be > (Width - X)
        - (and same for Y/H)
        """
        if self._slider_busy: return
        try:
            cam_w, cam_h = int(self.w_edit.text()), int(self.h_edit.text())
            
            # If a video file is loaded, use its actual dimensions
            if self.file_rb.isChecked() and self.latest_original_frame is not None:
                cam_h, cam_w = self.latest_original_frame.shape[:2]

            x, y = int(self.crop_x_edit.text()), int(self.crop_y_edit.text())
            w, h = int(self.crop_w_edit.text()), int(self.crop_h_edit.text())
            
            # Update ranges dynamically
            self.crop_x_sld.setRange(0, max(0, cam_w - w))
            self.crop_y_sld.setRange(0, max(0, cam_h - h))
            self.crop_w_sld.setRange(1, max(1, cam_w - x))
            self.crop_h_sld.setRange(1, max(1, cam_h - y))
        except ValueError:
            pass # Ignore errors if text boxes are invalid

    @pyqtSlot()
    def update_camera_source_ui(self):
        """
        Shows/hides UI elements based on whether "USB Webcam" or
        "Video File" is selected.
        """
        is_usb = self.usb_rb.isChecked()
        self.src_lbl.setVisible(is_usb)
        self.src_edit.setVisible(is_usb)
        self.src_file_edit.setVisible(not is_usb)
        self.src_browse_btn.setVisible(not is_usb)
        
        # Disable camera/realtime controls for video files
        self.cam_grp.setEnabled(is_usb)
        self.rt_grp.setEnabled(is_usb)

    def browse_file(self, edit_widget, is_open=True, file_filter="All Files (*)"):
        """
        Opens a file dialog to select a file or save location.

        Args:
            edit_widget (QLineEdit): The text box to populate with the path.
            is_open (bool): True for "Open", False for "Save As".
            file_filter (str): The file filter string.
        """
        if is_open:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save File As", "", file_filter)
        if path:
            edit_widget.setText(path)

    @pyqtSlot(str)
    def update_preprocessing_ui(self, method_name):
        """Shows the 'Flat Img' input only if 'Flat-field' is selected."""
        self.flat_edit.parentWidget().setVisible(method_name == "Flat-field")
    
    @pyqtSlot()
    def on_reload_source(self):
        """
        Slot for the "Reload Source" button.
        Stops the current preview and starts a new one with the
        updated settings.
        """
        self.statusBar().showMessage("Restarting preview...")
        self.stop_preview()
        # Use QTimer to ensure the stop event processes before starting again
        QTimer.singleShot(100, self.start_preview)

    @pyqtSlot(int)
    def update_exposure(self, value):
        """Sends an 'exposure' command to the camera command queue."""
        if self.camera_command_queue:
            self.camera_command_queue.put({'property': cv2.CAP_PROP_EXPOSURE, 'value': float(value)})

    @pyqtSlot(int)
    def update_gain(self, value):
        """Sends a 'gain' command to the camera command queue."""
        if self.camera_command_queue:
            self.camera_command_queue.put({'property': cv2.CAP_PROP_GAIN, 'value': float(value)})

    @pyqtSlot(int)
    def update_white_balance(self, value):
        """Sends a 'white balance' command to the camera command queue."""
        if self.camera_command_queue:
            self.camera_command_queue.put({'property': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 'value': float(value)})

    @pyqtSlot()
    def update_stats_display(self):
        """
        Called by `stats_timer` to update the statistics label.
        Calculates frame drops and CSV write FPS.
        """
        if not self.stop_btn.isEnabled(): return # Don't run if not analyzing
        
        current_time = time.monotonic()
        time_elapsed = current_time - self.last_stats_time
        if time_elapsed < 0.1: return # Avoid division by zero

        cap = self.frames_captured_counter.value
        enq = self.frames_enqueued_counter.value
        proc = self.frames_processed_counter.value
        drop = cap - enq # Frames dropped by VideoThread (queue full)
        
        csv_now = self.csv_write_counter.value
        csv_count_diff = csv_now - self.last_csv_count
        csv_fps = csv_count_diff / time_elapsed
        
        txt = (f"Cap:{cap} | Enq:{enq} | Proc:{proc} | Drop(Ovr):{drop} | CSV FPS: {csv_fps:.1f}")
        self.st_lbl.setText(txt)
        
        self.last_csv_count = csv_now
        self.last_stats_time = current_time

    # ==========================================================================
    # CORE APPLICATION LOGIC
    # ==========================================================================

    @pyqtSlot()
    def start_preview(self):
        """
        Starts the lightweight preview mode.
        
        - Stops any existing preview.
        - Gathers preview-specific settings.
        - Starts a `PreviewWorker` thread to grab frames.
        - If 'Video File' is selected, it loads the first frame as a static preview.
        """
        self.stop_preview() 
        preview_settings = gather_preview_settings(self)
        if preview_settings is None:
            QMessageBox.critical(self, "Preview Error", "Invalid settings. Cannot start preview.")
            return

        self.proc_lbl.setText("Processed (Awaiting Analysis)")
        self.orig_lbl.setText("Starting preview...")
        self.statusBar().showMessage("Preview running.")
        self.setWindowTitle("DLC Live - (Preview Mode)")

        if preview_settings['camera_source'] == "USB Webcam":
            self._update_crop_slider_ranges()
            self.preview_worker = PreviewWorker(preview_settings, self.camera_command_queue)
            self.preview_worker.new_frame.connect(self.update_preview_feed)
            self.preview_worker.start()
        
        elif preview_settings['camera_source'] == "Video File":
            # For video files, just load the first frame as a static image
            try:
                if not os.path.exists(preview_settings['path']):
                    raise IOError("File not found.")
                cap = cv2.VideoCapture(preview_settings['path'])
                ret, frame = cap.read()
                if not ret:
                    raise IOError("Could not read first frame.")
                cap.release()
                
                # Update UI with video dimensions
                h, w = frame.shape[:2]
                self.w_edit.setText(str(w))
                self.h_edit.setText(str(h))
                self._update_crop_slider_ranges()
                
                self.update_preview_feed(frame) # Show the frame
            except Exception as e:
                self.orig_lbl.setText("Could not load video file.")
                QMessageBox.warning(self, "Preview Error", f"Could not load video:\n{e}")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop_preview(self):
        """Stops the `PreviewWorker` thread if it is running."""
        if self.preview_worker:
            self.preview_worker.stop()
            self.preview_worker.wait(2000) # Wait for thread to finish
            self.preview_worker = None

    @pyqtSlot(object)
    def update_preview_feed(self, frame):
        """
        Slot for the `PreviewWorker.new_frame` signal.
        Displays the new frame in the 'Original' label and draws the
        crop rectangle if enabled.
        
        Args:
            frame (np.ndarray): The new frame from the camera.
        """
        self.latest_original_frame = frame
        self.latest_processed_frame = None # No processing in preview
        display_frame = frame.copy()
        
        # Draw the crop box overlay
        if self.crop_cb.isChecked():
            try:
                x, y = int(self.crop_x_edit.text()), int(self.crop_y_edit.text())
                w, h = int(self.crop_w_edit.text()), int(self.crop_h_edit.text())
                h_img, w_img = display_frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2) 
            except ValueError:
                pass # Ignore if crop values are invalid
                
        self.orig_lbl.setPixmap(self.to_pixmap(display_frame, self.orig_lbl))

    def start_analysis(self):
        """
        Starts the full analysis pipeline.
        
        - Stops the preview.
        - Gathers and validates all settings.
        - Resets all counters, events, and queues.
        - Starts the `VideoProcessingThread` (frame capture).
        - Starts the `InferenceProcessManager` (manages the inference process).
        - Starts the `GuiUpdateWorker` (updates the GUI with results).
        """
        self.stop_preview()
        
        self.settings = gather_settings(self)
        if not self.settings:
            self.start_preview() # Revert to preview if settings are invalid
            return
            
        self.shutdown_event.clear()
        
        # Re-create communication primitives
        self.results_queue = mp.Queue(maxsize=10)
        self.active_queue_ref.set(None) # Manager will set this
        
        # Reset stats
        self.frames_captured_counter.value = 0
        self.frames_enqueued_counter.value = 0
        self.frames_processed_counter.value = 0
        self.csv_write_counter.value = 0
        self.last_csv_count = 0
        self.last_stats_time = time.monotonic()
        self.stats_timer.start()
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Initializing model and camera...")
        self.setWindowTitle("DLC Live - (ANALYSIS RUNNING)")
        
        try:
            # 1. Start Video Thread
            self.video_processing_thread = VideoProcessingThread(
                self.settings, self.active_queue_ref, self.shutdown_event, 
                self.camera_command_queue, self.frames_captured_counter, 
                self.frames_enqueued_counter
            )
            
            # 2. Start Inference Process Manager Thread
            self.inference_manager_thread = InferenceProcessManager(
                self.settings, self.active_queue_ref, self.results_queue, 
                self.shutdown_event, self.frames_processed_counter, 
                self.csv_write_counter
            )
            
            # 3. Start GUI Update Worker Thread
            self.gui_update_worker = GuiUpdateWorker(
                self.results_queue, self.shutdown_event, self.settings['target_fps']
            )
            self.gui_update_worker.new_frame_ready.connect(self.update_video_feed)
            
            # Start all threads
            self.video_processing_thread.start()
            self.inference_manager_thread.start()
            self.gui_update_worker.start()
            
            self.statusBar().showMessage("Running analysis.")
            
        except Exception as e:
            QMessageBox.critical(self, "Start Error", f"Failed to start workers: {e}")
            self.on_stop_button_clicked()

    @pyqtSlot()
    def on_stop_button_clicked(self):
        """
        Slot for the "Stop Analysis" button.
        
        - Stops the analysis.
        - Restarts the preview mode.
        - Resets stats labels.
        """
        self.stop_analysis()
        
        # Restart the preview mode
        self.start_preview()
        
        # Reset displays
        self.proc_lbl.setText("Processed (Awaiting Analysis)")
        self.lat_lbl.setText("Inf: -- | Cap-CSV: -- | E2E: -- | RAM: --")
        self.st_lbl.setText("Cap: 0 | Enq: 0 | Proc: 0 | Drop: 0 | CSV FPS: 0.0")

    def stop_analysis(self):
        """
        Performs a graceful shutdown of all analysis threads and processes
        in the correct order.
        """
        if not self.stop_btn.isEnabled(): return # Already stopped
        
        self.statusBar().showMessage("Shutting down analysis...")
        self.stats_timer.stop()
        
        # 1. Signal all threads/processes to stop
        self.shutdown_event.set()
        
        # 2. Stop the GUI worker first (stops pulling from results_queue)
        if self.gui_update_worker and self.gui_update_worker.isRunning():
            self.gui_update_worker.new_frame_ready.disconnect()
            self.gui_update_worker.wait(2000)
        self.gui_update_worker = None
        
        # 3. Stop the Inference Manager (stops the inference process)
        if self.inference_manager_thread and self.inference_manager_thread.is_alive():
            self.inference_manager_thread.join(timeout=7.0) # Give it time
        self.inference_manager_thread = None
        
        # 4. Stop the Video Thread (stops capture)
        if self.video_processing_thread and self.video_processing_thread.is_alive():
            self.video_processing_thread.join(timeout=3.0)
        self.video_processing_thread = None
        
        # 5. Release video writer if recording
        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception as e:
                print(f"!!! ERROR: Failed to release video writer: {e}")
            self.video_writer = None
            if self.rec_btn.isChecked():
                self.rec_btn.setChecked(False); self.rec_btn.setText("Record")
        
        # 6. Clean up queues
        self._drain_queue_mp(self.results_queue)
        if self.results_queue:
            try:
                self.results_queue.close()
                self.results_queue.join_thread()
            except Exception:
                pass # Ignore errors on close
            self.results_queue = None
            
        self.active_queue_ref.set(None)
        
        # Update UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Analysis stopped. Preview mode.")

    def _drain_queue_mp(self, queue):
        """
        Safely empties a multiprocessing queue.
        
        Args:
            queue (mp.Queue): The queue to drain.
        """
        if not queue: return
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break

    # ==========================================================================
    # FRAME HANDLING & MISC
    # ==========================================================================

    @pyqtSlot(dict)
    def update_video_feed(self, packet):
        """
        The main slot for updating the GUI during analysis.
        Receives the processed packet from the `GuiUpdateWorker`.
        
        - Calculates and displays latencies.
        - Draws skeleton/overlays on the processed frame.
        - Writes frame to video file if recording.
        - Updates both 'Original' and 'Processed' QLabels.
        
        Args:
            packet (dict): The data packet containing frames, predictions,
                           and timestamps.
        """
        if self.shutdown_event.is_set(): return

        # --- 1. Calculate Latencies ---
        ts, ts['display_start'] = packet.get('timestamps', {}), time.monotonic()
        lat = {}
        lat['PreProc'] = (ts.get('processed', 0) - ts.get('capture', 0)) * 1000
        lat['InfQWait'] = (ts.get('dequeued_for_inference', 0) - ts.get('enqueued_for_inference', 0)) * 1000 if ts.get('enqueued_for_inference', -1) > 0 else 0
        lat['Infer'] = packet.get('inference_time_ms', 0)
        lat['GUIQWait'] = (ts.get('dequeued_for_gui', 0) - ts.get('enqueued_for_gui', 0)) * 1000 if ts.get('enqueued_for_gui', -1) > 0 else 0
        lat['GUIUpdate'] = (ts.get('display_start', 0) - ts.get('dequeued_for_gui', 0)) * 1000 if ts.get('dequeued_for_gui', -1) > 0 else 0
        lat['CapToCSV'] = packet.get('capture_to_csv_ms', 0)
        lat['E2E'] = (ts.get('display_start', 0) - ts.get('capture', 0)) * 1000
        
        ram = psutil.Process(os.getpid()).memory_info().rss / (1024**3) if PSUTIL_AVAILABLE else -1
        self.lat_lbl.setText(f"Inf: {lat['Infer']:.1f} | Cap-CSV: {lat['CapToCSV']:.1f} | E2E: {lat['E2E']:.1f} | RAM: {ram:.2f} GB")

        # --- 2. Prepare Original Frame ---
        orig_frame = packet['original_frame']
        self.latest_original_frame = orig_frame
        display_orig_frame = orig_frame.copy()
        
        # Draw crop box on original
        if self.crop_cb.isChecked():
            try:
                x, y, w, h = self.settings['crop_x'], self.settings['crop_y'], self.settings['crop_w'], self.settings['crop_h']
                cv2.rectangle(display_orig_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) 
            except Exception: pass

        # --- 3. Prepare Processed Frame ---
        proc_frame = packet['processed_frame']
        display_frame = proc_frame.copy()
        predictions = packet.get('predictions')
        
        # Draw skeleton
        if predictions and self.settings['show_skeleton']:
            draw_skeleton(display_frame, predictions, self.settings['skeleton_confidence'], self.settings['point_confidence'])
        
        # Draw latency overlay
        display_frame = draw_latency_overlay(display_frame, lat)
        self.latest_processed_frame = display_frame.copy()

        # --- 4. Handle Video Recording ---
        if self.video_writer and self.rec_btn.isChecked():
            try:
                # Ensure frame size matches video writer's expected size
                h_disp, w_disp = display_frame.shape[:2]
                rec_w = int(self.video_writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                rec_h = int(self.video_writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                frame_to_write = display_frame
                if h_disp != rec_h or w_disp != rec_w:
                    frame_to_write = cv2.resize(display_frame, (rec_w, rec_h))
                    
                self.video_writer.write(frame_to_write)
            except Exception as e:
                self.toggle_recording() # Stop recording
                QMessageBox.warning(self, "Recording Error", f"Error writing video frame: {e}\nRecording stopped.")
        
        # --- 5. Update GUI Labels ---
        self.orig_lbl.setPixmap(self.to_pixmap(display_orig_frame, self.orig_lbl))
        self.proc_lbl.setPixmap(self.to_pixmap(display_frame, self.proc_lbl))

    def to_pixmap(self, frame, target_label):
        """
        Converts an OpenCV (BGR) frame to a QPixmap scaled to fit the
        target QLabel.

        Args:
            frame (np.ndarray): The OpenCV BGR frame.
            target_label (QLabel): The QLabel to scale the pixmap to.

        Returns:
            QPixmap: A scaled pixmap, or an empty pixmap on error.
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            if h <= 0 or w <= 0: return QPixmap()
            
            q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img.copy())
            
            if target_label and not pixmap.isNull():
                # Scale pixmap to fit label while maintaining aspect ratio
                return pixmap.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pixmap
            
        except Exception as e:
            return QPixmap() # Return empty pixmap on failure

    def take_photo(self):
        """Saves the current original and processed frames as JPGs."""
        if self.latest_original_frame is None:
            QMessageBox.warning(self, "Error", "No frame captured yet.")
            return
            
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        path_o = f'original_{ts}.jpg'
        path_p = f'processed_{ts}.jpg'
        
        try:
            cv2.imwrite(path_o, self.latest_original_frame)
            msg = f"Saved: {path_o}"
            
            # Save processed frame if analysis is running
            if self.latest_processed_frame is not None:
                cv2.imwrite(path_p, self.latest_processed_frame)
                msg += f"\n- {path_p}"
            # Save cropped frame if in preview mode
            elif self.crop_cb.isChecked():
                try:
                    x,y,w,h = (int(self.crop_x_edit.text()), int(self.crop_y_edit.text()),
                               int(self.crop_w_edit.text()), int(self.crop_h_edit.text()))
                    cropped_frame = self.latest_original_frame[y:y+h, x:x+w]
                    cv2.imwrite(path_p, cropped_frame)
                    msg += f"\n- {path_p} (cropped)"
                except Exception:
                    pass # Failed to save cropped preview
                    
            QMessageBox.information(self, "Photo Saved", msg)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save photo: {e}")

    def toggle_recording(self):
        """
        Starts or stops saving the processed video feed to a file.
        """
        # Cannot record in preview mode
        if self.preview_worker is not None:
             QMessageBox.warning(self, "Error", "Cannot record during preview mode. Start the analysis first.")
             self.rec_btn.setChecked(False); return
        
        if self.rec_btn.isChecked(): # Start recording
            path = self.settings.get('video_output_path')
            if not path:
                QMessageBox.warning(self, "Error", "Set a video output path in Group 8 first.")
                self.rec_btn.setChecked(False); return
            if self.latest_original_frame is None:
                QMessageBox.warning(self, "Error", "No video feed to record.")
                self.rec_btn.setChecked(False); return
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Get correct FPS
            if self.settings['camera_source'] == "Video File":
                try:
                    cap = cv2.VideoCapture(self.settings['path'])
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if fps <= 0: fps = 30.0
                except Exception: fps = 30.0
            else:
                fps = float(self.settings.get('cam_fps', 30))
                
            # Get correct frame size (cropped or full)
            if self.settings.get('crop_enabled', False):
                try:
                    size = (self.settings['crop_w'], self.settings['crop_h'])
                    if size[0] <= 0 or size[1] <= 0: raise ValueError("Invalid crop size")
                except Exception:
                     QMessageBox.warning(self, "Error", "Invalid crop dimensions for recording.")
                     self.rec_btn.setChecked(False); return
            else:
                h, w = self.latest_original_frame.shape[:2]
                size = (w, h)
            
            # Create the writer
            try:
                self.video_writer = cv2.VideoWriter(path, fourcc, fps, size)
                if not self.video_writer.isOpened():
                    raise IOError("VideoWriter failed to open.")
                self.rec_btn.setText("Stop Rec")
            except Exception as e:
                QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {e}")
                self.rec_btn.setChecked(False)
                if self.video_writer: self.video_writer.release()
                self.video_writer = None
                
        else: # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.rec_btn.setText("Record")

    def closeEvent(self, event):
        """
        Overrides QMainWindow.closeEvent.
        Ensures all threads and processes are stopped gracefully
        before the application exits.
        
        Args:
            event (QCloseEvent): The close event.
        """
        self.stop_analysis()
        self.stop_preview()
        
        # Clean up the command queue
        try:
            self.camera_command_queue.close()
            self.camera_command_queue.join_thread()
        except Exception:
            pass
            
        event.accept()
