# Updated: gui/main_window.py
"""
Handles the main application window (GUI) for the DLC Live application.

This file defines the `App` class, which is a QMainWindow that:
- Creates and arranges all UI widgets (buttons, sliders, video feeds).
- Manages application state (preview mode vs. analysis mode).
- Loads and saves configuration settings from a JSON file.
- Connects UI elements to their corresponding functions (e.g., button clicks).
- Starts, stops, and coordinates all background threads and processes.
- Displays the original and processed video feeds.
- Handles user actions like taking photos.
- **MODIFIED**: Implements an automatic video recording system that
  starts on the first inferred frame and stops on the last.
- Displays real-time performance statistics.
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

# --- Psutil for memory monitoring ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: `psutil` not found. RAM monitoring will be disabled.")

# --- Local Module Imports ---
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
    The main application window class.
    
    This class orchestrates the entire application, from UI setup to managing
    background processing threads for video capture and inference.
    """
    
    def __init__(self):
        """
        Initializes the main application window.
        Sets up state variables, synchronization primitives, counters,
        and timers before building the UI and loading the config.
        """
        super().__init__()
        self.setWindowTitle("DLC Live - (Preview Mode)")
        
        # --- Application State ---
        self.settings = {}  # Holds all runtime settings gathered from the UI
        self.latest_original_frame = None # Caches the last frame from camera
        self.latest_processed_frame = None # Caches the last frame with overlays
        
        # --- MODIFIED: Video Recording State ---
        self.video_writer = None 
        self.recording_size = None # (width, height) tuple for the writer
        # --- End Modification ---

        self._slider_busy = False # Flag to prevent slider/textbox update loops

        # --- Threads & Processes ---
        self.video_processing_thread = None # Reads frames from source
        self.inference_manager_thread = None # Manages the inference process
        self.gui_update_worker = None # Updates GUI with results at target FPS
        self.preview_worker = None # Lightweight thread for live preview
        
        # --- Multiprocessing & Threading Synchronization ---
        self.shutdown_event = mp.Event() # Signals all threads/processes to stop
        self.results_queue = None       # mp.Queue for inference results (Proc -> GUI)
        self.camera_command_queue = mp.Queue() # mp.Queue for realtime cam settings (GUI -> Video/Preview)
        self.active_queue_ref = ActiveQueueReference() # Thread-safe ref to inference input queue
        
        # --- Shared Performance Counters ---
        self.frames_captured_counter = mp.Value('i', 0) # Frames read from source
        self.frames_enqueued_counter = mp.Value('i', 0) # Frames sent to inference
        self.frames_processed_counter = mp.Value('i', 0) # Frames out of inference
        self.csv_write_counter = mp.Value('i', 0) # Rows written to CSV
        
        # --- Stats Tracking ---
        self.last_csv_count = 0
        self.last_stats_time = time.monotonic()
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(500) # Update stats 2x per second
        self.stats_timer.timeout.connect(self.update_stats_display)

        # --- Build the UI ---
        self.create_widgets()
        
        # --- Load Config and Start ---
        self.load_app_config() # Load rt_config.json
        self.update_camera_source_ui() # Show/hide UI elements based on config
        
        # Start the camera preview shortly after app opens
        QTimer.singleShot(100, self.start_preview)

    # ==========================================================================
    # UI CREATION (Modified for new layout)
    # ==========================================================================

    def create_widgets(self):
        """Creates and arranges all widgets in the main window."""
        self.create_toolbar()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout: [Controls (Left) | Video Feeds (Right)]
        main_layout = QHBoxLayout(main_widget)

        # --- Video Feeds Layout (RIGHT side) ---
        # This layout stacks the two video feeds vertically.
        vid_layout = QVBoxLayout()
        main_layout.addLayout(vid_layout, 1) # Video layout takes expanding space
        
        vid_group = QGroupBox('Feeds')
        vid_layout.addWidget(vid_group, 1)
        
        # This box holds the two labels vertically.
        vid_box = QVBoxLayout(vid_group) 
        self.orig_lbl = QLabel('Original (Live Preview)')
        self.proc_lbl = QLabel('Processed (Awaiting Analysis)')
        
        for lbl in [self.orig_lbl, self.proc_lbl]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 240)
            # Set expanding policy so they fill the available space
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setScaledContents(False) # We handle scaling via to_pixmap
            vid_box.addWidget(lbl) # Add labels vertically
            
        # --- Controls Layout (LEFT side) ---
        # This layout holds the two columns of parameter groups.
        ctrl_container_layout = QHBoxLayout()
        main_layout.addLayout(ctrl_container_layout) # Add controls to main layout
        
        ctrl_col_1_layout = QVBoxLayout()
        ctrl_col_2_layout = QVBoxLayout()
        ctrl_container_layout.addLayout(ctrl_col_1_layout)
        ctrl_container_layout.addLayout(ctrl_col_2_layout)

        # --- Column 1: Groups 1-5 ---
        
        # Group 1: Input Model
        g1 = QGroupBox("1. Input Model")
        ctrl_col_1_layout.addWidget(g1)
        f1 = QFormLayout(g1)
        self.cfg_edit = self._create_file_input(f1, "Config:", file_filter="YAML Config (*.yaml)")
        self.snap_edit = self._create_file_input(f1, "Snapshot:", file_filter="Model Files (*.pt *.pb)")

        # Group 2: Source
        g2 = QGroupBox("2. Source")
        ctrl_col_1_layout.addWidget(g2)
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

        # Group 3: Cam Params
        self.cam_grp = QGroupBox("3. Cam Params")
        ctrl_col_1_layout.addWidget(self.cam_grp)
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
        
        # Connect W/H edits to update cropping slider ranges
        self.w_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.h_edit.textChanged.connect(self._update_crop_slider_ranges)

        # Group 4: Realtime
        self.rt_grp = QGroupBox("4. Realtime")
        ctrl_col_1_layout.addWidget(self.rt_grp)
        f4 = QFormLayout(self.rt_grp)
        self.exp_sld = self._create_slider(f4, "Exp:", -13, -1, -11, self.update_exposure)
        self.gain_sld = self._create_slider(f4, "Gain:", 0, 128, 0, self.update_gain)
        self.wb_sld = self._create_slider(f4, "WB:", 2000, 8000, 4000, self.update_white_balance)

        # Group 5: Cropping
        g5 = QGroupBox("5. Cropping")
        ctrl_col_1_layout.addWidget(g5)
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
        self._connect_crop_widgets() # Link sliders and text boxes
        
        ctrl_col_1_layout.addStretch() # Pushes groups to the top

        # --- Column 2: Groups 6-10 & Buttons ---
        
        # Group 6: Pre-processing
        g6 = QGroupBox("6. Pre-processing")
        ctrl_col_2_layout.addWidget(g6)
        f6 = QFormLayout(g6)
        self.pre_cmb = QComboBox()
        self.pre_cmb.addItems(["None", "Flat-field", "Morphological Opening"])
        self.pre_cmb.currentTextChanged.connect(self.update_preprocessing_ui)
        f6.addRow("Method:", self.pre_cmb)
        self.flat_edit = self._create_file_input(f6, "Flat Img:", file_filter="Image Files (*.png *.jpg *.bmp *.tif)")
        self.flat_edit.parentWidget().setVisible(False) # Hidden by default

        # Group 7: Performance & Display
        g7 = QGroupBox("7. Performance & Display")
        ctrl_col_2_layout.addWidget(g7)
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
        ctrl_col_2_layout.addWidget(g8)
        f8 = QFormLayout(g8)
        self.csv_cb = QCheckBox("Save CSV")
        self.csv_edit = self._create_file_input(f8, self.csv_cb, is_open=False, file_filter="CSV files (*.csv)")
        
        # --- MODIFIED: Re-added Save Video UI ---
        self.vid_cb = QCheckBox("Save Drawn Video")
        self.vid_edit = self._create_file_input(f8, self.vid_cb, is_open=False, file_filter="Video Files (*.mp4)")
        # --- End Modification ---

        # Group 9: Capture
        g9 = QGroupBox("9. Capture")
        ctrl_col_2_layout.addWidget(g9)
        h9 = QHBoxLayout(g9)
        self.pht_btn = QPushButton("Photo")
        self.pht_btn.clicked.connect(self.take_photo)
        # --- MODIFIED: Removed Record button, as it's now automatic ---
        h9.addWidget(self.pht_btn)
        # --- End Modification ---

        # Group 10: Stats
        g10 = QGroupBox("10. Stats")
        ctrl_col_2_layout.addWidget(g10)
        v10 = QVBoxLayout(g10)
        self.st_lbl = QLabel("Cap: 0 | Enq: 0 | Proc: 0 | Drop: 0 | CSV FPS: 0.0")
        self.st_lbl.setAlignment(Qt.AlignLeft)
        v10.addWidget(self.st_lbl)
        
        # Start/Stop Buttons
        h_start = QHBoxLayout()
        self.start_btn = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Analysis")
        self.stop_btn.clicked.connect(self.on_stop_button_clicked)
        self.start_btn.setEnabled(False) # Disabled until preview is running
        self.stop_btn.setEnabled(False)  # Disabled until analysis starts
        h_start.addWidget(self.start_btn)
        h_start.addWidget(self.stop_btn)
        ctrl_col_2_layout.addLayout(h_start)
        
        ctrl_col_2_layout.addStretch() # Pushes groups to the top

        # --- Status Bar ---
        self.lat_lbl = QLabel("Inf: -- | Cap-CSV: -- | E2E: -- | RAM: --")
        self.statusBar().addPermanentWidget(self.lat_lbl)
        self.statusBar().showMessage("Initializing preview...")

    def create_toolbar(self):
        """Creates the top application toolbar for file operations."""
        tb = self.addToolBar('File')
        
        # Save Settings Action
        save_action = QAction(QIcon.fromTheme("document-save"), 'Save Settings', self)
        save_action.triggered.connect(self.save_app_config)
        tb.addAction(save_action)
        
        tb.addSeparator()
        
        # Set Flat-field Image Action
        o = QAction('Set Flat-field Image', self)
        o.triggered.connect(lambda: self.browse_file(self.flat_edit, file_filter="Image Files (*.png *.jpg *.bmp *.tif)"))
        tb.addAction(o)
        
        # Exit Action
        e = QAction(QIcon.fromTheme("application-exit"), 'Exit', self)
        e.triggered.connect(self.close)
        tb.addAction(e)

    def _create_file_input(self, layout, label_widget, is_open=True, file_filter="All Files (*)"):
        """
        Helper function to create a text edit + browse button combo.
        """
        edit = QLineEdit()
        btn_txt = "..." if is_open else "Save As"
        btn = QPushButton(btn_txt)
        btn.clicked.connect(lambda: self.browse_file(edit, is_open, file_filter))
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(edit)
        h_layout.addWidget(btn)
        
        if isinstance(label_widget, str):
            layout.addRow(label_widget, h_layout)
        else:
            # Handle label being a QCheckBox for CSV/Vid outputs
            layout.addRow(label_widget, h_layout)
        return edit

    def _create_slider(self, layout, label, min_v, max_v, def_v, func=None):
        """
        Helper function to create a slider + value label combo.
        """
        sld = QSlider(Qt.Horizontal)
        sld.setRange(min_v, max_v)
        sld.setValue(def_v)
        v_lbl = QLabel(f"{def_v}") # Label to show current value
        
        # Connect slider to label
        sld.valueChanged.connect(lambda v, lbl=v_lbl: lbl.setText(f"{v}"))
        
        # Connect slider to optional function (e.g., update_exposure)
        if func:
            sld.valueChanged.connect(func)
            
        h_layout = QHBoxLayout()
        h_layout.addWidget(sld)
        h_layout.addWidget(v_lbl)
        layout.addRow(label, h_layout)
        return sld

    # ==========================================================================
    # CONFIG FILE HANDLING
    # ==========================================================================

    def load_app_config(self):
        """Loads settings from rt_config.json and populates the UI."""
        print("[App] Loading settings from rt_config.json...")
        settings = load_settings_from_file()
        if settings:
            self.populate_ui_from_settings(settings)
        else:
            # If no config, save the current UI defaults to create the file
            print("[App] No config file found. Saving current defaults.")
            self.save_app_config()

    @pyqtSlot()
    def save_app_config(self):
        """Saves the current UI settings to rt_config.json."""
        print("[App] Saving settings to rt_config.json...")
        settings_dict = self.get_ui_settings_as_dict()
        save_settings_to_file(settings_dict)
        self.statusBar().showMessage("Settings saved to rt_config.json.", 3000)

    def get_ui_settings_as_dict(self):
        """
        Gathers all values from the UI widgets into a dictionary.
        This dictionary is what gets saved to rt_config.json.
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
            # --- MODIFIED: Added video save path to config ---
            'save_video': self.vid_cb.isChecked(),
            'video_output_path': self.vid_edit.text()
            # --- End Modification ---
        }
        return settings

    def populate_ui_from_settings(self, settings):
        """
        Sets all UI fields from a loaded settings dictionary.
        """
        # Source
        if settings.get('source_is_usb', True):
            self.usb_rb.setChecked(True)
        else:
            self.file_rb.setChecked(True)
        self.src_edit.setText(str(settings.get('camera_path', '0')))
        self.src_file_edit.setText(str(settings.get('video_file_path', '')))
        
        # Model
        self.cfg_edit.setText(str(settings.get('config_path', '')))
        self.snap_edit.setText(str(settings.get('snapshot_path', '')))
        
        # Cam Params
        self.w_edit.setText(str(settings.get('cam_width', '640')))
        self.h_edit.setText(str(settings.get('cam_height', '480')))
        self.fps_edit.setText(str(settings.get('cam_fps', '60')))
        
        # Realtime
        self.exp_sld.setValue(settings.get('exposure', -11))
        self.gain_sld.setValue(settings.get('gain', 0))
        self.wb_sld.setValue(settings.get('white_balance', 4000))
        
        # Cropping
        self.crop_cb.setChecked(settings.get('crop_enabled', False))
        self.crop_x_edit.setText(str(settings.get('crop_x', '0')))
        self.crop_y_edit.setText(str(settings.get('crop_y', '0')))
        self.crop_w_edit.setText(str(settings.get('crop_w', '640')))
        self.crop_h_edit.setText(str(settings.get('crop_h', '480')))
        
        # Pre-processing
        self.pre_cmb.setCurrentText(str(settings.get('method', 'None')))
        self.flat_edit.setText(str(settings.get('flat_image_path', '')))
        
        # Performance
        self.ram_sld.setValue(settings.get('ram_threshold_gb', 16))
        self.fp16_cb.setChecked(settings.get('use_fp16', True))
        self.disp_fps_sld.setValue(settings.get('target_fps', 60))
        
        # Display
        self.skel_sld.setValue(int(settings.get('skeleton_confidence', 0.1) * 100))
        self.pt_sld.setValue(int(settings.get('point_confidence', 0.6) * 100))
        self.show_skel_cb.setChecked(settings.get('show_skeleton', True))
        
        # Output
        self.csv_cb.setChecked(settings.get('save_csv', False))
        self.csv_edit.setText(str(settings.get('csv_output_path', '')))
        # --- MODIFIED: Load video save path from config ---
        self.vid_cb.setChecked(settings.get('save_video', False))
        self.vid_edit.setText(str(settings.get('video_output_path', '')))
        # --- End Modification ---
        
        print("[App] UI populated from saved settings.")
        
    # ==========================================================================
    # UI SLOTS & CALLBACKS
    # ==========================================================================
    
    def _connect_crop_widgets(self):
        """Connects all crop sliders and text boxes for 2-way updates."""
        # Sliders update text boxes
        self.crop_x_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_x_edit, v))
        self.crop_y_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_y_edit, v))
        self.crop_w_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_w_edit, v))
        self.crop_h_sld.valueChanged.connect(lambda v: self._set_text_from_slider(self.crop_h_edit, v))
        
        # Text boxes update sliders
        self.crop_x_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_x_sld, t))
        self.crop_y_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_y_sld, t))
        self.crop_w_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_w_sld, t))
        self.crop_h_edit.textChanged.connect(lambda t: self._set_slider_from_text(self.crop_h_sld, t))
        
        # Editing any value updates all slider ranges dynamically
        self.crop_x_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.crop_y_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.crop_w_edit.textChanged.connect(self._update_crop_slider_ranges)
        self.crop_h_edit.textChanged.connect(self._update_crop_slider_ranges)

    def _set_text_from_slider(self, text_edit, value):
        """Update text box from slider, avoiding signal loops."""
        if self._slider_busy: return
        self._slider_busy = True
        text_edit.setText(str(value))
        self._slider_busy = False

    def _set_slider_from_text(self, slider, text):
        """Update slider from text box, avoiding signal loops."""
        if self._slider_busy: return
        self._slider_busy = True
        try:
            slider.setValue(int(text))
        except ValueError:
            pass # Ignore invalid text (e.g., empty string)
        self._slider_busy = False

    @pyqtSlot()
    def _update_crop_slider_ranges(self):
        """
        Dynamically adjusts the min/max ranges of the crop sliders
        based on the camera dimensions and other crop values.
        Ensures sliders don't allow selecting an out-of-bounds region.
        """
        if self._slider_busy: return
        try:
            # Get base dimensions
            cam_w, cam_h = int(self.w_edit.text()), int(self.h_edit.text())
            
            # If video file, use actual frame dimensions if available
            if self.file_rb.isChecked() and self.latest_original_frame is not None:
                cam_h, cam_w = self.latest_original_frame.shape[:2]

            # Get current crop values
            x, y = int(self.crop_x_edit.text()), int(self.crop_y_edit.text())
            w, h = int(self.crop_w_edit.text()), int(self.crop_h_edit.text())
            
            # Set dynamic ranges
            # X slider max: (Camera Width - Crop Width)
            self.crop_x_sld.setRange(0, max(0, cam_w - w))
            # Y slider max: (Camera Height - Crop Height)
            self.crop_y_sld.setRange(0, max(0, cam_h - h))
            # W slider max: (Camera Width - Crop X)
            self.crop_w_sld.setRange(1, max(1, cam_w - x))
            # H slider max: (Camera Height - Crop Y)
            self.crop_h_sld.setRange(1, max(1, cam_h - y))
        except ValueError:
            pass # Ignore errors from partial/invalid text

    @pyqtSlot()
    def update_camera_source_ui(self):
        """Shows/hides UI elements based on camera source (USB/File)."""
        is_usb = self.usb_rb.isChecked()
        
        # Toggle visibility of source inputs
        self.src_lbl.setVisible(is_usb)
        self.src_edit.setVisible(is_usb)
        self.src_file_edit.setVisible(not is_usb)
        self.src_browse_btn.setVisible(not is_usb)
        
        # Disable camera-specific controls if using a file
        self.cam_grp.setEnabled(is_usb)
        self.rt_grp.setEnabled(is_usb)

    def browse_file(self, edit_widget, is_open=True, file_filter="All Files (*)"):
        """
        Opens a file dialog to select or save a file.
        """
        if is_open:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save File As", "", file_filter)
        
        if path:
            edit_widget.setText(path)

    @pyqtSlot(str)
    def update_preprocessing_ui(self, method_name):
        """Shows/hides the 'Flat Img' input based on pre-processing method."""
        self.flat_edit.parentWidget().setVisible(method_name == "Flat-field")
    
    @pyqtSlot()
    def on_reload_source(self):
        """Callback for the 'Reload Source' button. Stops/starts preview."""
        print("[App] Reloading source...")
        self.statusBar().showMessage("Restarting preview...")
        self.stop_preview()
        # Short delay to allow camera device to be released
        QTimer.singleShot(100, self.start_preview)

    @pyqtSlot(int)
    def update_exposure(self, value):
        """Sends an 'exposure' command to the camera command queue."""
        if self.camera_command_queue:
            cmd = {'property': cv2.CAP_PROP_EXPOSURE, 'value': float(value)}
            self.camera_command_queue.put(cmd)

    @pyqtSlot(int)
    def update_gain(self, value):
        """Sends a 'gain' command to the camera command queue."""
        if self.camera_command_queue:
            cmd = {'property': cv2.CAP_PROP_GAIN, 'value': float(value)}
            self.camera_command_queue.put(cmd)

    @pyqtSlot(int)
    def update_white_balance(self, value):
        """Sends a 'white balance' command to the camera command queue."""
        if self.camera_command_queue:
            cmd = {'property': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 'value': float(value)}
            self.camera_command_queue.put(cmd)

    @pyqtSlot()
    def update_stats_display(self):
        """
        Periodically called by `stats_timer` to update performance labels.
        Calculates drop, CSV FPS, etc.
        """
        # Only update if analysis is running
        if not self.stop_btn.isEnabled(): return
        
        current_time = time.monotonic()
        time_elapsed = current_time - self.last_stats_time
        if time_elapsed < 0.1: return # Avoid division by zero
        
        # Read shared counters
        cap = self.frames_captured_counter.value
        enq = self.frames_enqueued_counter.value
        proc = self.frames_processed_counter.value
        drop = cap - enq # Frames dropped by Video thread (queue overflow)
        
        # Calculate CSV FPS
        csv_now = self.csv_write_counter.value
        csv_count_diff = csv_now - self.last_csv_count
        csv_fps = csv_count_diff / time_elapsed
        
        # Update text
        txt = (f"Cap:{cap} | Enq:{enq} | Proc:{proc} | Drop(Ovr):{drop} | CSV FPS: {csv_fps:.1f}")
        self.st_lbl.setText(txt)
        
        # Store last values for next calculation
        self.last_csv_count = csv_now
        self.last_stats_time = current_time

    # ==========================================================================
    # CORE APPLICATION LOGIC (Preview)
    # ==========================================================================

    @pyqtSlot()
    def start_preview(self):
        """
        Starts the lightweight preview worker for either USB cam or Video file.
        """
        print("[App] Starting preview...")
        self.stop_preview() # Ensure any old preview is stopped
        
        preview_settings = gather_preview_settings(self)
        if preview_settings is None:
            QMessageBox.critical(self, "Preview Error", "Invalid settings. Cannot start preview.")
            return

        self.proc_lbl.setText("Processed (Awaiting Analysis)")
        self.orig_lbl.setText("Starting preview...")
        self.statusBar().showMessage("Preview running.")
        self.setWindowTitle("DLC Live - (Preview Mode)")
        
        self._update_crop_slider_ranges()
        
        # --- MODIFIED: Use PreviewWorker for both USB and Video File ---
        self.preview_worker = PreviewWorker(preview_settings, self.camera_command_queue)
        self.preview_worker.new_frame.connect(self.update_preview_feed)
        self.preview_worker.video_dimensions_ready.connect(self.on_video_dimensions_ready)
        self.preview_worker.start()
        # --- End Modification ---
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(int, int)
    def on_video_dimensions_ready(self, width, height):
        """
        Slot to receive video dimensions from the PreviewWorker.
        This updates the UI and crop sliders when a video file is loaded.
        """
        if self.file_rb.isChecked():
            print(f"[App] Received video dimensions: {width}x{height}")
            self.w_edit.setText(str(width))
            self.h_edit.setText(str(height))
            self._update_crop_slider_ranges() # Update sliders with new ranges

    def stop_preview(self):
        """Stops the preview worker thread if it's running."""
        if self.preview_worker:
            print("[App] Stopping preview worker...")
            self.preview_worker.stop()
            self.preview_worker.wait(2000) # Wait for thread to finish
            if self.preview_worker.isRunning():
                print("!!! [App] WARN: Preview worker did not stop gracefully.")
            self.preview_worker = None
            print("[App] Preview worker stopped.")

    @pyqtSlot(object)
    def update_preview_feed(self, frame):
        """
        Slot to update the 'Original' feed label with a new frame
        from the PreviewWorker.
        """
        self.latest_original_frame = frame
        self.latest_processed_frame = None # Clear processed frame
        
        display_frame = frame.copy()
        
        # Draw crop box overlay if enabled
        if self.crop_cb.isChecked():
            try:
                x, y = int(self.crop_x_edit.text()), int(self.crop_y_edit.text())
                w, h = int(self.crop_w_edit.text()), int(self.crop_h_edit.text())
                
                # Ensure crop box is valid
                h_img, w_img = display_frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2) 
            except ValueError:
                pass # Ignore errors from partial/invalid text
                
        self.orig_lbl.setPixmap(self.to_pixmap(display_frame, self.orig_lbl))

    # ==========================================================================
    # CORE APPLICATION LOGIC (Analysis)
    # ==========================================================================

    def start_analysis(self):
        """
        Stops the preview and starts all background threads/processes
        for the full analysis pipeline.
        
        --- MODIFIED ---
        This now also initializes the cv2.VideoWriter if "Save Video" is checked.
        """
        self.stop_preview() # Stop lightweight preview
        
        # Gather and validate all settings
        self.settings = gather_settings(self)
        if not self.settings:
            self.start_preview() # Restart preview if settings are invalid
            return
            
        print("[App] Starting analysis...")
        
        # --- MODIFIED: Initialize VideoWriter ---
        self.video_writer = None # Ensure it's clear from any previous run
        self.recording_size = None # Clear recording size
        
        if self.settings.get('save_video', False):
            path = self.settings.get('video_output_path')
            if not path:
                QMessageBox.warning(self, "Video Save Error", "No output path specified for video. Recording disabled.")
            else:
                try:
                    # Determine frame size
                    if self.settings.get('crop_enabled', False):
                        size = (self.settings['crop_w'], self.settings['crop_h'])
                    else:
                        size = (self.settings['cam_width'], self.settings['cam_height'])
                    
                    # Determine FPS
                    fps = float(self.settings.get('cam_fps', 30))
                    if self.file_rb.isChecked():
                        # For video files, try to get the actual FPS
                        try:
                            cap = cv2.VideoCapture(self.settings['path'])
                            file_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            if file_fps > 0:
                                fps = file_fps
                        except Exception as e:
                            print(f"[App] Warn: Could not get video file FPS: {e}. Defaulting to {fps}.")
                    
                    if size[0] <= 0 or size[1] <= 0:
                        raise ValueError(f"Invalid recording size: {size}")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(path, fourcc, fps, size)
                    
                    if not self.video_writer.isOpened():
                        raise IOError("cv2.VideoWriter failed to open.")
                    
                    # --- FIX: Store the validated size ---
                    self.recording_size = size 
                    
                    print(f"[App] VideoWriter initialized. Saving to: {path} @ {fps} FPS, Size: {size}")
                    
                except Exception as e:
                    QMessageBox.critical(self, "VideoWriter Error", f"Failed to start video recording:\n{e}")
                    self.video_writer = None # Disable recording
                    self.recording_size = None
        # --- End Modification ---
        
        # Reset all synchronization objects and counters
        self.shutdown_event.clear()
        print("[App] Creating new multiprocessing queues...")
        self.results_queue = mp.Queue(maxsize=10)
        self.active_queue_ref.set(None) # Set to None until manager provides a queue
        
        self.frames_captured_counter.value = 0
        self.frames_enqueued_counter.value = 0
        self.frames_processed_counter.value = 0
        self.csv_write_counter.value = 0
        self.last_csv_count = 0
        self.last_stats_time = time.monotonic()
        
        # Start stats timer and update UI state
        self.stats_timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Initializing model and camera...")
        self.setWindowTitle("DLC Live - (ANALYSIS RUNNING)")
        
        try:
            # Start all worker threads/processes
            self.video_processing_thread = VideoProcessingThread(
                self.settings, self.active_queue_ref, self.shutdown_event, 
                self.camera_command_queue, self.frames_captured_counter, 
                self.frames_enqueued_counter
            )
            
            self.inference_manager_thread = InferenceProcessManager(
                self.settings, self.active_queue_ref, self.results_queue, 
                self.shutdown_event, self.frames_processed_counter, 
                self.csv_write_counter
            )
            
            self.gui_update_worker = GuiUpdateWorker(
                self.results_queue, self.shutdown_event, self.settings['target_fps']
            )
            
            self.gui_update_worker.new_frame_ready.connect(self.update_video_feed)
            
            self.video_processing_thread.start()
            self.inference_manager_thread.start()
            self.gui_update_worker.start()
            
            print("[App] All analysis threads started.")
            self.statusBar().showMessage("Running analysis.")
            
        except Exception as e:
            QMessageBox.critical(self, "Start Error", f"Failed to start workers: {e}")
            self.on_stop_button_clicked() # Attempt to clean up

    @pyqtSlot()
    def on_stop_button_clicked(self):
        """
        Slot for the 'Stop Analysis' button.
        Stops the analysis and restarts the preview.
        """
        self.stop_analysis()
        
        # Restart the preview
        self.start_preview()
        
        # Reset UI labels
        self.proc_lbl.setText("Processed (Awaiting Analysis)")
        self.lat_lbl.setText("Inf: -- | Cap-CSV: -- | E2E: -- | RAM: --")
        self.st_lbl.setText("Cap: 0 | Enq: 0 | Proc: 0 | Drop: 0 | CSV FPS: 0.0")

    def stop_analysis(self):
        """
        Stops all analysis threads and processes gracefully.
        
        --- MODIFIED ---
        This now also safely closes the cv2.VideoWriter if it was active.
        """
        if not self.stop_btn.isEnabled(): 
            return # Analysis is not running
            
        print("[App] Stopping analysis...")
        self.statusBar().showMessage("Shutting down analysis...")
        self.stats_timer.stop()
        
        # 1. Signal all threads/processes to shut down
        self.shutdown_event.set()
        
        # 2. Stop GUI worker (QThread)
        if self.gui_update_worker and self.gui_update_worker.isRunning():
            print("[App] Stopping GUI worker...")
            self.gui_update_worker.new_frame_ready.disconnect()
            self.gui_update_worker.wait(2000)
            if self.gui_update_worker.isRunning(): print("!!! WARN: GUI worker did not finish.")
            else: print("[App] GUI worker stopped.")
        self.gui_update_worker = None
        
        # 3. Stop Inference Manager (Python Thread)
        if self.inference_manager_thread and self.inference_manager_thread.is_alive():
            print("[App] Stopping Inference Manager...")
            self.inference_manager_thread.join(timeout=7.0) # Manager handles stopping its child process
            if self.inference_manager_thread.is_alive(): print("!!! WARN: Inference Manager thread did not join.")
            else: print("[App] Inference Manager stopped.")
        self.inference_manager_thread = None
        
        # 4. Stop Video Thread (Python Thread)
        if self.video_processing_thread and self.video_processing_thread.is_alive():
            print("[App] Stopping Video thread...")
            self.video_processing_thread.join(timeout=3.0)
            if self.video_processing_thread.is_alive(): print("!!! WARN: Video thread did not join.")
            else: print("[App] Video thread stopped.")
        self.video_processing_thread = None
        
        # 5. --- MODIFIED: Release VideoWriter ---
        if self.video_writer:
            print("[App] Releasing video writer...")
            try:
                self.video_writer.release()
                print(f"[App] Saved drawn video to: {self.settings.get('video_output_path')}")
            except Exception as e: 
                print(f"!!! ERROR: Failed to release video writer: {e}")
            self.video_writer = None
            self.recording_size = None # Clear stored size
        # --- End Modification ---

        # 6. Clean up queues
        print("[App] Draining analysis queues...")
        self._drain_queue_mp(self.results_queue)
        
        if self.results_queue:
            try:
                self.results_queue.close()
                self.results_queue.join_thread()
                print("[App] Results queue closed.")
            except Exception as e: 
                print(f"!!! WARN: Error closing results queue: {e}")
            self.results_queue = None
            
        self.active_queue_ref.set(None)
        
        # 7. Reset UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        print("[App] Analysis stop routine complete.")

    def _drain_queue_mp(self, queue):
        """Helper to safely empty a multiprocessing queue."""
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
        The main slot for updating video feeds. Receives the data packet
        from the GuiUpdateWorker.
        
        --- MODIFIED ---
        This now automatically writes the drawn frame to the
        cv2.VideoWriter if it exists (no button check needed).
        """
        if self.shutdown_event.is_set(): return
        
        # --- Timestamp and Latency Calculation ---
        ts, ts['display_start'] = packet.get('timestamps', {}), time.monotonic()
        lat = {}
        lat['PreProc'] = (ts.get('processed', 0) - ts.get('capture', 0)) * 1000
        lat['InfQWait'] = (ts.get('dequeued_for_inference', 0) - ts.get('enqueued_for_inference', 0)) * 1000 if ts.get('enqueued_for_inference', -1) > 0 else 0
        lat['Infer'] = packet.get('inference_time_ms', 0)
        lat['GUIQWait'] = (ts.get('dequeued_for_gui', 0) - ts.get('enqueued_for_gui', 0)) * 1000 if ts.get('dequeued_for_gui', -1) > 0 else 0
        lat['GUIUpdate'] = (ts.get('display_start', 0) - ts.get('dequeued_for_gui', 0)) * 1000 if ts.get('dequeued_for_gui', -1) > 0 else 0
        lat['CapToCSV'] = packet.get('capture_to_csv_ms', 0)
        lat['E2E'] = (ts.get('display_start', 0) - ts.get('capture', 0)) * 1000
        ram = psutil.Process(os.getpid()).memory_info().rss / (1024**3) if PSUTIL_AVAILABLE else -1
        self.lat_lbl.setText(f"Inf: {lat['Infer']:.1f} | Cap-CSV: {lat['CapToCSV']:.1f} | E2E: {lat['E2E']:.1f} | RAM: {ram:.2f} GB")
        
        # --- Frame Preparation ---
        orig_frame, proc_frame = packet['original_frame'], packet['processed_frame']
        self.latest_original_frame = orig_frame
        
        # 1. Prepare Original Frame (with crop box)
        display_orig_frame = orig_frame.copy()
        if self.crop_cb.isChecked():
            try:
                x, y, w, h = self.settings['crop_x'], self.settings['crop_y'], self.settings['crop_w'], self.settings['crop_h']
                cv2.rectangle(display_orig_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) 
            except Exception: pass
            
        # 2. Prepare Processed Frame (with drawings)
        display_frame = proc_frame.copy()
        predictions = packet.get('predictions')
        
        if predictions and self.settings['show_skeleton']:
            draw_skeleton(display_frame, predictions, self.settings['skeleton_confidence'], self.settings['point_confidence'])
            
        display_frame = draw_latency_overlay(display_frame, lat)
        
        self.latest_processed_frame = display_frame.copy()

        # 3. --- MODIFIED: Automatic Video Writing ---
        # Use self.recording_size (set in start_analysis) instead of self.video_writer.get()
        if self.video_writer and self.recording_size:
            try:
                # Ensure frame size matches writer
                h_disp, w_disp = display_frame.shape[:2]
                rec_w, rec_h = self.recording_size # <<< FIX: Use stored size
                
                frame_to_write = display_frame
                if h_disp != rec_h or w_disp != rec_w:
                    # Resize if necessary
                    frame_to_write = cv2.resize(display_frame, (int(rec_w), int(rec_h)))
                    
                self.video_writer.write(frame_to_write)
                
            except Exception as e:
                print(f"!!! ERROR: Failed to write video frame: {e}")
                # Don't stop analysis, just stop recording
                self.video_writer.release()
                self.video_writer = None
                self.recording_size = None
                QMessageBox.warning(self, "Recording Error", f"Error writing video frame: {e}\nRecording stopped.")
        # --- End Modification ---

        # 4. Update GUI Labels
        self.orig_lbl.setPixmap(self.to_pixmap(display_orig_frame, self.orig_lbl))
        self.proc_lbl.setPixmap(self.to_pixmap(display_frame, self.proc_lbl))

    def to_pixmap(self, frame, target_label):
        """Converts an OpenCV (BGR) frame to a scaled QPixmap."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            if h <= 0 or w <= 0: return QPixmap()
            
            q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img.copy())
            
            if target_label and not pixmap.isNull():
                # Scale pixmap to fit the label, keeping aspect ratio
                return pixmap.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pixmap
            
        except Exception as e:
            print(f"!!! ERROR: Failed to convert frame to Pixmap: {e}")
            return QPixmap()

    def take_photo(self):
        """Saves the current original and processed frames as JPGs."""
        if self.latest_original_frame is None:
            QMessageBox.warning(self, "Error", "No frame captured yet.")
            return
            
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        path_o, path_p = f'original_{ts}.jpg', f'processed_{ts}.jpg'
        
        try:
            # Save original frame
            cv2.imwrite(path_o, self.latest_original_frame)
            msg = f"Saved: {path_o}"
            
            # Save processed frame (if it exists)
            if self.latest_processed_frame is not None:
                cv2.imwrite(path_p, self.latest_processed_frame)
                msg += f"\n- {path_p}"
            # Fallback: save cropped preview if analysis isn't running
            elif self.crop_cb.isChecked():
                try:
                    x,y,w,h = (int(self.crop_x_edit.text()), int(self.crop_y_edit.text()),
                               int(self.crop_w_edit.text()), int(self.crop_h_edit.text()))
                    cropped_frame = self.latest_original_frame[y:y+h, x:x+w]
                    cv2.imwrite(path_p, cropped_frame)
                    msg += f"\n- {path_p} (cropped preview)"
                except Exception as e:
                    print(f"Could not save cropped photo: {e}")
                    
            QMessageBox.information(self, "Photo Saved", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save photo: {e}")

    def toggle_recording(self):
        """
        --- DEPRECATED ---
        This function is no longer used by the new automatic
        recording logic, but is kept to prevent crashes if
        an old UI element somehow calls it.
        """
        QMessageBox.information(self, "Recording", 
            "Recording is now automatic.\n\n"
            "To record, check 'Save Drawn Video' in Group 8, "
            "set a file path, and click 'Start Analysis'.")

    def closeEvent(self, event):
        """
        Handles the main window close event.
        Ensures all threads and processes are stopped cleanly.
        """
        print("[App] Close event triggered.")
        
        # Stop analysis (which also saves the video)
        self.stop_analysis()
        
        # Stop preview
        self.stop_preview()
        
        # Clean up camera command queue
        try:
            self.camera_command_queue.close()
            self.camera_command_queue.join_thread()
        except Exception as e:
            print(f"Error closing camera queue: {e}")
            
        print("[App] Shutdown complete. Accepting close event.")
        event.accept()
