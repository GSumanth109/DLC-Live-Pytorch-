"""
utils/config.py

Configuration and settings management functions.

Handles:
- Loading and saving the `rt_config.json` file.
- Gathering settings from the UI widgets.
- Validating settings before starting the analysis.
"""

import os
import json
from PyQt5.QtWidgets import QMessageBox

CONFIG_FILE_NAME = "rt_config.json"

def derive_pytorch_config_path(snapshot_path):
    """
    Infers the `pytorch_config.yaml` path from the snapshot path.
    Assumes it's in the same directory.
    
    Args:
        snapshot_path (str): Path to the .pt or .pb model file.

    Returns:
        str: The inferred path to the pytorch_config.yaml.
    """
    return os.path.join(os.path.dirname(snapshot_path), 'pytorch_config.yaml') if snapshot_path else ""

def load_settings_from_file():
    """
    Loads settings from the `CONFIG_FILE_NAME`.

    Returns:
        dict or None: The loaded settings dictionary, or None on failure.
    """
    if not os.path.exists(CONFIG_FILE_NAME):
        return None
    try:
        with open(CONFIG_FILE_NAME, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        print(f"!!! [Config] Error loading '{CONFIG_FILE_NAME}': {e}")
        return None

def save_settings_to_file(settings_dict):
    """
    Saves the provided settings dictionary to `CONFIG_FILE_NAME`.

    Args:
        settings_dict (dict): The dictionary of settings to save.
    """
    try:
        with open(CONFIG_FILE_NAME, 'w') as f:
            json.dump(settings_dict, f, indent=4)
    except Exception as e:
        QMessageBox.warning(None, "Config Error", f"Could not save settings to {CONFIG_FILE_NAME}:\n{e}")

def gather_preview_settings(ui):
    """
    Gathers only the essential settings needed to start the camera preview.

    Args:
        ui (App): The main application window instance.

    Returns:
        dict or None: A dictionary of preview settings, or None on failure.
    """
    try:
        if ui.usb_rb.isChecked():
            source_type = 'USB Webcam'
            path = ui.src_edit.text()
        else:
            source_type = 'Video File'
            path = ui.src_file_edit.text()
            
        settings = {
            'camera_source': source_type,
            'path': path, # Use one key for path/index
            'cam_width': int(ui.w_edit.text()),
            'cam_height': int(ui.h_edit.text()),
            'cam_fps': int(ui.fps_edit.text()),
            'exposure': ui.exp_sld.value(),
            'gain': ui.gain_sld.value(),
            'white_balance': ui.wb_sld.value(),
        }
        return settings
    except ValueError as e:
        QMessageBox.critical(ui, "Settings Error", f"Invalid numeric value in camera settings: {e}")
        return None

def gather_settings(ui):
    """
    Gathers ALL settings from the UI fields for the main analysis.
    Performs full validation and shows error popups.

    Args:
        ui (App): The main application window instance.

    Returns:
        dict or None: A complete, validated settings dictionary, or
                      None if validation fails.
    """
    try:
        settings = gather_preview_settings(ui)
        if settings is None:
            return None
            
        # Add all analysis-specific settings
        settings.update({
            'config_path': ui.cfg_edit.text(),
            'snapshot_path': ui.snap_edit.text(),
            'pytorch_config_path': derive_pytorch_config_path(ui.snap_edit.text()),
            'method': ui.pre_cmb.currentText(),
            'flat_image_path': ui.flat_edit.text(),
            'batch_size': 1, # Hardcoded
            'use_fp16': ui.fp16_cb.isChecked(),
            'ram_threshold_gb': ui.ram_sld.value(),
            'target_fps': ui.disp_fps_sld.value(),
            'crop_enabled': ui.crop_cb.isChecked(),
            'crop_x': int(ui.crop_x_edit.text()),
            'crop_y': int(ui.crop_y_edit.text()),
            'crop_w': int(ui.crop_w_edit.text()),
            'crop_h': int(ui.crop_h_edit.text()),
            'skeleton_confidence': ui.skel_sld.value() / 100.0,
            'point_confidence': ui.pt_sld.value() / 100.0,
            'show_skeleton': ui.show_skel_cb.isChecked(),
            'save_csv': ui.csv_cb.isChecked(),
            'csv_output_path': ui.csv_edit.text(),
            'save_video': ui.vid_cb.isChecked(),
            'video_output_path': ui.vid_edit.text(),
        })

        # --- Full Validation ---
        if not all([settings['config_path'], settings['snapshot_path']]):
            # Allow running without a model *only* if processing a video file
            # (e.g., to test pre-processing or recording)
            if not (settings['camera_source'] == "Video File"):
                 QMessageBox.critical(ui, "Settings Error", "DLC Config and Snapshot paths are required to start analysis.")
                 return None
            
        if (settings['config_path'] and not os.path.exists(settings['config_path'])):
            QMessageBox.critical(ui, "Settings Error", f"Config file not found:\n{settings['config_path']}")
            return None
        if (settings['snapshot_path'] and not os.path.exists(settings['snapshot_path'])):
            QMessageBox.critical(ui, "Settings Error", f"Snapshot file not found:\n{settings['snapshot_path']}")
            return None
        if settings['camera_source'] == "Video File" and not os.path.exists(settings['path']):
            QMessageBox.critical(ui, "Settings Error", f"Video file not found:\n{settings['path']}")
            return None
        if settings['save_csv'] and not settings['csv_output_path']:
             QMessageBox.critical(ui, "Settings Error", "CSV output path is required to save CSV.")
             return None
        if settings['save_video'] and not settings['video_output_path']:
             QMessageBox.critical(ui, "Settings Error", "Video output path is required to save video.")
             return None
        if settings['crop_enabled']:
            if settings['crop_w'] <= 0 or settings['crop_h'] <= 0:
                QMessageBox.critical(ui, "Settings Error", "Crop Width (W) and Height (H) must be greater than 0.")
                return None
            if settings['crop_x'] < 0 or settings['crop_y'] < 0:
                QMessageBox.critical(ui, "Settings Error", "Crop X and Y cannot be negative.")
                return None

        ui.bs_sld.setValue(1) # Ensure batch size is 1
        
        # Save the settings (using the UI-facing dict) on successful validation
        save_settings_to_file(ui.get_ui_settings_as_dict())
        
        return settings

    except ValueError as e:
        QMessageBox.critical(ui, "Settings Error", f"Invalid numeric value in settings (e.g., Crop W/H): {e}")
        return None
    except Exception as e:
        QMessageBox.critical(ui, "Settings Error", f"An error occurred while gathering settings: {e}")
        return None
