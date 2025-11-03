"""
gui/preview_worker.py

Defines the `PreviewWorker` QThread.

This is a lightweight thread used *only* for the "Preview Mode".
Its sole purpose is to grab frames from the camera (or loop a video)
and emit them to the GUI, without any heavy processing.
"""

import cv2
import time
import os
from PyQt5.QtCore import QThread, pyqtSignal

class PreviewWorker(QThread):
    """
    A lightweight QThread to grab frames for preview.
    - For Camera: Grabs live frames and applies realtime settings.
    - For Video File: Loops the video file.
    
    Signals:
        new_frame (pyqtSignal): Emits the new frame (np.ndarray)
        video_dimensions_ready (pyqtSignal): Emits (width, height) when
                                             a video file is opened.
    """
    new_frame = pyqtSignal(object) 
    video_dimensions_ready = pyqtSignal(int, int)

    def __init__(self, settings, command_queue):
        """
        Args:
            settings (dict): The "preview settings" dictionary.
            command_queue (mp.Queue): The queue for sending realtime
                                      commands (e.g., exposure) to the camera.
        """
        super().__init__()
        self.settings = settings
        self.command_queue = command_queue
        self.cap = None
        self._running = True

    def run(self):
        """
        Main loop for the preview thread.
        
        Initializes the camera or video file. In the loop, it reads
        a frame, emits it, and sleeps to match the source's FPS.
        """
        source = self.settings['path']
        is_video_file = self.settings['camera_source'] == "Video File"
        
        try:
            # --- Initialization ---
            if is_video_file:
                if not os.path.exists(source):
                    raise IOError(f"Video file not found: {source}")
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    raise IOError(f"Cannot open video file: {source}")
                
                # Emit video dimensions so the GUI can update
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_dimensions_ready.emit(w, h)
                
            else: # USB Webcam
                try:
                    source = int(source)
                except ValueError:
                    source = 0 # Default to 0
                
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(source)
                    if not self.cap.isOpened():
                        raise IOError(f"Cannot open video source: {source}")
                
                # Apply camera settings
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings['cam_width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['cam_height'])
                self.cap.set(cv2.CAP_PROP_FPS, self.settings['cam_fps'])
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings['exposure'])
                self.cap.set(cv2.CAP_PROP_GAIN, self.settings['gain'])
                self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.settings['white_balance'])

            # Get target FPS for sleeping
            if is_video_file:
                target_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if target_fps <= 0: target_fps = 30
            else:
                target_fps = self.settings.get('cam_fps', 30)
            
            target_interval = 1.0 / target_fps
            
        except Exception as e:
            print(f"!!! [PreviewWorker] CRITICAL INIT ERROR: {e}")
            self._running = False
            return

        # --- Main Preview Loop ---
        while self._running:
            start_time = time.monotonic()
            
            if not is_video_file:
                self._apply_camera_commands()
            
            ret, frame = self.cap.read()
            
            if not ret:
                if is_video_file:
                    # Loop the video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Camera disconnected
                    break
            
            self.new_frame.emit(frame.copy())
            
            # Sleep to match target FPS
            elapsed = time.monotonic() - start_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if self.cap:
            self.cap.release()

    def _apply_camera_commands(self):
        """
        Checks the command queue and applies any pending camera
        settings (exposure, gain, etc.)
        """
        try:
            while self.command_queue and not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()
                if 'property' in cmd and 'value' in cmd and self.cap:
                    self.cap.set(cmd['property'], cmd['value'])
        except Exception:
            pass # Ignore queue errors

    def stop(self):
        """Signals the `run` loop to exit."""
        self._running = False
