"""
GUI Preview Worker (QThread)

This file defines the `PreviewWorker`, a lightweight QThread that runs in
the main GUI process. Its purpose is to provide a fast, low-overhead live
preview from the selected video source (USB camera or video file) *before*
the main analysis is started.

This allows the user to:
-   Check camera focus and positioning.
-   Adjust real-time camera parameters (exposure, gain).
-   Visually configure the cropping rectangle.

It runs in a simple loop, grabbing a frame, emitting it via a Qt signal,
and sleeping to match the camera's FPS.
"""

import cv2
import time
import os
from PyQt5.QtCore import QThread, pyqtSignal

class PreviewWorker(QThread):
    """
    A lightweight QThread dedicated to grabbing frames for the preview feed.
    
    Emits:
        new_frame (pyqtSignal): Emits the captured frame as a numpy array.
        video_dimensions_ready (pyqtSignal): Emits the (width, height) of a
                                             video file when it's first opened.
    """
    
    # Signal to emit a new frame (as a numpy array)
    new_frame = pyqtSignal(object) 
    
    # Signal to update UI with video dimensions (for video file source)
    video_dimensions_ready = pyqtSignal(int, int)

    def __init__(self, settings, command_queue):
        """
        Initializes the preview worker.

        Args:
            settings (dict): The dictionary of preview settings
                             (source, path, W, H, FPS, etc.).
            command_queue (mp.Queue): The queue used to send
                                      real-time commands (like exposure changes)
                                      to the camera.
        """
        super().__init__()
        self.settings = settings
        self.command_queue = command_queue
        self.cap = None
        self._running = True # Flag to control the main loop
        print("[PreviewWorker] Initialized.")

    def run(self):
        """
        The main loop for the preview worker thread.
        
        This loop:
        1.  Initializes the `cv2.VideoCapture` object based on settings.
        2.  Sets camera properties (W, H, FPS, exposure, etc.).
        3.  Enters a loop that runs as long as `self._running` is True.
        4.  Inside the loop:
            -   Applies any real-time commands from the `command_queue`.
            -   Grabs a frame from the camera/video.
            -   If it's a video file and it ends, it loops back to the beginning.
            -   Emits the frame via the `new_frame` signal.
            -   Sleeps to maintain the target FPS.
        5.  Cleans up and releases the `VideoCapture` object on exit.
        """
        source = self.settings['path']
        is_video_file = self.settings['camera_source'] == "Video File"
        
        try:
            # --- Initialization ---
            if is_video_file:
                # --- Video File Setup ---
                if not os.path.exists(source):
                    raise IOError(f"Video file not found: {source}")
                print(f"[PreviewWorker] Opening video file: {source}")
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    raise IOError(f"Cannot open video file: {source}")
                
                # Get dimensions and emit them so the UI can update
                # (e.g., set W/H text boxes and crop slider ranges)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_dimensions_ready.emit(w, h)
                
            else:
                # --- USB Camera Setup ---
                try:
                    # Convert source (e.g., "0") to an integer index
                    source = int(source)
                except ValueError:
                    print(f"!!! [PreviewWorker] Invalid Cam Index '{source}'. Defaulting to 0.")
                    source = 0
                
                print(f"[PreviewWorker] Opening video source: {source}")
                # Try FFMPEG backend first for more control
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    print(f"!!! [PreviewWorker] FFMPEG failed. Retrying default...")
                    self.cap = cv2.VideoCapture(source) # Fallback to default
                    if not self.cap.isOpened():
                        raise IOError(f"Cannot open video source: {source}")
                
                print("[PreviewWorker] Camera open. Setting properties...")
                # Set desired camera properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce-latency
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings['cam_width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['cam_height'])
                self.cap.set(cv2.CAP_PROP_FPS, self.settings['cam_fps'])
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings['exposure'])
                self.cap.set(cv2.CAP_PROP_GAIN, self.settings['gain'])
                self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.settings['white_balance'])
                print(f"[PreviewWorker] Set: {self.settings['cam_width']}x{self.settings['cam_height']} @ {self.settings['cam_fps']} FPS")

            # --- Target FPS Calculation ---
            if is_video_file:
                target_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if target_fps <= 0: target_fps = 30 # Default if metadata is missing
            else:
                target_fps = self.settings.get('cam_fps', 30)
            
            # Calculate the time in seconds each frame should take
            target_interval = 1.0 / target_fps
            
        except Exception as e:
            print(f"!!! [PreviewWorker] CRITICAL INIT ERROR: {e}")
            self._running = False
            # We can't emit a signal here because the thread might not be
            # connected yet. The main window will handle this by seeing
            # no frames arrive.
            return

        # --- Main Preview Loop ---
        while self._running:
            start_time = time.monotonic() # Note start time for FPS matching
            
            if not is_video_file:
                # Only apply camera commands if it's a live USB cam
                self._apply_camera_commands()
            
            ret, frame = self.cap.read()
            
            if not ret:
                # If frame read fails
                if is_video_file:
                    # If it's a video, loop back to the beginning
                    print("[PreviewWorker] Reached end of video, looping.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # If it's a camera, it's likely disconnected. Stop.
                    print("[PreviewWorker] Failed to read frame (Camera disconnected?). Stopping.")
                    break
            
            # Emit a *copy* of the frame
            self.new_frame.emit(frame.copy())
            
            # --- Sleep to maintain target FPS ---
            elapsed = time.monotonic() - start_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                # Only sleep if we have time to spare
                time.sleep(sleep_time)
        
        # --- Cleanup ---
        if self.cap:
            self.cap.release()
        print("[PreviewWorker] Stopped.")

    def _apply_camera_commands(self):
        """
        Check the command queue for any new camera settings
        (like exposure) and apply them to the `VideoCapture` object.
        """
        try:
            # Drain the queue of all pending commands
            while self.command_queue and not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()
                if 'property' in cmd and 'value' in cmd and self.cap:
                    # e.g., self.cap.set(cv2.CAP_PROP_EXPOSURE, -11.0)
                    self.cap.set(cmd['property'], cmd['value'])
        except Exception:
            # Ignore errors (e.g., queue.Empty, which shouldn't happen
            # with `not empty()` check, but good to be safe).
            pass

    def stop(self):
        """
        Public method to signal the worker thread to stop.
        This sets the `_running` flag to False, causing the `run` loop
        to exit.
        """
        print("[PreviewWorker] Stop signal received.")
        self._running = False
