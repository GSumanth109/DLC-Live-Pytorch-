"""
processing/video.py

Defines the `VideoProcessingThread`.

This thread runs during the main "Analysis" mode. It is responsible for:
- Reading frames from the camera or video file.
- Applying pre-processing (Flat-field, etc.).
- Applying cropping.
- Creating a "packet" (dict) with the frame and timestamps.
- Putting this packet into the *active* inference queue, using a
  "drain-to-latest" strategy to only send the most recent frame.
- Signaling shutdown if the video file ends.
"""

import threading
import cv2
import numpy as np
import time
import os

class VideoProcessingThread(threading.Thread):
    """ 
    Reads frames, applies pre-processing/cropping, and puts
    the latest frame in the active inference queue.
    
    - For Camera: Grabs live frames.
    - For Video File: Streams until end, then signals shutdown.
    
    This thread is designed to "overflow" - it drops its own frames
    if the inference process can't keep up, by overwriting the
    single slot in the input queue.
    """
    def __init__(self, settings, active_queue_ref, shutdown_event, command_queue, captured_counter, enqueued_counter):
        """
        Args:
            settings (dict): The main application settings.
            active_queue_ref (ActiveQueueReference): A thread-safe reference
                to the *current* active inference input queue.
            shutdown_event (mp.Event): The main event to signal shutdown.
            command_queue (mp.Queue): Queue for receiving camera commands.
            captured_counter (mp.Value): Shared counter for frames read.
            enqueued_counter (mp.Value): Shared counter for frames sent
                                         to inference.
        """
        super().__init__()
        self.settings = settings
        self.active_queue_ref = active_queue_ref
        self.shutdown_event = shutdown_event
        self.command_queue = command_queue
        self.daemon = True
        self.cap = None
        self.flat_img = None
        self.captured_counter = captured_counter
        self.enqueued_counter = enqueued_counter
        
        # Load the flat-field image if specified
        if self.settings['method'] == 'Flat-field':
            path = self.settings['flat_image_path']
            if path and os.path.exists(path):
                try:
                    self.flat_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if self.flat_img is None:
                         print(f"!!! WARN: Failed to load flat image: {path}")
                except Exception as e:
                    print(f"!!! WARN: Load flat image fail '{path}': {e}")
            elif path:
                print(f"!!! WARN: Flat image not found: {path}")

    def _process_frame(self, frame):
        """
        Applies the selected pre-processing method to the frame.

        Args:
            frame (np.ndarray): The input BGR frame.

        Returns:
            np.ndarray: The processed BGR frame.
        """
        method = self.settings.get('method', 'None')
        try:
            if method == 'Flat-field' and self.flat_img is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                # Resize flat-field image if it doesn't match frame
                if gray.shape != self.flat_img.shape:
                    self.flat_img = cv2.resize(self.flat_img, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                corr = cv2.divide(gray, self.flat_img.astype(np.float32) + 1e-6, scale=255.0)
                # Convert back to 3-channel BGR for the model
                return cv2.cvtColor(np.uint8(np.clip(corr, 0, 255)), cv2.COLOR_GRAY2BGR)
                
            elif method == 'Morphological Opening':
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kern)
                
        except Exception as e:
            # Disable pre-processing on failure to avoid spamming logs
            self.settings['method'] = 'None'
        
        return frame # Return original frame if 'None' or on error

    def _apply_camera_commands(self):
        """Applies pending camera settings from the command queue."""
        try:
            while self.command_queue and not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()
                if 'property' in cmd and 'value' in cmd and self.cap:
                    self.cap.set(cmd['property'], cmd['value'])
        except Exception:
            pass

    def run(self):
        """
        Main loop for the video processing thread.
        
        Initializes the video source, then enters a loop to:
        1. Read a frame.
        2. Apply camera commands (if live).
        3. Pre-process and crop the frame.
        4. Construct a 'packet' dictionary.
        5. Put the packet in the active inference queue (using
           a "drain-to-latest" strategy).
        6. Sleep to match video FPS if reading from a file.
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
            
            else: # USB Webcam
                try:
                    source = int(source)
                except ValueError:
                    source = 0
                
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(source) # Fallback
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
                    
        except Exception as e:
            self.shutdown_event.set() # Signal critical failure
            if self.cap: self.cap.release()
            return

        # --- Main Capture Loop ---
        while not self.shutdown_event.is_set():
            if not is_video_file:
                self._apply_camera_commands()
            
            ret, frame = self.cap.read()
            t_cap = time.monotonic()

            if not ret:
                # End of video file or camera disconnected
                self.shutdown_event.set()
                break

            self.captured_counter.value += 1
            
            # --- Process and Crop ---
            processed_frame = self._process_frame(frame)
            t_proc = time.monotonic()

            if self.settings.get('crop_enabled', False):
                try:
                    x, y, w, h = self.settings['crop_x'], self.settings['crop_y'], self.settings['crop_w'], self.settings['crop_h']
                    h_orig, w_orig = processed_frame.shape[:2]
                    # Clamp values to be safe
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, w_orig - x), min(h, h_orig - y)
                    if w > 0 and h > 0:
                        processed_frame = processed_frame[y:y+h, x:x+w]
                except Exception as e:
                    self.settings['crop_enabled'] = False # Disable on error
            
            # --- Create Packet ---
            packet = {
                'original_frame': frame.copy(),
                'processed_frame': processed_frame,
                'timestamps': { 'capture': t_cap, 'processed': t_proc, 'enqueued_for_inference': -1 }
            }

            # --- Enqueue for Inference ---
            q = self.active_queue_ref.get() # Get the current active queue
            if q:
                try: 
                    # "Drain-to-latest" logic:
                    # If the queue is not empty, discard the old frame
                    if not q.empty():
                        try: q.get_nowait()
                        except Exception: pass
                        
                    # Put the new frame
                    packet['timestamps']['enqueued_for_inference'] = time.monotonic()
                    q.put_nowait(packet)
                    self.enqueued_counter.value += 1
                except Exception:
                    pass # Queue was full, frame is dropped
            else:
                # Manager might be swapping queues, wait briefly
                time.sleep(0.01)
                
            if is_video_file:
                # For video files, we must sleep to match the file's FPS
                # to avoid processing the whole file in seconds.
                file_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if file_fps <= 0: file_fps = 30
                time.sleep(1.0 / file_fps)

        if self.cap:
            self.cap.release()
