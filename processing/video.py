# Updated: processing/video.py
"""
Handles reading frames from the video source (camera or file).

This file defines the VideoProcessingThread, which:
- Opens the selected video source (USB cam, IP cam, or video file).
- Continuously reads frames.
- Applies pre-processing (like flat-field correction).
- Applies user-defined cropping.
- Puts the frame packet into the queue for the inference process.
- **MODIFIED**: This version now loops video files instead of shutting down.
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
    """
    def __init__(self, settings, active_queue_ref, shutdown_event, command_queue, captured_counter, enqueued_counter):
        """
        Initializes the video processing thread.

        Args:
            settings (dict): The main application settings dictionary.
            active_queue_ref (ActiveQueueReference): Thread-safe ref to the inference input queue.
            shutdown_event (mp.Event): Event to signal this thread to stop.
            command_queue (mp.Queue): Queue for receiving real-time camera commands (e.g., exposure).
            captured_counter (mp.Value): Shared counter for total frames read.
            enqueued_counter (mp.Value): Shared counter for frames sent to inference.
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
        
        print("[VideoThread] Initialized.")
        
        # Load flat-field image if specified in settings
        if self.settings['method'] == 'Flat-field':
            path = self.settings['flat_image_path']
            if path and os.path.exists(path):
                try:
                    self.flat_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if self.flat_img is None:
                         print(f"!!! WARN: Failed to load flat image (is it an image?): {path}")
                    else:
                         print(f"[VideoThread] Flat-field image loaded from {path}")
                except Exception as e:
                    print(f"!!! WARN: Load flat image fail '{path}': {e}")
            elif path:
                print(f"!!! WARN: Flat image not found: {path}")

    def _process_frame(self, frame):
        """
        Applies pre-processing methods (e.g., flat-field) to the frame.
        """
        method = self.settings.get('method', 'None')
        try:
            if method == 'Flat-field' and self.flat_img is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if gray.shape != self.flat_img.shape:
                    # Resize flat-field image on the fly if dimensions don't match
                    self.flat_img = cv2.resize(self.flat_img, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                corr = cv2.divide(gray, self.flat_img.astype(np.float32) + 1e-6, scale=255.0)
                return cv2.cvtColor(np.uint8(np.clip(corr, 0, 255)), cv2.COLOR_GRAY2BGR)
            elif method == 'Morphological Opening':
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kern)
        except Exception as e:
            # If pre-processing fails, log it and disable it to prevent crashes
            print(f"!!! WARN: Pre-processing '{method}' error: {e}")
            self.settings['method'] = 'None'
            print("!!! Disabling pre-processing.")
        return frame

    def _apply_camera_commands(self):
        """
        Applies any pending real-time commands (exposure, gain) from the GUI.
        """
        try:
            while self.command_queue and not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()
                if 'property' in cmd and 'value' in cmd and self.cap:
                    self.cap.set(cmd['property'], cmd['value'])
        except Exception:
            pass # Ignore if queue is empty

    def run(self):
        """
        Main loop of the video thread.
        Reads frames, processes them, and puts them in the active queue.
        """
        source = self.settings['path']
        is_video_file = self.settings['camera_source'] == "Video File"
        
        try:
            print("[VideoThread] Run starting...")
            
            if is_video_file:
                if not os.path.exists(source):
                    raise IOError(f"Video file not found: {source}")
                print(f"[VideoThread] Opening video file: {source}")
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    raise IOError(f"Cannot open video file: {source}")
            
            else: # USB Webcam
                try:
                    source = int(source)
                except ValueError:
                    print(f"!!! CRIT: Invalid USB Cam Index '{source}'. Defaulting to 0.")
                    source = 0
                
                print(f"[VideoThread] Opening video source: {source}")
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    print(f"!!! [VideoThread] WARN: Opening with FFMPEG failed. Retrying with default...")
                    self.cap = cv2.VideoCapture(source)
                    if not self.cap.isOpened():
                        raise IOError(f"Cannot open video source: {source}")
                
                print("[VideoThread] Video source open.")
                buffer_ok = self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[VideoThread] Set BUFFERS=1, success={buffer_ok}")

                print("[VideoThread] Setting USB camera properties...")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings['cam_width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['cam_height'])
                self.cap.set(cv2.CAP_PROP_FPS, self.settings['cam_fps'])
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings['exposure'])
                self.cap.set(cv2.CAP_PROP_GAIN, self.settings['gain'])
                self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.settings['white_balance'])
                print(f"[VideoThread] Requested: {self.settings['cam_width']}x{self.settings['cam_height']} @ {self.settings['cam_fps']} FPS")
                    
        except Exception as e:
            print(f"!!! [VideoThread] CRITICAL INIT ERROR: {e}")
            self.shutdown_event.set()
            if self.cap: self.cap.release()
            return

        print("[VideoThread] Loop starting...");
        frames_read = 0; frames_put = 0
        
        while not self.shutdown_event.is_set():
            if not is_video_file:
                self._apply_camera_commands()
            
            ret, frame = self.cap.read()
            t_cap = time.monotonic()

            if not ret:
                # *** MODIFIED: Video Looping Logic ***
                if is_video_file:
                    print(f"[VideoThread] Reached end of video file. Looping...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to frame 0
                    continue # Continue to the next loop iteration
                else:
                    print(f"[VideoThread] Read fail (ret={ret}). Camera disconnected?");
                
                # Only signals shutdown if it's a camera error
                self.shutdown_event.set()
                break
            # *** End of Modification ***

            frames_read += 1
            self.captured_counter.value += 1
            
            processed_frame = self._process_frame(frame)
            t_proc = time.monotonic()

            if self.settings.get('crop_enabled', False):
                try:
                    x, y, w, h = self.settings['crop_x'], self.settings['crop_y'], self.settings['crop_w'], self.settings['crop_h']
                    h_orig, w_orig = processed_frame.shape[:2]
                    # Clamp crop values to be within frame dimensions
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, w_orig - x), min(h, h_orig - y)
                    if w > 0 and h > 0:
                        processed_frame = processed_frame[y:y+h, x:x+w]
                except Exception as e:
                    print(f"!!! [VideoThread] WARN: Error during cropping: {e}. Disabling crop.")
                    self.settings['crop_enabled'] = False
            
            packet = {
                'original_frame': frame.copy(),
                'processed_frame': processed_frame,
                'timestamps': { 'capture': t_cap, 'processed': t_proc, 'enqueued_for_inference': -1 }
            }

            q = self.active_queue_ref.get()
            if q:
                try: 
                    # Overwrite logic: if queue is full, pop old frame, push new one
                    if not q.empty():
                        try: q.get_nowait()
                        except Exception: pass
                    packet['timestamps']['enqueued_for_inference'] = time.monotonic()
                    q.put_nowait(packet)
                    self.enqueued_counter.value += 1
                    frames_put += 1
                except Exception as e:
                    pass # Queue is full, frame is dropped
            else:
                # Inference isn't running, wait a bit
                time.sleep(0.01)
                
            if is_video_file:
                # For video files, we sleep to match the file's FPS
                # This prevents the file from being processed too quickly
                file_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if file_fps <= 0: file_fps = 30
                time.sleep(1.0 / file_fps)

        print(f"[VideoThread] Exit loop. Read:{frames_read}, Put:{frames_put}");
        if self.cap:
            self.cap.release()
        print("[VideoThread] Stopped.")

