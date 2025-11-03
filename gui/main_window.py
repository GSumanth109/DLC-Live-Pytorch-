"""
gui/gui_update_worker.py

Defines the `GuiUpdateWorker` QThread.

This thread's job is to poll the `results_queue` (which is fed by the
inference process) at a fixed rate (the target display FPS). It uses a
"drain-to-latest-frame" strategy to prevent the GUI from lagging behind
if inference is faster than the display rate.
"""

import time
from PyQt5.QtCore import QThread, pyqtSignal

class GuiUpdateWorker(QThread):
    """ 
    Gets the latest results from the results_queue for GUI update
    at a target FPS. This prevents the GUI from locking up if
    inference is faster than the display refresh rate.
    
    Lives in the main (GUI) process as a QThread.
    
    Signals:
        new_frame_ready (pyqtSignal): Emits the latest data packet (dict)
                                      for the GUI to display.
    """
    new_frame_ready = pyqtSignal(dict) 

    def __init__(self, results_queue, shutdown_event, target_fps):
        """
        Args:
            results_queue (mp.Queue): The queue to pull processed frames from.
            shutdown_event (mp.Event): The event to signal stopping.
            target_fps (int): The target frames-per-second for GUI updates.
        """
        super().__init__()
        self.results_queue = results_queue
        self.shutdown_event = shutdown_event
        self.target_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.min_sleep = 0.001 # 1ms minimum sleep to yield control

    def run(self):
        """
        Main loop for the GUI update thread.
        
        The loop performs three main actions:
        1. Drains the queue to get only the latest available frame.
        2. Emits that frame via the `new_frame_ready` signal.
        3. Sleeps just long enough to maintain the target FPS.
        """
        while not self.shutdown_event.is_set():
            start_time = time.monotonic()
            packet = None
            
            try:
                # --- Drain-to-latest-frame logic ---
                qsize = self.results_queue.qsize()
                if qsize > 1:
                    # Discard all but the last frame
                    for _ in range(qsize - 1):
                        try:
                            self.results_queue.get_nowait()
                        except Exception:
                            break # Queue became empty, stop draining
                            
                # Get the latest frame (or wait briefly for one)
                timeout = self.target_interval / 2 if self.target_interval > 0 else 0.005
                packet = self.results_queue.get(timeout=timeout)
                packet['timestamps']['dequeued_for_gui'] = time.monotonic()
                
            except Exception:
                pass # Ignore queue.Empty or timeout

            # If we got a frame, emit it
            if packet:
                self.new_frame_ready.emit(packet)

            # --- Sleep to maintain target FPS ---
            elapsed = time.monotonic() - start_time
            sleep_needed = self.target_interval - elapsed
            
            # Sleep for the remaining time, but at least 1ms to yield
            sleep_duration = max(self.min_sleep, sleep_needed)
            time.sleep(sleep_duration)
