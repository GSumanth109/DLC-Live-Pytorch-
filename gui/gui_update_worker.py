"""
GUI Update Worker (QThread)

This file defines the `GuiUpdateWorker`, a QThread that runs in the main GUI
process. Its sole purpose is to act as a bridge between the high-speed
inference process (which puts results in a queue) and the main GUI thread
(which needs to display those results).

This worker is essential for a responsive UI:
1.  **Decoupling:** It separates the task of "getting a frame" from the task
    of "displaying a frame."
2.  **Rate Limiting:** It sleeps to match a target display FPS, preventing
    the GUI from trying to update on every single frame if inference is
    running very fast. This saves significant CPU resources on the GUI thread.
3.  **Queue Draining:** It intelligently "drains" the results queue,
    discarding stale frames to ensure that the GUI always displays the
    most recent available result. This is key to a low-latency feel.
"""

import time
from PyQt5.QtCore import QThread, pyqtSignal

class GuiUpdateWorker(QThread):
    """
    Manages fetching results from the multiprocessing queue at a fixed
    rate (target_fps) and emitting them as a Qt signal.
    
    This thread lives in the main GUI process. It polls the `results_queue`
    and emits the `new_frame_ready` signal, which the main window (App)
    is connected to.
    """
    
    # Signal to emit the new data packet (which contains frames, stats, etc.)
    # The 'dict' will be the full data packet from the inference process.
    new_frame_ready = pyqtSignal(dict)

    def __init__(self, results_queue, shutdown_event, target_fps):
        """
        Initializes the GUI update worker.

        Args:
            results_queue (multiprocessing.Queue): The queue from which to
                pull inference results.
            shutdown_event (multiprocessing.Event): The global event that
                signals all threads and processes to stop.
            target_fps (int): The desired frames-per-second to update the GUI.
                This worker will sleep to try and match this rate.
        """
        super().__init__()
        self.results_queue = results_queue
        self.shutdown_event = shutdown_event
        
        # Calculate the target time interval between frames in seconds.
        # If target_fps is 0, default to no delay (0 interval).
        self.target_interval = 1.0 / target_fps if target_fps > 0 else 0
        
        # A small minimum sleep to prevent pure CPU-spinning if interval is 0
        self.min_sleep = 0.001 # 1 millisecond

    def run(self):
        """
        The main loop for the GUI update thread.
        
        This loop continuously:
        1.  Checks if the `results_queue` has multiple frames.
        2.  If so, it drains the queue, discarding all but the most recent frame.
        3.  It gets that most recent frame.
        4.  It emits the frame via the `new_frame_ready` signal.
        5.  It calculates how long it took and sleeps to maintain the
            `target_interval` (target_fps).
        """
        print("[GuiUpdateWorker] Starting loop...")
        
        # Loop until the main application sets the shutdown_event
        while not self.shutdown_event.is_set():
            start_time = time.monotonic()
            packet = None
            
            try:
                # --- Drain-to-latest-frame logic ---
                # This is the most important part for a low-latency feel.
                # If the inference process is faster than the GUI display rate,
                # this queue will build up. We don't want to show old,
                # "laggy" frames. We only want the newest one.
                
                qsize = self.results_queue.qsize()
                if qsize > 1:
                    # If there's more than 1 frame, discard all but the last one.
                    for _ in range(qsize - 1):
                        try:
                            # Use get_nowait() to instantly remove an item.
                            self.results_queue.get_nowait()
                        except Exception:
                            # If queue becomes empty during draining, stop.
                            break 
                            
                # --- Get the latest frame ---
                # We now attempt to get the single latest frame.
                # We use a short timeout (half the frame interval) to wait
                # for a frame if the queue was empty.
                timeout = self.target_interval / 2 if self.target_interval > 0 else 0.005
                packet = self.results_queue.get(timeout=timeout)
                
                # Timestamp when the GUI thread received the packet
                packet['timestamps']['dequeued_for_gui'] = time.monotonic()
                
            except Exception:
                # This will most likely be a `queue.Empty` exception if no
                # frame was available within the timeout. This is normal.
                # We just 'pass' and let the loop sleep.
                pass 

            # If we successfully got a packet, emit it for the GUI
            if packet:
                self.new_frame_ready.emit(packet)

            # --- Sleep to maintain target FPS ---
            # Calculate how long the drain-and-get process took
            elapsed = time.monotonic() - start_time
            
            # Calculate how long we *still* need to sleep to hit our target interval
            sleep_needed = self.target_interval - elapsed
            
            # Sleep for the required time, but enforce a minimum sleep
            # to yield control back to the OS and prevent 100% CPU usage.
            sleep_duration = max(self.min_sleep, sleep_needed)
            time.sleep(sleep_duration)
            
        print("[GuiUpdateWorker] Stopped.")
