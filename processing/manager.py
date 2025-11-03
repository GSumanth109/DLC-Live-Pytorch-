"""
processing/manager.py

Defines the `InferenceProcessManager` thread.

This thread's *only* job is to manage the separate `inference_worker`
process. This is the core of the memory-leak mitigation strategy.

It works by:
- Running two inference processes (`A` and `B`) in standby.
- Directing the `VideoProcessingThread` to send frames to one
  process (e.g., `A`) via the `ActiveQueueReference`.
- Monitoring the RAM usage of the active process (`A`).
- If RAM exceeds a trigger, it pre-starts the standby process (`B`).
- If RAM exceeds the limit, it "hot-swaps" the `ActiveQueueReference`
  to point to process `B`'s queue.
- It then safely shuts down and cleans up process `A`.
- Process `B` is now active, and `A` becomes the standby.
"""

import threading
import multiprocessing as mp
import time
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from processing.inference import inference_worker

class InferenceProcessManager(threading.Thread):
    """ 
    Manages the separate inference process, including hot-swapping
    to mitigate potential memory leaks.
    
    This thread lives in the main GUI process but spawns and manages
    child processes for inference.
    """
    def __init__(self, settings, active_queue_ref, results_queue, shutdown_event, processed_counter, csv_counter):
        """
        Args:
            settings (dict): The main application settings.
            active_queue_ref (ActiveQueueReference): The thread-safe
                reference that this manager will update to point to the
                currently active input queue.
            results_queue (mp.Queue): The single queue where all
                inference processes send their results.
            shutdown_event (mp.Event): The main application shutdown event.
            processed_counter (mp.Value): Shared counter for processed frames.
            csv_counter (mp.Value): Shared counter for CSV rows written.
        """
        super().__init__()
        self.settings = settings
        self.active_queue_ref = active_queue_ref
        self.results_queue = results_queue
        self.main_shutdown_event = shutdown_event
        self.daemon = True
        
        # --- Process "A" and "B" resources ---
        self.processes = [None, None]
        self.process_queues = [mp.Queue(maxsize=1), mp.Queue(maxsize=1)]
        self.shutdown_events = [mp.Event(), mp.Event()]
        self.ready_events = [mp.Event(), mp.Event()]    
        
        self.active_idx = 0
        self.swap_headstart_seconds = 5.0
        self.processed_counter = processed_counter
        self.csv_counter = csv_counter

    def _start_process(self, idx):
        """
        Starts (or restarts) the inference process at the given index.

        Args:
            idx (int): The index (0 or 1) of the process to start.
        """
        if self.processes[idx] and self.processes[idx].is_alive():
            return # Already running

        self.shutdown_events[idx].clear()
        self.ready_events[idx].clear()
        self._drain_queue(self.process_queues[idx])
        
        args = (
            self.settings, 
            self.process_queues[idx], 
            self.results_queue,
            self.shutdown_events[idx], 
            self.ready_events[idx], 
            self.processed_counter,
            self.csv_counter
        )
        
        self.processes[idx] = mp.Process(target=inference_worker, args=args)
        self.processes[idx].start()

    def _stop_process(self, idx):
        """
        Stops the inference process at the given index gracefully.
        Tries to `join`, and if that fails, `terminates`.

        Args:
            idx (int): The index (0 or 1) of the process to stop.
        """
        proc = self.processes[idx]
        queue = self.process_queues[idx]
        
        if proc and proc.is_alive():
            self.shutdown_events[idx].set()
            self._drain_queue(queue)
            
            # Give the process 5 seconds to shut down gracefully
            proc.join(timeout=5.0) 
            
            if proc.is_alive():
                # Process did not exit, force terminate
                proc.terminate()
                proc.join(timeout=2.0)
                
        self.processes[idx] = None
        self._drain_queue(queue)

    def _drain_queue(self, queue):
        """Utility to empty a multiprocessing queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break

    def run(self):
        """
        Main loop for the manager thread.
        
        If `psutil` is available, it monitors the active process's RAM.
        If RAM exceeds the threshold, it triggers the hot-swap logic.
        
        If `psutil` is not available, it runs in a "simple mode" with
        only one process and no RAM monitoring.
        """
        
        # --- Simple Mode (psutil not installed) ---
        if not PSUTIL_AVAILABLE:
            self._start_process(0)
            self.active_queue_ref.set(self.process_queues[0])
            # Wait for the main app to signal shutdown
            self.main_shutdown_event.wait()
            self._stop_process(0)
            return

        # --- Hot-Swap Mode (psutil is available) ---
        ram_limit_gb = self.settings['ram_threshold_gb']
        trigger_gb = ram_limit_gb * 0.9 # Start standby at 90%
        
        self._start_process(self.active_idx)
        self.active_queue_ref.set(self.process_queues[self.active_idx])
        standby_started = False

        while not self.main_shutdown_event.is_set():
            active_proc = self.processes[self.active_idx]
            current_mem_gb = 0.0
            
            # --- 1. Check Active Process State ---
            if active_proc and active_proc.is_alive():
                try:
                    process = psutil.Process(active_proc.pid)
                    current_mem_gb = process.memory_info().rss / (1024 ** 3)
                except psutil.NoSuchProcess:
                    # Process died unexpectedly, restart it
                    self._start_process(self.active_idx)
                    self.active_queue_ref.set(self.process_queues[self.active_idx])
                    standby_started = False
                    time.sleep(5)
                    continue
            else:
                if not self.main_shutdown_event.is_set():
                    # Process is not running, restart it
                    self._start_process(self.active_idx)
                    self.active_queue_ref.set(self.process_queues[self.active_idx])
                    standby_started = False
                    time.sleep(5)
                continue

            # --- 2. Hot-Swap Trigger Logic ---
            
            # If RAM > 90% and standby isn't started, start it
            if not standby_started and current_mem_gb > trigger_gb:
                stby_idx = 1 - self.active_idx
                self._start_process(stby_idx)
                standby_started = True

            # If RAM > 100%, perform the swap
            if current_mem_gb > ram_limit_gb:
                stby_idx = 1 - self.active_idx
                
                if not self.processes[stby_idx] or not self.processes[stby_idx].is_alive():
                    # Standby died or wasn't started properly
                    self._start_process(stby_idx)
                    time.sleep(1) # Give it a sec

                # Wait for the standby process to signal it's ready
                ready = self.ready_events[stby_idx].wait(timeout=self.swap_headstart_seconds)
                
                # --- 3. Perform the Swap ---
                # Point the ActiveQueueReference to the standby queue
                self.active_queue_ref.set(self.process_queues[stby_idx])
                
                old_idx = self.active_idx
                self.active_idx = stby_idx
                standby_started = False
                
                # Stop the old (leaky) process
                self._stop_process(old_idx)
                
            time.sleep(1.0) # Check RAM every second

        # --- Shutdown ---
        self._stop_process(0)
        self._stop_process(1)
