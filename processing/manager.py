"""
Inference Process Manager (Thread)

This file defines the `InferenceProcessManager`, a standard Python
`threading.Thread` that runs in the main GUI process.

This thread's sole purpose is to start, stop, and monitor the *actual*
inference `multiprocessing.Process` (defined in `inference.py`).

Its most critical feature is handling memory-leak mitigation:
-   PyTorch models (especially on CUDA) can sometimes have memory leaks
    that build up over hours of continuous use.
-   This manager monitors the RAM usage of the active inference process.
-   If RAM exceeds a "trigger" threshold (e.g., 90% of the limit), it
    pre-emptively starts a *second*, clean inference process in the background.
-   Once that new process is fully loaded and "ready", the manager
    "hot-swaps" the active queue, directing all new frames from the
    `VideoProcessingThread` to this new process.
-   It then safely shuts down the old, high-RAM process.
-   This allows the application to run indefinitely without crashing.

If `psutil` is not installed, this thread simply starts one process
and leaves it running (i.e., the hot-swap feature is disabled).
"""

import threading
import multiprocessing as mp
import time
import os

# --- Psutil for memory monitoring ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("!!! WARNING: `psutil` not found. RAM monitoring and automatic restart will be disabled.")

# The function that runs in the separate process
from processing.inference import inference_worker

class InferenceProcessManager(threading.Thread):
    """
    Manages the separate inference `mp.Process`, including starting,
    stopping, and hot-swapping for memory leak mitigation.
    """
    def __init__(self, settings, active_queue_ref, results_queue, shutdown_event, processed_counter, csv_counter):
        """
        Initializes the Inference Process Manager.

        Args:
            settings (dict): The full dictionary of runtime settings.
            active_queue_ref (ActiveQueueReference): Thread-safe object
                holding a reference to the *current* input queue. This
                manager will `set()` this reference to swap queues.
            results_queue (mp.Queue): The single queue where all
                inference processes will put their final results for the GUI.
            shutdown_event (mp.Event): The global event to signal stopping.
            processed_counter (mp.Value): Shared counter for total
                frames processed by inference.
            csv_counter (mp.Value): Shared counter for total CSV rows written.
        """
        super().__init__()
        self.settings = settings
        self.active_queue_ref = active_queue_ref
        self.results_queue = results_queue
        self.main_shutdown_event = shutdown_event
        self.daemon = True # Ensure thread exits when main app exits
        
        # --- Process Pool (size 2 for hot-swapping) ---
        # We hold two of everything: one for the "active" process
        # and one for the "standby" process.
        self.processes = [None, None]               # Holds the mp.Process objects
        self.process_queues = [mp.Queue(maxsize=1), # Input queue for Proc 0
                               mp.Queue(maxsize=1)] # Input queue for Proc 1
        self.shutdown_events = [mp.Event(),         # Shutdown signal for Proc 0
                                mp.Event()]         # Shutdown signal for Proc 1
        self.ready_events = [mp.Event(),            # "Ready" signal from Proc 0
                             mp.Event()]            # "Ready" signal from Proc 1
        
        self.active_idx = 0 # Index (0 or 1) of the currently active process
        self.swap_headstart_seconds = 5.0 # Time to wait for standby to load
        self.processed_counter = processed_counter
        self.csv_counter = csv_counter

    def _start_process(self, idx):
        """
        Starts (or restarts) the inference process at the given index (0 or 1).

        Args:
            idx (int): The index (0 or 1) of the process slot to start.
        """
        if self.processes[idx] and self.processes[idx].is_alive():
            print(f"[Manager] Process {idx} is already running.")
            return

        print(f"[Manager] Starting inference process {idx}...")
        
        # Clear any old events for this index
        self.shutdown_events[idx].clear()
        self.ready_events[idx].clear()
        
        # Drain its input queue
        self._drain_queue(self.process_queues[idx])
        
        # Arguments to pass to the `inference_worker` function
        args = (
            self.settings, 
            self.process_queues[idx],   # This process's unique input queue
            self.results_queue,         # The *shared* results queue
            self.shutdown_events[idx],  # This process's unique shutdown event
            self.ready_events[idx],     # This process's unique ready event
            self.processed_counter,     # Shared counter
            self.csv_counter            # Shared counter
        )
        
        # Create and start the new process
        self.processes[idx] = mp.Process(target=inference_worker, args=args)
        self.processes[idx].start()

    def _stop_process(self, idx):
        """
        Gracefully stops the inference process at the given index.

        Args:
            idx (int): The index (0 or 1) of the process slot to stop.
        """
        proc = self.processes[idx]
        queue = self.process_queues[idx]
        
        if proc and proc.is_alive():
            print(f"[Manager] Stopping inference process {idx} (PID: {proc.pid})...")
            
            # 1. Signal the process to shut down
            self.shutdown_events[idx].set()
            
            # 2. Drain its input queue to unblock any `queue.put()`
            self._drain_queue(queue)
            
            # 3. Wait for the process to exit
            proc.join(timeout=5.0) # Wait 5 seconds
            
            # 4. If it's still alive, terminate it forcefully
            if proc.is_alive():
                print(f"[Manager] Process {idx} did not exit gracefully. Terminating...")
                proc.terminate()
                proc.join(timeout=2.0)
                
            print(f"[Manager] Process {idx} stopped.")
            
        self.processes[idx] = None
        self._drain_queue(queue) # Final drain

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
        Monitors RAM of the active process and performs hot-swap if
        the threshold is breached.
        """
        
        # --- Simple Mode (if psutil is missing) ---
        if not PSUTIL_AVAILABLE:
            print("[Manager] psutil not available. Running in simple, no-restart mode.")
            self._start_process(0) # Start process 0
            self.active_queue_ref.set(self.process_queues[0]) # Set it as active
            self.main_shutdown_event.wait() # Wait for global shutdown
            self._stop_process(0) # Stop process 0
            print("[Manager] Stopped.")
            return

        # --- Full Hot-Swap Mode ---
        ram_limit_gb = self.settings['ram_threshold_gb']
        # Start standby process at 90% of the limit
        trigger_gb = ram_limit_gb * 0.9
        
        # Start the initial process (index 0)
        self._start_process(self.active_idx)
        # Point the Video thread to its queue
        self.active_queue_ref.set(self.process_queues[self.active_idx])
        standby_started = False # Flag to track if standby is booting

        print(f"[Manager] Monitoring process {self.active_idx}. RAM Limit: {ram_limit_gb:.2f} GB. Trigger: {trigger_gb:.2f} GB.")

        while not self.main_shutdown_event.is_set():
            active_proc = self.processes[self.active_idx]
            current_mem_gb = 0.0
            
            # --- Check Active Process Health ---
            if active_proc and active_proc.is_alive():
                try:
                    # Get RAM usage
                    process = psutil.Process(active_proc.pid)
                    current_mem_gb = process.memory_info().rss / (1024 ** 3)
                except psutil.NoSuchProcess:
                    # Process died unexpectedly
                    print(f"[Manager] Active process {self.active_idx} (PID: {active_proc.pid}) died unexpectedly.")
                    self._start_process(self.active_idx) # Restart it
                    self.active_queue_ref.set(self.process_queues[self.active_idx])
                    standby_started = False
                    time.sleep(5) # Wait for it to load
                    continue
            else:
                # Process is not running (and we're not shutting down)
                if not self.main_shutdown_event.is_set():
                    print(f"[Manager] Process {self.active_idx} is not alive. Restarting...")
                    self._start_process(self.active_idx) # Restart it
                    self.active_queue_ref.set(self.process_queues[self.active_idx])
                    standby_started = False
                    time.sleep(5) # Wait for it to load
                continue

            # --- Hot-Swap Logic ---
            
            # 1. Trigger: RAM is high, but not critical. Start standby.
            if not standby_started and current_mem_gb > trigger_gb:
                stby_idx = 1 - self.active_idx # The *other* index (0 or 1)
                print(f"[Manager] RAM {current_mem_gb:.2f}/{ram_limit_gb:.2f} GB. Pre-starting standby process {stby_idx}...")
                self._start_process(stby_idx)
                standby_started = True

            # 2. Limit Breach: RAM is critical. Perform the swap.
            if current_mem_gb > ram_limit_gb:
                print(f"[Manager] RAM LIMIT BREACH ({current_mem_gb:.2f}/{ram_limit_gb:.2f} GB). Swapping processes.")
                stby_idx = 1 - self.active_idx
                
                # Check if standby is alive (it should be, from step 1)
                if not self.processes[stby_idx] or not self.processes[stby_idx].is_alive():
                    print(f"[Manager] Standby {stby_idx} not running! Starting it now...")
                    self._start_process(stby_idx)
                    time.sleep(1) # Give it a second

                # Wait for the standby process to signal it's "ready"
                print(f"[Manager] Waiting for standby {stby_idx} to be ready...")
                ready = self.ready_events[stby_idx].wait(timeout=self.swap_headstart_seconds)
                
                if ready:
                    print(f"[Manager] SWAPPING: {self.active_idx} -> {stby_idx}.")
                else:
                    # If it's not ready after 5s, we swap anyway
                    print(f"[Manager] WARNING: Standby {stby_idx} not ready, but swapping anyway.")

                # --- THE SWAP ---
                # Point the Video thread to the *new* queue
                self.active_queue_ref.set(self.process_queues[stby_idx])
                
                old_idx = self.active_idx
                self.active_idx = stby_idx # The standby is now the active
                standby_started = False
                
                # Finally, stop the old, high-RAM process
                self._stop_process(old_idx)
                print(f"[Manager] Swap complete. Now monitoring process {self.active_idx}.")
            
            # Check RAM every second
            time.sleep(1.0) 

        # --- Global Shutdown ---
        print("[Manager] Shutdown signal received.")
        self._stop_process(0)
        self._stop_process(1)
        print("[Manager] Stopped.")
