"""
Utility Helpers

This file contains small utility classes or functions used across the
application to solve specific, shared problems.
"""

import threading

class ActiveQueueReference:
    """
    Thread-safe reference to the active multiprocessing input queue.

    This class solves a specific problem:
    -   The `VideoProcessingThread` (in the main process) needs to put
        frames into an `mp.Queue` (e.g., `queue_0`).
    -   The `InferenceProcessManager` (in the main process) needs to be
        able to *change* which queue the `VideoProcessingThread` is
        putting frames into (e.g., swap it to `queue_1`).

    If the `VideoProcessingThread` held a direct reference `q = queue_0`,
    it would be very difficult to update that reference safely from
    another thread.

    Instead, both threads hold a reference to *this object*.
    -   `VideoProcessingThread`: Calls `active_queue_ref.get()` in its
        loop to get the *current* queue just before putting a frame.
    -   `InferenceProcessManager`: Calls `active_queue_ref.set(queue_1)`
        to atomically point all future `get()` calls to the new queue.

    A `threading.Lock()` ensures that a `get()` and `set()` cannot
    happen at the exact same time, preventing race conditions.
    """
    def __init__(self):
        """Initializes the reference with a thread lock."""
        self._lock = threading.Lock()
        self._queue = None  # The currently referenced mp.Queue

    def set(self, queue):
        """
        Atomically sets the new active queue.
        This is called by the InferenceProcessManager.

        Args:
            queue (mp.Queue): The new multiprocessing queue to use.
        """
        with self._lock:
            self._queue = queue

    def get(self):
        """
        Atomically gets the current active queue.
        This is called by the VideoProcessingThread.

        Returns:
            mp.Queue: The current active multiprocessing queue.
        """
        with self._lock:
            return self._queue
