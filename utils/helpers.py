"""
utils/helpers.py

Contains helper classes and functions for thread-safe operations.
"""

import threading

class ActiveQueueReference:
    """ 
    A thread-safe wrapper for a multiprocessing.Queue object.
    
    This class is essential for the "hot-swap" logic.
    - The `InferenceProcessManager` (manager thread) `set`s the
      reference to point to the currently active process's input queue.
    - The `VideoProcessingThread` (video thread) `get`s the
      reference in every loop iteration to know where to send
      the latest frame.
      
    This allows the manager to redirect the video thread's output
    from one process to another without stopping the video thread.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = None

    def set(self, queue):
        """
        Atomically sets the new active queue.
        
        Args:
            queue (mp.Queue or None): The new queue to be used.
        """
        with self._lock:
            self._queue = queue

    def get(self):
        """
        Atomically gets the current active queue.
        
        Returns:
            mp.Queue or None: The current active queue.
        """
        with self._lock:
            return self._queue
