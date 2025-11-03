"""
processing/inference.py

Defines the `inference_worker` function.

This is the core "work" function that runs in its own separate
multiprocessing.Process. It is completely isolated from the main
GUI application.

It performs the following steps:
1.  Initializes the DeepLabCut model (this is the slow part).
2.  Initializes the CSV writer if requested.
3.  Signals the `InferenceProcessManager` that it is "ready".
4.  Enters a loop, waiting for frame packets from its `process_queue`.
5.  Runs DLC inference on the frame.
6.  Formats the results and writes them to the CSV file.
7.  Puts the full data packet (with predictions) into the
    `results_queue` for the GUI to pick up.
8.  Cleans up GPU memory on exit.
"""

import os
import time
import cv2
import numpy as np
import torch
import gc
import csv
import traceback
from deeplabcut.pose_estimation_pytorch import get_pose_inference_runner
from deeplabcut.core.config import read_config_as_dict

def inference_worker(settings, process_queue, results_queue, shutdown_event, ready_event, processed_counter, csv_counter):
    """
    The target function for the dedicated inference process.
    
    - Initializes the DLC model.
    - Initializes the CSV writer.
    - Enters a loop, getting frames from `process_queue`.
    - Runs inference.
    - Writes results to CSV.
    - Puts results packet into `results_queue`.

    Args:
        settings (dict): The main application settings.
        process_queue (mp.Queue): The *input* queue for this specific
                                  process (maxsize=1).
        results_queue (mp.Queue): The *output* queue, shared by all
                                  processes, feeding the GUI.
        shutdown_event (mp.Event): The event to signal this process to stop.
        ready_event (mp.Event): The event this process sets when its
                                model is loaded and it's ready to work.
        processed_counter (mp.Value): Shared counter for processed frames.
        csv_counter (mp.Value): Shared counter for CSV rows written.
    """
    pose_runner = None
    csv_writer = None
    csv_file = None
    frame_idx = 0
    worker_pid = os.getpid() # Get PID for logging/monitoring
    
    try:
        # --- 1. Initialize Model ---
        model_cfg = read_config_as_dict(settings['pytorch_config_path'])
        
        # Infer method (TopDown/BottomUp) if not present
        if 'method' not in model_cfg:
            main_cfg = read_config_as_dict(settings['config_path'])
            model_cfg['method'] = 'BottomUp' if not main_cfg.get('multianimalproject') else 'TopDown'
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pose_runner = get_pose_inference_runner(
            model_config=model_cfg,
            snapshot_path=settings['snapshot_path'],
            device=device
        )
        
        # Apply FP16 (half precision) if requested
        if settings['use_fp16'] and device == "cuda":
            try:
                pose_runner.model.half()
            except Exception as e:
                print(f"!!! [Proc-{worker_pid}] WARN: Could not set FP16: {e}")

        # --- 2. Initialize CSV Writer ---
        if settings['save_csv']:
            # Build CSV header
            header = ['timestamp', 'frame_index', 'capture_to_csv_ms', 'inference_ms']
            bodyparts = read_config_as_dict(settings['config_path'])['bodyparts']
            for bp in bodyparts:
                header.extend([f'{bp}_x', f'{bp}_y', f'{bp}_likelihood'])
            
            csv_file = open(settings['csv_output_path'], 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)

        # --- 3. Signal Ready ---
        ready_event.set() 
        
    except Exception as e:
        # If init fails, print error and exit process
        print(f"!!! [Proc-{worker_pid}] CRITICAL INIT ERROR: {e}")
        traceback.print_exc()
        if csv_file:
            csv_file.close()
        return # Exit process

    # --- 4. Main Inference Loop ---
    while not shutdown_event.is_set():
        try:
            packet = None
            try:
                # Wait for a frame packet from the queue
                packet = process_queue.get(timeout=0.1) # 100ms timeout
                packet['timestamps']['dequeued_for_inference'] = time.monotonic()
            except Exception: # queue.Empty
                if shutdown_event.is_set():
                    break # Exit loop if shutdown is requested
                continue # No frame ready, loop again

            # Convert BGR (OpenCV) to RGB (DLC Model)
            rgb_frame = cv2.cvtColor(packet['processed_frame'], cv2.COLOR_BGR2RGB)

            # --- 5. Run Inference ---
            inf_start = time.monotonic()
            predictions = pose_runner.inference(images=[rgb_frame])
            inf_end = time.monotonic()
            inference_ms = (inf_end - inf_start) * 1000

            # Add results to the packet
            packet['predictions'] = predictions[0]
            packet['inference_time_ms'] = inference_ms
            packet['timestamps']['inferred'] = inf_end

            # --- 6. CSV Writing ---
            csv_write_time = time.monotonic()
            capture_to_csv_ms = (csv_write_time - packet['timestamps']['capture']) * 1000
            packet['capture_to_csv_ms'] = capture_to_csv_ms

            if csv_writer:
                row = [
                    packet['timestamps']['capture'], 
                    frame_idx,
                    f"{capture_to_csv_ms:.2f}", 
                    f"{inference_ms:.2f}"
                ]
                keypoints = predictions[0]["bodyparts"]
                
                if keypoints is not None and keypoints.size > 0:
                    if keypoints.ndim == 3: # Multi-animal (N, K, 3)
                        # TODO: Handle multi-animal CSV format properly
                        kp_set = keypoints[0] # Just take first animal
                        row.extend(kp_set.flatten())
                    elif keypoints.ndim == 2: # Single animal (K, 3)
                         row.extend(keypoints.flatten())
                else:
                    # No keypoints found, fill with NaNs
                    num_bodyparts = len(read_config_as_dict(settings['config_path'])['bodyparts'])
                    row.extend(['NaN'] * (num_bodyparts * 3))

                csv_writer.writerow(row)
                csv_counter.value += 1 # Increment shared counter

            # --- 7. Send to GUI ---
            packet['timestamps']['enqueued_for_gui'] = time.monotonic()
            results_queue.put(packet) # This can block if GUI queue is full
            frame_idx += 1
            processed_counter.value += 1 # Increment shared counter

        except Exception as e:
            if not shutdown_event.is_set():
                print(f"!!! [Proc-{worker_pid}] CRASH in loop: {e}")
                traceback.print_exc()
            break # Exit loop on error

    # --- 8. Cleanup ---
    del pose_runner
    if csv_file:
        csv_file.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
