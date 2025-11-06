"""
The inference worker function.
This runs in a separate process to perform DLC inference.

(This file is based on the original modular project structure.)
"""

import os
import time
import cv2
import numpy as np
import torch
import gc
import csv
from deeplabcut.pose_estimation_pytorch import get_pose_inference_runner
from deeplabcut.core.config import read_config_as_dict
# *** MODIFIED: Import torch.amp for the new autocast syntax ***
import torch.amp 

def inference_worker(settings, process_queue, results_queue, shutdown_event, ready_event, processed_counter, csv_counter):
    """
    The target function for the dedicated inference process.
    - Initializes the DLC model.
    - Initializes the CSV writer.
    - Enters a loop, getting frames from process_queue.
    - Runs inference.
    - Writes results to CSV.
    - Puts results packet into results_queue.
    """
    pose_runner = None
    csv_writer = None
    csv_file = None
    frame_idx = 0
    worker_pid = os.getpid() # Get PID for logging/monitoring
    
    # Store device and fp16 status for use in the loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = settings['use_fp16'] and device == "cuda"
    
    try:
        print(f"[Proc-{worker_pid}] Initializing model...")
        
        # Load DLC model configuration
        model_cfg = read_config_as_dict(settings['pytorch_config_path'])
        
        # Infer method (TopDown/BottomUp) if not present in pytorch_config.yaml
        if 'method' not in model_cfg:
            main_cfg = read_config_as_dict(settings['config_path'])
            model_cfg['method'] = 'BottomUp' if not main_cfg.get('multianimalproject') else 'TopDown'
            
        # Get the pose runner (inference engine)
        pose_runner = get_pose_inference_runner(
            model_config=model_cfg,
            snapshot_path=settings['snapshot_path'],
            device=device
        )
        print(f"[Proc-{worker_pid}] Model loaded on {device}.")
        
        if use_fp16:
            try:
                pose_runner.model.half()
                print(f"[Proc-{worker_pid}] Model set to FP16 (half-precision).")
            except Exception as e:
                print(f"!!! [Proc-{worker_pid}] WARN: Could not set FP16: {e}")

        # Initialize CSV file if requested
        if settings['save_csv']:
            print(f"[Proc-{worker_pid}] Initializing CSV: {settings['csv_output_path']}")
            
            # Build CSV header
            header = ['timestamp', 'frame_index', 'capture_to_csv_ms', 'inference_ms']
            bodyparts = read_config_as_dict(settings['config_path'])['bodyparts']
            for bp in bodyparts:
                header.extend([f'{bp}_x', f'{bp}_y', f'{bp}_likelihood'])
            
            csv_file = open(settings['csv_output_path'], 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)

        print(f"[Proc-{worker_pid}] Model ready.");
        ready_event.set() # Signal to manager that we are ready
        
    except Exception as e:
        print(f"!!! [Proc-{worker_pid}] CRITICAL INIT ERROR: {e}")
        import traceback
        traceback.print_exc()
        if csv_file:
            csv_file.close()
        return # Exit process if init fails

    # --- Main Inference Loop ---
    print(f"[Proc-{worker_pid}] Entering main inference loop...")
    
    while not shutdown_event.is_set():
        try:
            packet = None
            try:
                # Wait for a frame packet from the queue
                packet = process_queue.get(timeout=0.1) # 100ms timeout
                packet['timestamps']['dequeued_for_inference'] = time.monotonic()
            except Exception: # queue.Empty or timeout
                if shutdown_event.is_set():
                    break # Exit loop if shutdown is requested
                continue # No frame ready, loop again

            # Convert BGR (OpenCV) to RGB (DLC Model)
            rgb_frame = cv2.cvtColor(packet['processed_frame'], cv2.COLOR_BGR2RGB)

            # --- Run Inference ---
            inf_start = time.monotonic()
            
            # *** REVERTED FP16 FIX ***
            # We go back to calling pose_runner.inference() directly,
            # as your DLC version does not have .preprocess()
            
            # *** UPDATED SYNTAX to fix FutureWarning ***
            # Use torch.amp.autocast with the modern syntax
            with torch.amp.autocast(device_type=device, enabled=use_fp16):
                predictions = pose_runner.inference(images=[rgb_frame])
                
            inf_end = time.monotonic()
            inference_ms = (inf_end - inf_start) * 1000

            # Add results to the packet
            packet['predictions'] = predictions[0]
            packet['inference_time_ms'] = inference_ms
            packet['timestamps']['inferred'] = inf_end

            # --- CSV Writing ---
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
                    if keypoints.ndim == 3: # Multi-animal (N_animals, N_keypoints, 3)
                        # Flatten for CSV, assumes 1 animal for now
                        # TODO: Handle multi-animal CSV format properly if needed
                        kp_set = keypoints[0] # Just take first animal
                        for kp in kp_set:
                            row.extend(kp)
                    elif keypoints.ndim == 2 and keypoints.shape[1] == 3: # Single animal (N_keypoints, 3)
                         for kp in keypoints:
                            row.extend(kp)
                    else:
                        print(f"[Proc-{worker_pid}] Warn: Unexpected keypoints shape: {keypoints.shape}")
                else:
                    # No keypoints found, fill with NaNs
                    num_bodyparts = len(read_config_as_dict(settings['config_path'])['bodyparts'])
                    row.extend(['NaN'] * (num_bodyparts * 3))

                csv_writer.writerow(row)
                csv_counter.value += 1 # Increment shared counter

            # --- Send to GUI ---
            packet['timestamps']['enqueued_for_gui'] = time.monotonic()
            results_queue.put(packet) # This can block if GUI queue is full
            frame_idx += 1
            processed_counter.value += 1 # Increment shared counter

        except Exception as e:
            if not shutdown_event.is_set():
                # Log any crashes in the loop
                print(f"!!! [Proc-{worker_pid}] CRASH in loop: {e}")
                import traceback
                traceback.print_exc()
            break # Exit loop on error

    # --- Cleanup ---
    print(f"[Proc-{worker_pid}] Cleaning up and shutting down...")
    del pose_runner
    if csv_file:
        csv_file.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Proc-{worker_pid}] Stopped.")

