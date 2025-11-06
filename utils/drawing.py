"""
Drawing Utilities

This file provides functions for drawing overlays on video frames
using OpenCV. These functions are called by the `update_video_feed`
slot in `main_window.py` *after* a result packet is received
from the GUI worker.
"""

import cv2
import numpy as np

def draw_skeleton(frame, predictions, skel_conf, pt_conf):
    """
    Draws the skeleton and keypoints onto a frame.
    This function modifies the 'frame' in-place.

    Args:
        frame (np.ndarray): The OpenCV (BGR) image frame to draw on.
        predictions (dict): The 'predictions' dictionary from the
                            inference packet, which contains the 'bodyparts' key.
        skel_conf (float): Minimum confidence (0.0-1.0) to draw a skeleton line.
        pt_conf (float): Minimum confidence (0.0-1.0) to draw a keypoint.
    """
    
    # !!! IMPORTANT !!!
    # This is an example skeleton structure.
    # You MUST replace this with the skeleton from your DLC model's config.yaml
    # e.g., skeleton: [('nose', 'left_eye'), ('left_eye', 'left_ear'), ...]
    # You will need to map the part names to their indices.
    # For now, it assumes a simple 8-point chain.
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)
    ] 

    try:
        # 'bodyparts' is a list of N_animals, where each animal
        # has a (N_keypoints, 3) array [x, y, likelihood]
        bodyparts = predictions.get("bodyparts", [])
        
        for keypoints in bodyparts: # Iterate over each detected animal
            if keypoints is None or keypoints.size == 0:
                continue # Skip if no keypoints for this animal

            # --- 1. Draw Skeleton Lines ---
            for i, j in skeleton:
                # Check if both indices are within the keypoints array
                if max(i, j) < len(keypoints):
                    p1 = keypoints[i] # [x1, y1, c1]
                    p2 = keypoints[j] # [x2, y2, c2]
                    
                    # Check if keypoints are valid (x, y, likelihood)
                    if (isinstance(p1, (np.ndarray, list, tuple)) and len(p1) == 3 and
                        isinstance(p2, (np.ndarray, list, tuple)) and len(p2) == 3):
                        
                        # Draw line only if both points are above skeleton confidence
                        if p1[2] > skel_conf and p2[2] > skel_conf:
                            pt1_int = (int(p1[0]), int(p1[1]))
                            pt2_int = (int(p2[0]), int(p2[1]))
                            cv2.line(frame, pt1_int, pt2_int, (255, 100, 0), 2) # Blue-ish line

            # --- 2. Draw Keypoints (Circles) ---
            for kp_data in keypoints:
                if isinstance(kp_data, (np.ndarray, list, tuple)) and len(kp_data) == 3:
                    x, y, confidence = kp_data
                    # Draw circle if point is above point confidence
                    if confidence > pt_conf:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1) # Green circle
                        
    except Exception as e:
        print(f"!!! [Drawing Error] Failed to draw skeleton: {e}")
        # This error is not critical, so we just print it and continue
        pass


def draw_latency_overlay(frame, latencies):
    """
    Draws the semi-transparent latency overlay onto the top-left
    of the frame.

    Args:
        frame (np.ndarray): The OpenCV (BGR) image frame to draw on.
        latencies (dict): A dictionary of latency values (in ms), e.g.,
                          {'Infer': 10.5, 'E2E': 30.2, ...}

    Returns:
        np.ndarray: The frame with the latency overlay drawn on it.
    """
    y = 20
    overlay_height = 160 # Fixed height for 8 lines of text
    
    # 1. Create a black rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (5, 5),                 # Top-left corner
                  (200, overlay_height),  # Bottom-right corner
                  (0, 0, 0),              # Color (black)
                  -1)                     # Fill rectangle
    
    # 2. Blend the black rectangle with the frame
    #    This creates the semi-transparent effect
    disp = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    def draw_text(label, value, color=(0, 255, 0)):
        """Helper to draw a line of text on the 'disp' frame."""
        nonlocal y
        # Format: "Label    : 10.1" (left-aligned, right-aligned)
        text = f"{label:<9}: {value:>5.1f}"
        cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 18 # Move y-coord down for the next line

    # --- Draw all latency components ---
    draw_text("LATENCY(ms)", 0.0, (255, 255, 0)) # Yellow title
    y -= 18 # Reuse last position for the title (value is ignored)
    
    draw_text("PreProc", latencies.get('PreProc', 0))
    draw_text("Inf Q Wt", latencies.get('InfQWait', 0))
    draw_text("Inference", latencies.get('Infer', 0))
    draw_text("GUI Q Wt", latencies.get('GUIQWait', 0))
    draw_text("GUI Updt", latencies.get('GUIUpdate', 0))
    draw_text("Cap->CSV", latencies.get('CapToCSV', 0))
    draw_text("E2E Disp", latencies.get('E2E', 0), (255, 255, 0)) # Yellow E2E
    
    return disp
