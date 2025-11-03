"""
utils/drawing.py

Utility functions for drawing overlays on OpenCV frames.

Includes functions for:
- Drawing skeletons and keypoints.
- Drawing a semi-transparent latency/statistics overlay.
"""

import cv2
import numpy as np

def draw_skeleton(frame, predictions, skel_conf, pt_conf):
    """
    Draws the skeleton and keypoints onto a frame.
    
    Args:
        frame (np.ndarray): The OpenCV (BGR) image frame to draw on.
        predictions (dict): The 'predictions' dictionary from the
                            inference packet.
        skel_conf (float): Minimum confidence (0.0-1.0) to draw a
                           skeleton line.
        pt_conf (float): Minimum confidence (0.0-1.0) to draw a keypoint.
    """
    
    # """
    # TODO: This skeleton structure is hardcoded.
    # For a more general solution, this should be loaded from the
    # DeepLabCut project's config.yaml file (`skeleton` key) and
    # passed to this function.
    # """
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)
    ] 

    try:
        bodyparts = predictions.get("bodyparts", [])
        
        # Iterate over each detected animal
        for keypoints in bodyparts: 
            if keypoints is None or keypoints.size == 0:
                continue

            # Draw Skeleton
            for i, j in skeleton:
                if max(i, j) < len(keypoints):
                    p1, p2 = keypoints[i], keypoints[j]
                    
                    # Check if keypoints are valid (x, y, likelihood)
                    if (isinstance(p1, (np.ndarray, list, tuple)) and len(p1) == 3 and
                        isinstance(p2, (np.ndarray, list, tuple)) and len(p2) == 3):
                        
                        # Draw line if both points are above skeleton confidence
                        if p1[2] > skel_conf and p2[2] > skel_conf:
                            pt1 = (int(p1[0]), int(p1[1]))
                            pt2 = (int(p2[0]), int(p2[1]))
                            cv2.line(frame, pt1, pt2, (255, 100, 0), 2)

            # Draw Keypoints
            for kp_data in keypoints:
                if isinstance(kp_data, (np.ndarray, list, tuple)) and len(kp_data) == 3:
                    x, y, confidence = kp_data
                    # Draw circle if point is above point confidence
                    if confidence > pt_conf:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        
    except Exception as e:
        # This error is not critical, so we just log it and continue
        print(f"!!! [Drawing Error] Failed to draw skeleton: {e}")
        pass


def draw_latency_overlay(frame, latencies):
    """
    Draws the semi-transparent latency overlay onto the frame.
    
    Args:
        frame (np.ndarray): The OpenCV (BGR) image frame to draw on.
        latencies (dict): A dictionary of latency values (in ms).
        
    Returns:
        np.ndarray: The frame with the overlay drawn on it.
    """
    y = 20
    overlay_height = 160 # Height for 8 lines of text
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (200, overlay_height), (0, 0, 0), -1)
    
    # Blend the overlay with the frame
    disp = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    def draw_text(label, value, color=(0, 255, 0)):
        """Helper to draw a formatted text line."""
        nonlocal y
        text = f"{label:<9}: {value:>5.1f}"
        cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 18

    # --- Draw all latency components ---
    draw_text("LATENCY(ms)", 0.0, (255, 255, 0))
    y -= 18 # Reuse last position
    
    draw_text("PreProc", latencies.get('PreProc', 0))
    draw_text("Inf Q Wt", latencies.get('InfQWait', 0))
    draw_text("Inference", latencies.get('Infer', 0))
    draw_text("GUI Q Wt", latencies.get('GUIQWait', 0))
    draw_text("GUI Updt", latencies.get('GUIUpdate', 0))
    draw_text("Cap->CSV", latencies.get('CapToCSV', 0))
    draw_text("E2E Disp", latencies.get('E2E', 0), (255, 255, 0))
    
    return disp
