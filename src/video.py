"""Video processing module for Matrix-style character effects.

This module processes multiple video streams (original, depth, and alpha mask)
to generate Matrix-like character animations. It applies frame-by-frame processing
using the frame.py module to render binary digits based on brightness and depth.
"""

import cv2
import numpy as np
from frame import process_frame

from config import (
    INPUT_VIDEO_ORIGINAL,
    INPUT_VIDEO_DEPTH,
    INPUT_VIDEO_ALPHA,
    OUTPUT_VIDEO_FILENAME,
    OUTPUT_VIDEO_DEBUG_FILENAME,
)


def process_video(original_video_path, depth_video_path, mask_video_path, output_video_path):
    """Process complete video with Matrix-style character effect.
    
    This function reads three synchronized video streams (original RGB, depth map,
    and alpha mask) and processes each frame to generate the final output video
    with rendered characters.
    
    Args:
        original_video_path: Path to source RGB video file
        depth_video_path: Path to depth map video file
        mask_video_path: Path to alpha mask video file
        output_video_path: Path where output video will be saved
    """
    # Open the video files
    cap_original = cv2.VideoCapture(original_video_path)
    cap_depth = cv2.VideoCapture(depth_video_path)
    cap_mask = cv2.VideoCapture(mask_video_path)
    
    # Check if all videos opened successfully
    if not cap_original.isOpened() or not cap_depth.isOpened() or not cap_mask.isOpened():
        print("Error: Could not open one or more video files")
        return
    
    # Extract video properties from original video
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Initialize video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # Process each frame in the video
    while True:
        # Read frames from all three video sources
        ret_original, frame_original = cap_original.read()
        ret_depth, frame_depth = cap_depth.read()
        ret_mask, frame_mask = cap_mask.read()
        
        # Exit loop if any video has ended
        if not ret_original or not ret_depth or not ret_mask:
            break
        
        # Ensure depth and mask are in grayscale format
        if len(frame_depth.shape) == 3:
            frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
        if len(frame_mask.shape) == 3:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        
        # Process frame using frame.py module
        processed_frame = process_frame(frame_original, frame_depth, frame_mask)
        
        # Write processed frame to output video
        out.write(processed_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    # Release all resources
    cap_original.release()
    cap_depth.release()
    cap_mask.release()
    out.release()
    
    print(f"Video processing complete! Output saved to: {output_video_path}")
    print(f"Total frames processed: {frame_count}")


def test_process_video(original_video_path, depth_video_path, mask_video_path, output_debug_frame_path):
    """Process and save the first frame of a video for testing/debugging.
    
    This function is useful for quickly testing processing parameters without
    rendering an entire video. It outputs a single frame for visual inspection.
    
    Args:
        original_video_path: Path to source RGB video file
        depth_video_path: Path to depth map video file
        mask_video_path: Path to alpha mask video file
        output_debug_frame_path: Path where debug frame image will be saved
    """
    # Initialize video capture objects
    cap_original = cv2.VideoCapture(original_video_path)
    cap_depth = cv2.VideoCapture(depth_video_path)
    cap_mask = cv2.VideoCapture(mask_video_path)

    # Read first frame from each video
    ret_original, frame_original = cap_original.read()
    ret_depth, frame_depth = cap_depth.read()
    ret_mask, frame_mask = cap_mask.read()

    # Validate frames were read successfully
    if not ret_original or not ret_depth or not ret_mask:
        print("Error: Could not read the first frame from one or more video files")
        return

    # Ensure depth and mask are in grayscale format
    if len(frame_depth.shape) == 3:
        frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
    if len(frame_mask.shape) == 3:
        frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
    
    # Process first frame
    processed_frame = process_frame(frame_original, frame_depth, frame_mask)
    
    # Save debug output
    cv2.imwrite(output_debug_frame_path, processed_frame)

    # Release video captures
    cap_original.release()
    cap_depth.release()
    cap_mask.release()

    print(f"Test frame processed and saved to: {output_debug_frame_path}")


if __name__ == '__main__':
    """Main execution block for video processing.
    
    This section uses file paths from config.py and runs both
    test frame generation and full video processing.
    """
    
    # Process test frame first (uncomment to run)
    # print("=== Testing with first frame ===")
    # test_process_video(INPUT_VIDEO_ORIGINAL, INPUT_VIDEO_DEPTH, INPUT_VIDEO_ALPHA, OUTPUT_VIDEO_DEBUG_FILENAME)
    
    # Process full video (uncomment to run)
    print("\n=== Processing full video ===")
    process_video(INPUT_VIDEO_ORIGINAL, INPUT_VIDEO_DEPTH, INPUT_VIDEO_ALPHA, OUTPUT_VIDEO_FILENAME)
