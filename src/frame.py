"""Frame processing module for character rendering.

This module takes RGB frames, depth maps, and alpha masks as input and produces
a stylized output frame where the subject is rendered as binary digits
(0 for bright pixels, 1 for dark pixels).
The position of the subject is calculated based on the full resoltion frame
for a smoother motion effect.
The size of the digits is modulated by depth information from the depth map.
"""

import cv2
import numpy as np

from config import (
    FONT_THICKNESS,
    RESIZE_FACTOR,
    BRIGHTNESS_THRESHOLD,
    BACKGROUND_THRESHOLD,
    MASK_THRESHOLD,
    FONT,
    FONT_COLOR,
    FONT_SCALE_MIN,
    FONT_SCALE_MAX,
    BACKGROUND_COLOR,
    INPUT_IMAGE_ORIGINAL,
    INPUT_IMAGE_DEPTH,
    INPUT_IMAGE_MASK,
    OUTPUT_IMAGE_FILENAME,
)

def pixelate_frame(frame):
    """
    Downsample a frame by the specified factor to create a pixelated effect.
    
    Args:
        frame: Input frame as a numpy array (can be grayscale or BGR)
        factor: Downsampling factor (int). Higher values create more pixelation
    
    Returns:
        Pixelated frame with dimensions (width // factor, height // factor)
    """
    height, width = frame.shape[:2]
    pixelated = cv2.resize(frame, (width // RESIZE_FACTOR, height // RESIZE_FACTOR), interpolation=cv2.INTER_LINEAR)
    return pixelated

def get_subject_matrix(pixelated_mask):
    """
    Generate a binary subject matrix from an alpha mask.
    
    Args:
        pixelated_mask: Downsampled grayscale mask frame
    
    Returns:
        Binary numpy array where 1 = subject area, 0 = background
    """
    return np.where(pixelated_mask > BACKGROUND_THRESHOLD, 1, 0)

def get_brightness_matrix(pixelated_frame):
    """
    Generate a binary brightness matrix from an RGB frame.
    
    Args:
        pixelated_frame: Downsampled BGR frame
    
    Returns:
        Binary numpy array where 1 = dark pixel, 0 = bright pixel
    """
    gray_frame = cv2.cvtColor(pixelated_frame, cv2.COLOR_BGR2GRAY)
    return  np.where(gray_frame > BRIGHTNESS_THRESHOLD, 0, 1)

def get_depth_matrix(pixelated_depth_map):
    """
    Extract depth information from a depth map (pass-through function).
    
    Args:
        pixelated_depth_map: Downsampled grayscale depth map
    
    Returns:
        The depth map unchanged
    """
    return pixelated_depth_map

def get_subject_position(mask):
    """
    Find the top-left position of the subject in the full-resolution frame.
    
    Args:
        mask: Full-resolution grayscale alpha mask
    
    Returns:
        Tuple (row, column) of subject's top-left corner, or (0, 0) if no subject found
    """
    subject_indexes = np.argwhere(mask > MASK_THRESHOLD)
    if subject_indexes.size == 0:
        return (0, 0)  # No subject found
    top_row, left_col = subject_indexes.min(axis=0)
    return (top_row, left_col)

def process_frame(original_frame, frame_depth_map, frame_mask):
    """
    Process a single frame to generate a binary code style character rendering.
    
    This function combines brightness, depth, and mask information to render
    binary digits (0 or 1) on a black background. Characters are sized based
    on depth using cubic mapping and only appear within the subject area.
    
    Args:
        original_frame: Source RGB frame (BGR format)
        frame_depth_map: Grayscale depth map where lighter = closer
        frame_mask: Binary alpha mask where white = subject, black = background
    
    Returns:
        Processed frame with rendered digits on black background
    """
    
    # Step 1: Pixelate the input frames
    pixelated_frame = pixelate_frame(original_frame)
    pixelated_depth_map = pixelate_frame(frame_depth_map)
    pixelated_mask = pixelate_frame(frame_mask)

    # Step 2: Generate matrices from the pixelated frames
    brightness_matrix = get_brightness_matrix(pixelated_frame)
    subject_matrix = get_subject_matrix(pixelated_mask)
    depth_matrix = get_depth_matrix(pixelated_depth_map)

    # Get bounding box from subject matrix
    subject_indexes = np.argwhere(subject_matrix == 1)
    if subject_indexes.size == 0:
        # No subject found, return black frame
        return np.zeros_like(original_frame)
    
    top_row, left_col = subject_indexes.min(axis=0)
    bottom_row, right_col = subject_indexes.max(axis=0)
    
    # Crop all matrices to the same subject bounding box
    cropped_brightness_matrix = brightness_matrix[top_row:bottom_row + 1, left_col:right_col + 1]
    cropped_depth_matrix = depth_matrix[top_row:bottom_row + 1, left_col:right_col + 1]
    cropped_subject_matrix = subject_matrix[top_row:bottom_row + 1, left_col:right_col + 1]

    # Get subject position
    subject_position = get_subject_position(frame_mask)

    # Create a blank frame for rendering
    final_frame = np.full_like(original_frame, BACKGROUND_COLOR)
    
    # Render digits on the final frame
    matrix_height, matrix_width = cropped_subject_matrix.shape[:2]
    
    for row in range(matrix_height):
        for col in range(matrix_width):
            
            if cropped_subject_matrix[row, col] > 0:
                
                digit_type = cropped_brightness_matrix[row, col]
                
                depth_value = cropped_depth_matrix[row, col]
                
                y_pos = subject_position[0] + (row * RESIZE_FACTOR) + (RESIZE_FACTOR // 2)
                x_pos = subject_position[1] + (col * RESIZE_FACTOR) + (RESIZE_FACTOR // 2)
                
                digit_text = str(digit_type)
                
                # Scale font size according to depth value using cubic mapping for more dramatic size differences
                font_scale = FONT_SCALE_MIN + ((depth_value / 255.0) ** 3) * (FONT_SCALE_MAX - FONT_SCALE_MIN)
                
                # Get text size to center it at position
                (text_width, text_height), baseline = cv2.getTextSize(digit_text, FONT, font_scale, FONT_THICKNESS)
                
                # Adjust position to center the text
                x = int(x_pos - text_width // 2)
                y = int(y_pos + text_height // 2)
                
                # Render the digit on the frame
                cv2.putText(final_frame, digit_text, (x, y), FONT, font_scale, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    return final_frame

if __name__ == '__main__':
    """Main execution block for frame processing.
    
    This section loads test images and processes a single frame.
    File paths are configured in config.py.
    """
    
    # Load input images
    original_frame = cv2.imread(INPUT_IMAGE_ORIGINAL)
    frame_depth_map = cv2.imread(INPUT_IMAGE_DEPTH, cv2.IMREAD_GRAYSCALE)
    frame_mask = cv2.imread(INPUT_IMAGE_MASK, cv2.IMREAD_GRAYSCALE)
    
    # Validate images loaded successfully
    if original_frame is None or frame_depth_map is None or frame_mask is None:
        print("Error: Could not load one or more input images. Check file paths.")
    else:
        # Process frame
        final_frame = process_frame(original_frame, frame_depth_map, frame_mask)
        
        # Save output
        cv2.imwrite(OUTPUT_IMAGE_FILENAME, final_frame)
        print(f"Frame processed successfully! Output saved to: {OUTPUT_IMAGE_FILENAME}")