"""Audio generation module for video-based sonification.

This module generates audio waveforms from video data by analyzing
brightness and subject matrices. The audio frequency is modulated
based on the ratio of dark pixels within the subject area.
"""

import cv2
import numpy as np
from scipy.io import wavfile
from frame import pixelate_frame, get_brightness_matrix, get_subject_matrix

from config import (
    RESIZE_FACTOR,
    BRIGHTNESS_THRESHOLD,
    BACKGROUND_THRESHOLD,
    BASE_FREQUENCY,
    AMPLITUDE,
    SAMPLE_RATE,
    INPUT_VIDEO_ORIGINAL,
    INPUT_VIDEO_ALPHA,
    OUTPUT_AUDIO_FILENAME,
)


def calculate_matrix_ratio(final_matrix):
    """Calculate the ratio of bright to dark pixels in the subject area.
    
    This ratio is used to modulate the audio frequency, creating
    a sonification of the visual darkness within the subject.
    
    Args:
        final_matrix: Binary matrix combining subject and brightness data
    
    Returns:
        Ratio of bright (0) pixels to dark (1) pixels
    """
    total_elements = final_matrix.size
    if total_elements == 0:
        return 0.0
    
    ones_count = np.sum(final_matrix == 1)
    zeros_count = np.sum(final_matrix == 0)
    
    # Avoid division by zero
    if ones_count == 0:
        return 0.0
    
    # Return ratio of zeros to ones
    ratio = zeros_count / ones_count
    return ratio


def generate_audio_from_video(original_video_path, mask_video_path, output_audio_path):
    """Generate audio waveform from video by analyzing brightness within subject areas.
    
    This function processes each frame to determine the ratio of dark to bright
    pixels within the subject area, then generates a sine wave with frequency
    modulated by this ratio.
    
    Args:
        original_video_path: Path to the source RGB video file
        mask_video_path: Path to the alpha mask video file
        output_audio_path: Path where the output WAV file will be saved
    """
    # Initialize video capture objects
    cap_original = cv2.VideoCapture(original_video_path)
    cap_mask = cv2.VideoCapture(mask_video_path)
    
    # Validate video files were opened successfully
    if not cap_original.isOpened() or not cap_mask.isOpened():
        print("Error: Could not open one or more video files")
        return
    
    # Extract video properties
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {total_frames} frames @ {fps} fps")
    print(f"Base frequency: {BASE_FREQUENCY} Hz, Amplitude sensitivity: {AMPLITUDE}")
    
    # Store frequency modulation values for each frame
    frequency_modulations = []
    frame_count = 0
    
    # Process each frame of the video
    while True:
        # Read frames from both video sources
        ret_original, frame_original = cap_original.read()
        ret_mask, frame_mask = cap_mask.read()
        
        # Exit loop if either video has ended
        if not ret_original or not ret_mask:
            break
        
        # Ensure mask is in grayscale format
        if len(frame_mask.shape) == 3:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        
        # Downsample both frames for analysis
        pixelated_frame = pixelate_frame(frame_original)
        pixelated_mask = pixelate_frame(frame_mask)
        
        # Generate binary matrices for brightness and subject
        brightness_matrix = get_brightness_matrix(pixelated_frame)
        subject_matrix = get_subject_matrix(pixelated_mask)
        
        # Combine matrices to get dark pixels within subject area
        final_matrix = brightness_matrix * subject_matrix
        
        # Calculate the brightness ratio for this frame
        ratio = calculate_matrix_ratio(final_matrix)
        
        # Store ratio for audio synthesis
        frequency_modulations.append(ratio)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    # Release video capture resources
    cap_original.release()
    cap_mask.release()
    
    print(f"Generating audio waveform from {frame_count} frames...")
    
    # Calculate audio timing parameters
    duration_per_frame = 1.0 / fps
    samples_per_frame = int(SAMPLE_RATE * duration_per_frame)
    
    audio_data = []
    
    # Generate audio samples for each frame
    for i, ratio in enumerate(frequency_modulations):
        # Modulate frequency based on the frame's brightness ratio
        # Higher ratio = higher frequency
        frequency = BASE_FREQUENCY * (1.0 + AMPLITUDE * ratio)
        
        # Create time array for this frame's duration
        t = np.linspace(0, duration_per_frame, samples_per_frame, endpoint=False)
        
        # Generate sine wave at the modulated frequency
        wave = np.sin(2 * np.pi * frequency * t)
        
        audio_data.extend(wave)
    
    # Convert to numpy array and prepare for export
    audio_array = np.array(audio_data, dtype=np.float32)
    
    # Normalize to 16-bit PCM integer range (-32768 to 32767)
    audio_array = np.int16(audio_array * 32767)
    
    # Export audio file
    wavfile.write(output_audio_path, SAMPLE_RATE, audio_array)
    
    print(f"Audio generation complete! Output saved to: {output_audio_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Audio duration: {len(audio_array) / SAMPLE_RATE:.2f} seconds")


if __name__ == '__main__':
    """Main execution block for audio generation.
    
    This section uses file paths from config.py and runs
    audio generation.
    """
    
    # Generate audio from video
    generate_audio_from_video(INPUT_VIDEO_ORIGINAL, INPUT_VIDEO_ALPHA, OUTPUT_AUDIO_FILENAME)