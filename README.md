# Depth-Aware Binary Rendering Tool

A Python-based video processing toolkit that creates binary-style visual effects by rendering video content as0s and 1s
with depth-aware character sizing. The program also generates audio through sonification of visual brightness patterns.

## Features

- **Binary Digit Rendering**: Converts video frames into stylized binary code displays where digits represent brightness levels
- **Depth-Aware Scaling**: Uses depth maps to dynamically scale character sizes (closer objects appear larger)
- **Smooth Subject Motion**: Calculates the subject position from the full resolution alpha frame for a smoother motion effect
- **Pixelated Processing**: Configurable pixelation effect for artistic styling
- **Audio Generation**: Creates audio waveforms synchronized with video brightness patterns (sonification)
- **Batch Processing**: Supports both single image and full video processing
- **Debug**: Quick frame preview for testing parameters before full video rendering

## Installation

### Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- SciPy

### Setup

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install opencv-python numpy scipy
```

3. Create the required directory structure (if not already present):
```
GFX Assisted Program/
├── input/
│   ├── image/
│   └── video/
├── output/
│   ├── audio/
│   ├── image/
│   └── video/
│       └── debug/
└── src/
    ├── config.py
    ├── frame.py
    ├── video.py
    └── audio.py
```

## Usage

### Processing Single Images

1. Place your input files in `input/image/`:
   - `Original.jpg` - Source RGB image
   - `Depth.jpg` - Grayscale depth map (lighter = closer)
   - `Mask.jpg` - Alpha mask (white = subject, black = background)

2. Run the frame processor:
```bash
cd src
python frame.py
```

3. Find output in `output/image/frame_output.png`

### Processing Videos

1. Place your input videos in `input/video/`:
   - `Original.mov` - Source RGB video
   - `Depth.mov` - Depth map video
   - `Alpha.mov` - Alpha mask video

2. **Test First Frame** (recommended):
   
   Edit [video.py](src/video.py) and uncomment the test section:
   ```python
   # Process test frame first (uncomment to run)
   print("=== Testing with first frame ===")
   test_process_video(original_video, depth_video, alpha_video, output_debug_frame)
   ```
   
   Run:
   ```bash
   python video.py
   ```
   
   Check `output/video/debug/test_frame_output.png`

3. **Process Full Video**:
   
   Ensure the full processing section is active:
   ```python
   print("\n=== Processing full video ===")
   process_video(original_video, depth_video, alpha_video, output_video)
   ```
   
   Find output in `output/video/video_output.mov`

### Generating Audio

1. Place your input videos in `input/video/`:
   - `Original.mov` - Source RGB video
   - `Alpha.mov` - Alpha mask video

2. Run the audio generator:
```bash
python audio.py
```

3. Find output in `output/audio/audio_output.wav`

## Configuration

All parameters can be adjusted in [config.py](src/config.py):

### Frame Processing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RESIZE_FACTOR` | 10 | Pixelation intensity (higher = more pixelated) |
| `BRIGHTNESS_THRESHOLD` | 90 | Threshold for dark/bright classification (0-255) |
| `BACKGROUND_THRESHOLD` | 20 | Threshold for subject detection in mask |
| `MASK_THRESHOLD` | 60 | Threshold for precise subject positioning |

### Font Rendering Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FONT` | `cv2.FONT_HERSHEY_SIMPLEX` | OpenCV font type |
| `FONT_COLOR` | `(0, 255, 0)` | Green in BGR format |
| `FONT_SCALE_MIN` | 0.1 | Minimum font size (farthest depth) |
| `FONT_SCALE_MAX` | 0.5 | Maximum font size (closest depth) |
| `FONT_THICKNESS` | 1 | Character line thickness |
| `BACKGROUND_COLOR` | `(0, 0, 0)` | Black background in BGR |

### Audio Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_FREQUENCY` | 10.0 | Base frequency in Hz (bass range) |
| `AMPLITUDE` | 2.5 | Frequency modulation sensitivity |
| `SAMPLE_RATE` | 44100 | Audio sample rate in Hz |

### File Paths

Input and output paths are configurable. Default filenames:
- **Input Images**: `Original.jpg`, `Depth.jpg`, `Mask.jpg`
- **Input Videos**: `Original.mov`, `Depth.mov`, `Alpha.mov`
- **Output**: Auto-named files in respective output directories

**Note on Cross-Platform Compatibility**: All file paths in [config.py](src/config.py) use forward slashes (`/`) which work correctly on both Windows and Linux/WSL. Do not change these to backslashes (`\`) as this will cause path resolution issues in Linux environments.

## How It Works

### Frame Processing Pipeline

1. **Pixelation**: Input frames are downsampled by `RESIZE_FACTOR` to create pixelated blocks
2. **Matrix Generation**:
   - **Brightness Matrix**: Binary classification of each pixel (dark = 1, bright = 0)
   - **Subject Matrix**: Binary mask indicating subject area
   - **Depth Matrix**: Depth information for each pixel
3. **Subject Detection**: Identifies bounding box of subject from full-resolution mask
4. **Character Rendering**: For each pixel block in the subject area:
   - Determines digit (0 or 1) based on brightness
   - Calculates position from full-res mask for smooth motion
   - Scales font size using cubic mapping of depth value: 
     ```python
     font_scale = MIN + ((depth/255)³) × (MAX - MIN)
     ```
   - Renders digit on black background

### Audio Generation Pipeline

1. **Frame Analysis**: For each video frame:
   - Pixelate original and mask frames
   - Generate brightness and subject matrices
   - Calculate ratio of dark-to-bright pixels within subject area
2. **Frequency Modulation**: 
   ```python
   frequency = BASE_FREQUENCY × (1.0 + AMPLITUDE × ratio)
   ```
3. **Waveform Synthesis**: Generate sine wave at modulated frequency for frame duration
4. **Export**: Concatenate all frame audio and export as 16-bit PCM WAV file

### Depth Mapping

The depth map controls character sizing using a **cubic scaling function** for dramatic size variation:
- **Lighter pixels** (closer to camera) → Larger characters (up to `FONT_SCALE_MAX`)
- **Darker pixels** (farther away) → Smaller characters (down to `FONT_SCALE_MIN`)

The cubic relationship `(depth/255)³` creates pronounced differences between foreground and background.

## Module Documentation

### [frame.py](src/frame.py)
Core frame processing module. Contains:
- `pixelate_frame()`: Downsamples frames for pixelation effect
- `get_brightness_matrix()`: Converts RGB to binary brightness classification
- `get_subject_matrix()`: Extracts subject mask from alpha channel
- `get_depth_matrix()`: Processes depth information
- `get_subject_position()`: Calculates subject position in full-resolution frame
- `process_frame()`: Main processing pipeline for single frames

### [video.py](src/video.py)
Video stream processing module. Contains:
- `process_video()`: Processes complete video with three input streams
- `test_process_video()`: Renders first frame only for quick testing

### [audio.py](src/audio.py)
Audio generation module. Contains:
- `calculate_matrix_ratio()`: Computes brightness ratio for frequency modulation
- `generate_audio_from_video()`: Main audio synthesis pipeline

### [config.py](src/config.py)
Central configuration file with all adjustable parameters and file paths.

## Tips & Best Practices

1. **Start with Test Frames**: Always use `test_process_video()` to preview effects before processing full videos
2. **Iterate on Parameters**: Adjust `config.py` settings and re-run tests until desired effect is achieved
3. **Optimization**: 
   - Higher `RESIZE_FACTOR` = faster processing but chunkier appearance
   - Lower `RESIZE_FACTOR` = more detail but slower processing
4. **Depth Map Quality**: Better depth maps produce more dramatic and realistic 3D effects
5. **Color Customization**: Change `FONT_COLOR` and `BACKGROUND_COLOR` for different aesthetic styles
6. **Audio Tuning**: Adjust `BASE_FREQUENCY` and `AMPLITUDE` to change audio characteristics

## Input Requirements

### Depth Maps
- Grayscale images/videos where pixel brightness represents distance
- Lighter = closer to camera
- Darker = farther from camera
- Should match dimensions of original content

### Alpha Masks
- Grayscale or binary images/videos
- White (255) = subject area
- Black (0) = background
- Should match dimensions of original content

### Synchronization
All three video inputs must be:
- Same resolution
- Same frame count
- Same frame rate
- Temporally aligned

## Troubleshooting

**Error: "Could not load one or more input images"**
- Check that input files exist in the correct directories
- Verify filenames match those specified in `config.py`
- Ensure image files are valid and not corrupted

**Error: "Could not open one or more video files"**
- Verify video codecs are supported by OpenCV
- Check file paths in `config.py`
- Ensure all three video files exist

**No output or black frames**
- Check `MASK_THRESHOLD` and `BACKGROUND_THRESHOLD` values
- Verify alpha mask has sufficient white areas
- Increase brightness threshold if no digits appear

**Characters too small/large**
- Adjust `FONT_SCALE_MIN` and `FONT_SCALE_MAX`
- Verify depth map has good contrast
- Check `RESIZE_FACTOR` isn't too high/low

**Audio too quiet or distorted**
- Adjust `AMPLITUDE` (lower for subtler changes)
- Modify `BASE_FREQUENCY` to change pitch range
- Check that videos have clear subject/background separation

## Technical Details

- **Color Space**: BGR (OpenCV default)
- **Video Codec**: MP4V (MPEG-4)
- **Audio Format**: 16-bit PCM WAV, 44.1 kHz
- **Interpolation**: Linear interpolation for downsampling
- **Text Rendering**: Anti-aliased (cv2.LINE_AA)

## Future Enhancements

Potential improvements for future versions:
- Support for additional video codecs
- Real-time processing mode
- GPU acceleration with CUDA
- Alternative character sets beyond binary digits
- Color-based audio channel separation
- Interactive parameter tuning GUI
- Batch processing for multiple videos

## License

This project is provided as-is for portfolio demonstration purposes.

## Author

Created as part of a portfolio project demonstrating computer vision and audio synthesis techniques.

---

**Note**: This program requires three synchronized input streams (original, depth, alpha) for full functionality. Depth maps and alpha masks can be generated using various tools such as depth estimation neural networks or video editing software with rotoscoping capabilities usch as Adobe Premiere Pro, Adobe After Effects or DaVinci Resolve.
