import cv2

# Frame processing settings
RESIZE_FACTOR = 10              # Higher = More Pixelated (must be > 1)
BRIGHTNESS_THRESHOLD = 90      # Higher = More Dark Areas
BACKGROUND_THRESHOLD = 20       # Higher = More Subject Area (for mask)
MASK_THRESHOLD = 60             # Higher = More Subject Area (for mask)

BACKGROUND_COLOR = (0, 0, 0)    # Black (BGR format)

# Font rendering settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (0, 255, 0)        # Green (BGR format)
FONT_SCALE_MIN = 0.1           # Minimum font scale (for farthest depth)
FONT_SCALE_MAX = 0.5              # Maximum font scale (for closest depth)
FONT_THICKNESS = 1              

# Audio generation settings
BASE_FREQUENCY = 10.0           # Base frequency in Hz (lower bass range)
AMPLITUDE = 2.5                 # Amplitude sensitivity (higher = more variation)
SAMPLE_RATE = 44100             # Audio sample rate in Hz

# Input/Output paths
# Input filenames
INPUT_IMAGE_ORIGINAL = '../input/image/Original.png'
INPUT_IMAGE_DEPTH = '../input/image/Depth.png'
INPUT_IMAGE_MASK = '../input/image/Alpha.png'

INPUT_VIDEO_ORIGINAL = '../input/video/Original.mov'
INPUT_VIDEO_DEPTH = '../input/video/Depth.mov'
INPUT_VIDEO_ALPHA = '../input/video/Alpha.mov'

# Output filenames
OUTPUT_IMAGE_FILENAME = '../output/image/frame_output.png'
OUTPUT_VIDEO_FILENAME = '../output/video/video_output.mov'
OUTPUT_VIDEO_DEBUG_FILENAME = '../output/video/debug/test_frame_output.png'
OUTPUT_AUDIO_FILENAME = '../output/audio/audio_output.wav'
