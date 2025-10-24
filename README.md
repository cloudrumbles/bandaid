# Band-Aid Overlay Program 🩹

A Python program that automatically detects a person's arm in a photo and digitally places a band-aid on it. The program uses computer vision and pose estimation to detect arm position and orientation, then overlays a realistic band-aid image that's properly scaled and rotated to match the arm's angle.

## Features

✨ **Automatic Arm Detection**: Uses MediaPipe Pose detection to identify arms in photos
🎯 **Smart Placement**: Automatically calculates the best position on the forearm
📐 **Natural Positioning**: Scales and rotates the band-aid to match the arm's orientation
🖼️ **Side-by-Side Comparison**: Displays original and modified images together
💾 **Save Results**: Outputs both individual and comparison images

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pillow
- Matplotlib

## Installation

1. Clone or download this repository
2. Install the package:

```bash
pip install -e .
```

Or if using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Usage

### Step 1: Prepare Your Image

Place an image containing a person with a visible arm in the project directory and name it `sample_arm.jpg`. The program works best with:
- Clear, well-lit images
- Person facing the camera
- Arm visible (preferably extended)
- Good contrast between person and background

### Step 2: Run the Program

```bash
python -m bandaid
```

Or:

```bash
bandaid
```

The program will:
1. Generate the band-aid image if it doesn't exist
2. Detect the arm in the image
3. Calculate the optimal band-aid placement
4. Apply the band-aid overlay
5. Display both original and modified images
6. Save the results as:
   - `result_with_bandaid.jpg` - The modified image
   - `result_comparison.png` - Side-by-side comparison

## Programmatic Usage

You can also use the `BandAidPlacer` class directly in your code:

```python
from bandaid import BandAidPlacer

# Initialize the placer (band-aid image is auto-generated if needed)
placer = BandAidPlacer()

# Process an image
original, result, detected = placer.process_image(
    'your_image.jpg',
    arm_side='right',  # or 'left'
    show_landmarks=False  # Set True to see pose detection points
)

# Display and save results
if detected:
    placer.display_results(original, result, save_path='comparison.png')
```

## File Structure

```
bandaid/
├── src/
│   └── bandaid/
│       ├── __init__.py      # Package initialization
│       ├── __main__.py      # Main entry point
│       └── placer.py        # BandAidPlacer class
├── tests/
│   └── test_real_image.py   # Test with single image
├── pyproject.toml           # Project configuration
├── README.md                # This file
├── bandaid.png              # Generated band-aid overlay image (auto-created)
├── sample_arm.jpg           # Your input image (you provide this)
├── result_with_bandaid.jpg  # Output: Image with band-aid
└── result_comparison.png    # Output: Side-by-side comparison
```

## How It Works

1. **Pose Detection**: The program uses MediaPipe's Pose estimation model to detect body landmarks, including shoulders, elbows, and wrists.

2. **Arm Identification**: It identifies the arm landmarks (shoulder, elbow, wrist) based on the selected side (left or right).

3. **Band-Aid Placement**: The band-aid is positioned at 40% of the distance from elbow to wrist on the forearm.

4. **Transform Calculation**:
   - **Position**: Calculated based on arm landmarks
   - **Rotation**: Matches the angle of the forearm
   - **Scale**: Proportional to the arm length in the image

5. **Overlay**: The band-aid is overlaid using alpha blending for realistic transparency.

## Configuration

You can customize the band-aid placement by modifying parameters in the code:

- `arm_side`: Choose 'left' or 'right' arm (default: 'right')
- `show_landmarks`: Set to `True` to visualize pose detection points
- `position_ratio`: Adjust where on the forearm the band-aid is placed (default: 0.4)

## Troubleshooting

**"No pose detected in the image!"**
- Ensure the person is clearly visible in the image
- Try with better lighting or a different angle
- Make sure the arm is not obscured

**"Band-aid image not found"**
- The band-aid image should be generated automatically. If not, try deleting any existing `bandaid.png` file and re-running the program.

**Band-aid appears in wrong position**
- Try switching `arm_side` from 'right' to 'left' or vice versa
- Ensure the arm is clearly visible and not bent at unusual angles

## Example Output

The program will display two images side by side:
- **Left**: Original image
- **Right**: Image with digitally applied band-aid

The band-aid will be properly scaled, rotated, and positioned to look natural on the arm.

## Credits

This program uses:
- [MediaPipe](https://google.github.io/mediapipe/) for pose detection
- [OpenCV](https://opencv.org/) for image processing
- [Pillow](https://python-pillow.org/) for image creation

## License

This project is provided as-is for educational and personal use.
