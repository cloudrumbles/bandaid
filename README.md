# Band-Aid Application Program

A Python program that automatically detects arms in photos and digitally places a band-aid on them.

## Features

- **Automatic Arm Detection**: Uses skin color detection to identify arm regions in images
- **Natural Band-Aid Placement**: Automatically scales and rotates the band-aid to fit naturally on the detected arm
- **Side-by-Side Comparison**: Shows both the original and modified images for easy comparison
- **Simple to Use**: Command-line interface with minimal setup required

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

## Installation

### Ubuntu/Debian Systems

```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-pil
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python3 bandaid.py <input_image> [output_image]
```

### Examples

```bash
# Process an image and save the comparison
python3 bandaid.py my_arm_photo.jpg output.jpg

# Process with default output name
python3 bandaid.py my_arm_photo.jpg
```

### Test with Example Image

The repository includes an example image for testing:

```bash
python3 bandaid.py example_arm.jpg
```

## How It Works

1. **Image Loading**: The program loads your input image and the band-aid overlay image
2. **Skin Detection**: Uses HSV and YCrCb color space analysis to detect skin-colored regions
3. **Arm Localization**: Identifies the largest skin region (the arm) and determines optimal band-aid placement
4. **Band-Aid Application**: Overlays the band-aid image with appropriate scaling and rotation
5. **Output Generation**: Creates a side-by-side comparison showing the original and modified images

## Technical Details

- **Skin Detection**: Combines HSV and YCrCb color spaces for robust skin tone detection
- **Morphological Operations**: Uses erosion and dilation to reduce noise in detected regions
- **Contour Analysis**: Finds the largest contour to identify the main arm region
- **Alpha Blending**: Properly blends the band-aid overlay for a natural appearance

## Files

- `bandaid.py` - Main program
- `bandaid.png` - Band-aid overlay image
- `example_arm.jpg` - Example test image
- `requirements.txt` - Python dependencies

## Output

The program generates a comparison image showing:
- **Left side**: Original image
- **Right side**: Image with band-aid applied

If a display is available, the result is shown in a window. The comparison is always saved to the specified output file.

## Demo

![Band-Aid Application Demo](https://github.com/user-attachments/assets/5226e42c-39dc-4254-a67e-c8d91f7d83b9)

The demo shows the program successfully detecting an arm and placing a band-aid on it.
