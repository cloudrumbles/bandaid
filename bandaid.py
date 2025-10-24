#!/usr/bin/env python3
"""
Band-Aid Application Program
This program detects arms in images and digitally places a band-aid on them.
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def detect_skin_regions(image):
    """
    Detect skin-colored regions in the image using color detection.
    
    Args:
        image: BGR image from OpenCV
    
    Returns:
        mask: Binary mask where skin regions are white
    """
    # Convert to different color spaces for better skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color ranges in HSV
    # These ranges work well for various skin tones
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    
    # Define skin color ranges in YCrCb
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    
    # Create masks for both color spaces
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combine masks
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Dilate to make regions more prominent
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def find_arm_location(mask, image_shape):
    """
    Find a suitable location on the arm to place the band-aid.
    
    Args:
        mask: Binary mask of skin regions
        image_shape: Shape of the original image
    
    Returns:
        (x, y, angle): Position and angle for band-aid placement
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No skin detected, place in center
        return (image_shape[1] // 2, image_shape[0] // 2, 0)
    
    # Find the largest contour (likely the arm)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the rotated bounding box for the contour
    if len(largest_contour) >= 5:
        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (width, height), angle = rect
    else:
        # Fallback to moments
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = image_shape[1] // 2, image_shape[0] // 2
        angle = 0
    
    # Place band-aid slightly offset from center of arm region
    x = int(cx)
    y = int(cy)
    
    return (x, y, angle)


def overlay_bandaid(image, bandaid_img, position, scale=1.0):
    """
    Overlay the band-aid image onto the arm image at the specified position.
    
    Args:
        image: Original BGR image
        bandaid_img: Band-aid image with alpha channel
        position: (x, y, angle) tuple for placement
        scale: Scale factor for the band-aid
    
    Returns:
        result: Image with band-aid overlay
    """
    x, y, angle = position
    result = image.copy()
    
    # Resize band-aid if needed
    if scale != 1.0:
        new_width = int(bandaid_img.shape[1] * scale)
        new_height = int(bandaid_img.shape[0] * scale)
        bandaid_img = cv2.resize(bandaid_img, (new_width, new_height))
    
    # Rotate band-aid
    if angle != 0:
        center = (bandaid_img.shape[1] // 2, bandaid_img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        bandaid_img = cv2.warpAffine(bandaid_img, rotation_matrix, 
                                      (bandaid_img.shape[1], bandaid_img.shape[0]),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0, 0))
    
    # Calculate placement coordinates
    h, w = bandaid_img.shape[:2]
    x1 = max(0, x - w // 2)
    y1 = max(0, y - h // 2)
    x2 = min(result.shape[1], x1 + w)
    y2 = min(result.shape[0], y1 + h)
    
    # Adjust band-aid crop if it extends beyond image
    bx1 = 0 if x1 >= 0 else -(x - w // 2)
    by1 = 0 if y1 >= 0 else -(y - h // 2)
    bx2 = w if x2 <= result.shape[1] else w - ((x1 + w) - result.shape[1])
    by2 = h if y2 <= result.shape[0] else h - ((y1 + h) - result.shape[0])
    
    # Handle alpha channel if present
    if bandaid_img.shape[2] == 4:
        # Band-aid has alpha channel
        bandaid_crop = bandaid_img[by1:by2, bx1:bx2]
        alpha = bandaid_crop[:, :, 3] / 255.0
        
        # Blend band-aid with background
        for c in range(3):
            result[y1:y2, x1:x2, c] = (alpha * bandaid_crop[:, :, c] +
                                       (1 - alpha) * result[y1:y2, x1:x2, c])
    else:
        # No alpha channel, simple overlay
        result[y1:y2, x1:x2] = bandaid_img[by1:by2, bx1:bx2]
    
    return result


def process_image(input_path, bandaid_path, output_path=None, scale=0.8):
    """
    Main function to process an image and apply a band-aid.
    
    Args:
        input_path: Path to input image
        bandaid_path: Path to band-aid image
        output_path: Optional path to save output image
        scale: Scale factor for band-aid size
    """
    # Load the input image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Could not load image from {input_path}")
        return
    
    # Load the band-aid image
    bandaid = cv2.imread(str(bandaid_path), cv2.IMREAD_UNCHANGED)
    if bandaid is None:
        print(f"Error: Could not load band-aid image from {bandaid_path}")
        return
    
    # Add alpha channel if not present
    if len(bandaid.shape) < 3:
        print(f"Error: Band-aid image must be a color image")
        return
    
    if bandaid.shape[2] == 3:
        # Create alpha channel (make black pixels transparent)
        alpha = np.ones((bandaid.shape[0], bandaid.shape[1], 1), dtype=np.uint8) * 255
        # Make very dark pixels semi-transparent
        gray = cv2.cvtColor(bandaid, cv2.COLOR_BGR2GRAY)
        alpha[gray < 50] = 0
        bandaid = np.concatenate([bandaid, alpha], axis=2)
    
    print("Detecting skin regions...")
    skin_mask = detect_skin_regions(image)
    
    print("Finding optimal band-aid placement...")
    position = find_arm_location(skin_mask, image.shape)
    
    print(f"Placing band-aid at position ({position[0]}, {position[1]}) with angle {position[2]:.1f}°")
    result = overlay_bandaid(image, bandaid, position, scale=scale)
    
    # Create side-by-side comparison
    h1, w1 = image.shape[:2]
    h2, w2 = result.shape[:2]
    
    # Make them the same height
    max_height = max(h1, h2)
    
    # Create canvas
    comparison = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
    
    # Place original image on left
    comparison[0:h1, 0:w1] = image
    
    # Place result on right
    comparison[0:h2, w1:w1+w2] = result
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "With Band-Aid", (w1 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(str(output_path), comparison)
        print(f"Saved comparison image to {output_path}")
    
    # Display the results (only if DISPLAY is available)
    import os
    if os.environ.get('DISPLAY'):
        print("\nDisplaying results...")
        print("Press any key to close the window")
        
        # Resize for display if image is very large
        display_width = 1600
        if comparison.shape[1] > display_width:
            scale_factor = display_width / comparison.shape[1]
            new_width = display_width
            new_height = int(comparison.shape[0] * scale_factor)
            comparison = cv2.resize(comparison, (new_width, new_height))
        
        cv2.imshow('Band-Aid Application - Original vs Modified', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nNo display available. Results saved to file.")
    
    return result


def main():
    """Main entry point for the program."""
    # Check for input image
    if len(sys.argv) < 2:
        print("Usage: python3 bandaid.py <input_image> [output_image]")
        print("\nExample: python3 bandaid.py example_arm.jpg output.jpg")
        return
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input image '{input_path}' not found")
        return
    
    # Check for band-aid image
    bandaid_path = Path("bandaid.png")
    if not bandaid_path.exists():
        print(f"Error: Band-aid image 'bandaid.png' not found")
        print("Please ensure bandaid.png is in the current directory")
        return
    
    # Output path
    output_path = None
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = Path(input_path.stem + "_bandaid" + input_path.suffix)
    
    # Process the image
    process_image(input_path, bandaid_path, output_path)


if __name__ == "__main__":
    main()
