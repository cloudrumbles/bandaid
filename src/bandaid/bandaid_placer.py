"""
Automated bandaid placement on detected skin regions.
"""

import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Try relative import first, fall back to absolute
try:
    from .skin_segmenter import SkinSegmenter
except ImportError:
    from skin_segmenter import SkinSegmenter


def find_largest_contiguous_block(binary_mask):
    """
    Find the largest contiguous block in a binary mask.

    Args:
        binary_mask: Binary mask (uint8) where 255 = skin, 0 = background

    Returns:
        largest_mask: Binary mask containing only the largest contiguous region
        contour: The contour of the largest region
    """
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1:  # Only background
        return None, None

    # Find largest component (excluding background at index 0)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create mask with only the largest component
    largest_mask = (labels == largest_idx).astype(np.uint8) * 255

    # Find contour of the largest component
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None

    return largest_mask, contour


def get_limb_orientation(contour):
    """
    Get the orientation of a limb using PCA.

    Args:
        contour: OpenCV contour

    Returns:
        angle: Angle of the major axis in degrees (0-180)
        center: Center point (x, y)
        major_axis_vec: Unit vector along major axis
        minor_axis_vec: Unit vector along minor axis (perpendicular to major)
    """
    # Get all points
    points = contour.reshape(-1, 2).astype(np.float32)

    # Compute PCA
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    center = tuple(mean[0])

    # Major axis (length direction)
    major_axis_vec = eigenvectors[0]
    # Minor axis (breadth direction) - perpendicular to major
    minor_axis_vec = eigenvectors[1]

    # Calculate angle of major axis
    angle = np.arctan2(major_axis_vec[1], major_axis_vec[0]) * 180 / np.pi

    return angle, center, major_axis_vec, minor_axis_vec


def detect_hand_wrist(image_array, mask):
    """
    Detect wrist landmark using MediaPipe Hand Landmarker.

    Args:
        image_array: Original image as RGB numpy array
        mask: Binary mask of the arm region

    Returns:
        wrist_point: (x, y) coordinates of wrist, or None if not detected
    """
    try:
        # Download hand landmarker model if needed
        model_path = "hand_landmarker.task"
        if not Path(model_path).exists():
            print("Downloading hand landmarker model...")
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(model_url, model_path)
            print("Hand landmarker model downloaded.")

        # Initialize hand landmarker
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )

        with vision.HandLandmarker.create_from_options(options) as landmarker:
            # Convert to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_array
            )

            # Detect hands
            detection_result = landmarker.detect(mp_image)

            if detection_result.hand_landmarks:
                # Check each detected hand to see if wrist is within the mask
                for hand_landmarks in detection_result.hand_landmarks:
                    # Wrist is landmark 0
                    wrist = hand_landmarks[0]
                    wrist_x = int(wrist.x * image_array.shape[1])
                    wrist_y = int(wrist.y * image_array.shape[0])

                    # Check if wrist is within the mask
                    if (0 <= wrist_x < mask.shape[1] and
                        0 <= wrist_y < mask.shape[0] and
                        mask[wrist_y, wrist_x] > 0):
                        print(f"Wrist detected at: ({wrist_x}, {wrist_y})")
                        return (wrist_x, wrist_y)

        print("No wrist detected within arm region")
        return None

    except Exception as e:
        print(f"Hand detection failed: {e}")
        return None


def find_q1_breadth_point(mask, contour, major_axis_vec, minor_axis_vec, center, wrist_point=None):
    """
    Sample 200 positions along the arm and find the point where breadth is closest to Q1.

    Args:
        mask: Binary mask of the limb
        contour: Contour of the limb
        major_axis_vec: Unit vector along the major axis (length)
        minor_axis_vec: Unit vector along the minor axis (breadth)
        center: Center point of the limb
        wrist_point: (x, y) coordinates of wrist if detected, or None

    Returns:
        q1_point: (x, y) coordinates of the center at Q1 breadth position
        q1_breadth: The breadth value at Q1 position
        breadth_angle: Angle of the breadth direction in degrees
    """
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Determine scan range along major axis
    points = contour.reshape(-1, 2)

    # Project all points onto major axis to find extent
    center_np = np.array(center)
    projections = np.dot(points - center_np, major_axis_vec)
    min_proj, max_proj = projections.min(), projections.max()

    # Determine hand region if wrist is detected
    hand_region_start = None
    hand_region_end = None
    if wrist_point is not None:
        wrist_np = np.array(wrist_point)
        wrist_proj = np.dot(wrist_np - center_np, major_axis_vec)

        # Project wrist onto major axis to determine hand direction
        # Find the arm extent in both directions from wrist
        dist_to_min = abs(wrist_proj - min_proj)
        dist_to_max = abs(wrist_proj - max_proj)

        # Hand is on the side with shorter distance
        if dist_to_min < dist_to_max:
            hand_region_start = min_proj
            hand_region_end = wrist_proj
            print(f"Hand region identified: from projection {min_proj:.1f} to {wrist_proj:.1f}")
        else:
            hand_region_start = wrist_proj
            hand_region_end = max_proj
            print(f"Hand region identified: from projection {wrist_proj:.1f} to {max_proj:.1f}")

    # Sample 200 points along the major axis
    num_samples = 200
    scan_positions = np.linspace(min_proj, max_proj, num_samples)

    breadth_measurements = []
    position_data = []

    for proj in scan_positions:
        # Skip if in hand region
        if hand_region_start is not None:
            if min(hand_region_start, hand_region_end) <= proj <= max(hand_region_start, hand_region_end):
                continue

        # Position along major axis
        scan_center = center_np + proj * major_axis_vec

        # Measure breadth perpendicular to major axis at this position
        # Count pixels in both directions along minor axis until edge of mask
        breadth_positive = 0
        breadth_negative = 0

        # Measure positive direction
        for ray_len in range(1, max(w, h)):
            pos_point = (scan_center + ray_len * minor_axis_vec).astype(int)
            if (0 <= pos_point[0] < mask.shape[1] and 0 <= pos_point[1] < mask.shape[0] and
                mask[pos_point[1], pos_point[0]] > 0):
                breadth_positive = ray_len
            else:
                break

        # Measure negative direction
        for ray_len in range(1, max(w, h)):
            neg_point = (scan_center - ray_len * minor_axis_vec).astype(int)
            if (0 <= neg_point[0] < mask.shape[1] and 0 <= neg_point[1] < mask.shape[0] and
                mask[neg_point[1], neg_point[0]] > 0):
                breadth_negative = ray_len
            else:
                break

        total_breadth = breadth_positive + breadth_negative

        if total_breadth > 0:  # Only include non-zero breadth measurements
            # Calculate the actual center of the width at this position
            # If breadth is asymmetric, adjust the center point
            offset = (breadth_positive - breadth_negative) / 2.0
            actual_center = scan_center + offset * minor_axis_vec

            breadth_measurements.append(total_breadth)
            position_data.append({
                'projection': proj,
                'center': tuple(actual_center.astype(int)),
                'breadth': total_breadth
            })

    if not breadth_measurements:
        print("No valid breadth measurements found!")
        return center, 0, 0

    # Calculate Q1 (first quartile) of breadth measurements
    q1_breadth = np.percentile(breadth_measurements, 25)
    print(f"Breadth statistics: min={min(breadth_measurements):.1f}, "
          f"Q1={q1_breadth:.1f}, median={np.median(breadth_measurements):.1f}, "
          f"max={max(breadth_measurements):.1f}")

    # Find position with breadth closest to Q1
    min_diff = float('inf')
    q1_position = None
    q1_breadth_actual = 0

    for pos_data in position_data:
        diff = abs(pos_data['breadth'] - q1_breadth)
        if diff < min_diff:
            min_diff = diff
            q1_position = pos_data['center']
            q1_breadth_actual = pos_data['breadth']

    # Calculate the angle of the breadth direction (perpendicular to major axis)
    breadth_angle = np.arctan2(minor_axis_vec[1], minor_axis_vec[0]) * 180 / np.pi

    print(f"Selected Q1 position with breadth {q1_breadth_actual:.1f}px (target Q1: {q1_breadth:.1f}px)")

    return q1_position, q1_breadth_actual, breadth_angle


def place_bandaid(base_image, bandaid_image, position, angle, scale_factor=1.0):
    """
    Place a bandaid on the base image at the specified position and angle.

    Args:
        base_image: Original image (RGB numpy array)
        bandaid_image: Bandaid image (RGBA PIL Image or numpy array)
        position: (x, y) tuple for bandaid center
        angle: Rotation angle in degrees
        scale_factor: Scaling factor for the bandaid

    Returns:
        result_image: Image with bandaid applied
    """
    # Convert to PIL for easier manipulation
    if isinstance(base_image, np.ndarray):
        base_pil = Image.fromarray(base_image)
    else:
        base_pil = base_image.copy()

    # Load bandaid if it's a path
    if isinstance(bandaid_image, (str, Path)):
        bandaid_pil = Image.open(bandaid_image).convert('RGBA')
    elif isinstance(bandaid_image, np.ndarray):
        bandaid_pil = Image.fromarray(bandaid_image)
    else:
        bandaid_pil = bandaid_image.copy()

    # Scale bandaid
    if scale_factor != 1.0:
        new_size = (int(bandaid_pil.width * scale_factor), int(bandaid_pil.height * scale_factor))
        bandaid_pil = bandaid_pil.resize(new_size, Image.Resampling.LANCZOS)

    # Rotate bandaid around its center
    bandaid_rotated = bandaid_pil.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # Calculate paste position (top-left corner)
    paste_x = position[0] - bandaid_rotated.width // 2
    paste_y = position[1] - bandaid_rotated.height // 2

    # Ensure base image is in RGBA mode for alpha compositing
    if base_pil.mode != 'RGBA':
        base_pil = base_pil.convert('RGBA')

    # Paste bandaid with alpha channel
    base_pil.paste(bandaid_rotated, (paste_x, paste_y), bandaid_rotated)

    # Convert back to RGB
    result_image = base_pil.convert('RGB')

    return np.array(result_image)


def process_image_with_bandaid(image_path, output_dir=None, debug=False, bandaid_scale=None, bandaid_path="bandaid.png"):
    """
    Complete pipeline: segment skin, find widest point, place bandaid.

    Args:
        image_path: Path to input image
        bandaid_path: Path to bandaid image
        output_dir: Directory to save outputs (optional)
        debug: If True, save debug visualizations
        bandaid_scale: Manual scale factor for bandaid (if None, auto-calculates)

    Returns:
        result_image: Image with bandaid placed
        info: Dictionary with processing information
    """
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image.convert("RGB"))

    print(f"Processing image: {image_path}")
    print(f"Image shape: {image_array.shape}")

    # Get body-skin mask
    print("Segmenting body-skin...")
    segmenter = SkinSegmenter()
    body_skin_mask = segmenter.get_body_skin_mask(image_array)
    segmenter.close()

    skin_pixels = np.sum(body_skin_mask > 0)
    print(f"Body-skin pixels detected: {skin_pixels}")

    if skin_pixels == 0:
        print("No body-skin detected!")
        return None, None

    # Find largest contiguous block
    print("Finding largest contiguous block...")
    largest_mask, contour = find_largest_contiguous_block(body_skin_mask)

    if contour is None:
        print("No contiguous block found!")
        return None, None

    largest_pixels = np.sum(largest_mask > 0)
    print(f"Largest block size: {largest_pixels} pixels")

    # Get limb orientation
    print("Analyzing limb orientation...")
    angle, center, major_axis_vec, minor_axis_vec = get_limb_orientation(contour)
    print(f"Major axis angle: {angle:.2f}°")
    print(f"Center point: {center}")

    # Detect hand/wrist
    print("Detecting hand/wrist...")
    wrist_point = detect_hand_wrist(image_array, largest_mask)

    # Find Q1 breadth point
    print("Finding Q1 breadth point...")
    q1_point, q1_breadth, breadth_angle = find_q1_breadth_point(
        largest_mask, contour, major_axis_vec, minor_axis_vec, center, wrist_point
    )
    print(f"Q1 point: {q1_point}")
    print(f"Q1 breadth: {q1_breadth} pixels")
    print(f"Breadth angle: {breadth_angle:.2f}°")

    # Calculate appropriate bandaid scale based on breadth
    bandaid_img = Image.open(bandaid_path)

    if bandaid_scale is not None:
        scale_factor = bandaid_scale
        print(f"Using manual bandaid scale factor: {scale_factor:.3f}")
    else:
        # Scale bandaid length to 80% of breadth at Q1 position
        # The bandaid's length (width) should be 80% of the arm breadth
        target_bandaid_length = q1_breadth * 0.8
        scale_factor = target_bandaid_length / bandaid_img.width
        print(f"Auto-calculated bandaid scale factor: {scale_factor:.3f} (80% of breadth: {target_bandaid_length:.1f}px)")

    print(f"Bandaid original size: {bandaid_img.width}x{bandaid_img.height}")
    print(f"Bandaid scaled size: {int(bandaid_img.width * scale_factor)}x{int(bandaid_img.height * scale_factor)}")

    # Place bandaid
    # The bandaid image is horizontal (width > height)
    # We want its length (width) to align with the breadth direction
    # PIL rotates counter-clockwise, and we want the bandaid's length to match breadth_angle
    print("Placing bandaid...")
    result_image = place_bandaid(
        image_array,
        bandaid_path,
        q1_point,
        -breadth_angle,  # Negate for proper PIL rotation direction
        scale_factor
    )

    # Prepare info dictionary
    info = {
        'image_shape': image_array.shape,
        'skin_pixels': int(skin_pixels),
        'largest_block_pixels': int(largest_pixels),
        'center': center,
        'major_axis_angle': float(angle),
        'wrist_point': wrist_point,
        'q1_point': q1_point,
        'q1_breadth': float(q1_breadth),
        'breadth_angle': float(breadth_angle),
        'scale_factor': float(scale_factor)
    }

    # Debug visualizations
    if debug and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create debug visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(image_array)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # Body-skin mask
        axes[0, 1].imshow(body_skin_mask, cmap='gray')
        axes[0, 1].set_title(f"Body-skin Mask\n({skin_pixels} pixels)")
        axes[0, 1].axis('off')

        # Largest block
        axes[0, 2].imshow(largest_mask, cmap='gray')
        axes[0, 2].set_title(f"Largest Block\n({largest_pixels} pixels)")
        axes[0, 2].axis('off')

        # Orientation visualization
        vis_orient = image_array.copy()
        center_int = (int(center[0]), int(center[1]))

        # Draw major axis (red)
        major_end = (int(center[0] + major_axis_vec[0] * 200),
                    int(center[1] + major_axis_vec[1] * 200))
        cv2.arrowedLine(vis_orient, center_int, major_end, (255, 0, 0), 3)

        # Draw minor axis (blue)
        minor_end = (int(center[0] + minor_axis_vec[0] * 100),
                    int(center[1] + minor_axis_vec[1] * 100))
        cv2.arrowedLine(vis_orient, center_int, minor_end, (0, 0, 255), 3)

        axes[1, 0].imshow(vis_orient)
        axes[1, 0].set_title(f"Orientation\nRed=Length, Blue=Breadth")
        axes[1, 0].axis('off')

        # Q1 point visualization
        vis_q1 = image_array.copy()
        cv2.circle(vis_q1, q1_point, 10, (0, 255, 0), -1)

        # Draw wrist if detected
        if wrist_point:
            cv2.circle(vis_q1, wrist_point, 8, (255, 0, 255), -1)

        # Draw line across the Q1 breadth (exactly the measured breadth)
        line_len = int(q1_breadth / 2)
        line_end1 = (int(q1_point[0] + minor_axis_vec[0] * line_len),
                     int(q1_point[1] + minor_axis_vec[1] * line_len))
        line_end2 = (int(q1_point[0] - minor_axis_vec[0] * line_len),
                     int(q1_point[1] - minor_axis_vec[1] * line_len))
        cv2.line(vis_q1, line_end1, line_end2, (255, 255, 0), 3)

        axes[1, 1].imshow(vis_q1)
        wrist_text = " (Wrist detected)" if wrist_point else ""
        axes[1, 1].set_title(f"Q1 Breadth Point{wrist_text}\nBreadth={q1_breadth:.0f}px")
        axes[1, 1].axis('off')

        # Final result
        axes[1, 2].imshow(result_image)
        axes[1, 2].set_title("Result with Bandaid")
        axes[1, 2].axis('off')

        plt.tight_layout()
        debug_path = output_path / 'debug_visualization.png'
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        print(f"Debug visualization saved to: {debug_path}")
        plt.close()

    # Save result
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        result_path = output_path / 'bandaid_result.png'
        result_img = Image.fromarray(result_image)
        result_img.save(result_path)
        print(f"Result saved to: {result_path}")

    # Generate comparison visualization
    if output_dir:
        print("Generating comparison visualization...")
        comparison_path = output_path / 'comparison.png'
        create_comparison_visualization(image_array, result_image, comparison_path)
        print(f"Comparison saved to: {comparison_path}")

    return result_image, info


def create_comparison_visualization(original_image, result_image, save_path):
    """
    Create a side-by-side comparison visualization of original and result images.

    Args:
        original_image: Original image as numpy array (RGB)
        result_image: Result image with bandaid as numpy array (RGB)
        save_path: Path to save the comparison image
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Result image
    axes[1].imshow(result_image)
    axes[1].set_title("With Bandaid", fontsize=16, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python bandaid_placer.py <image_path> [output_dir] [scale_factor]")
        print("  output_dir: Optional output directory (default: bandaid_output)")
        print("  scale_factor: Optional manual scale (e.g., 0.5, 1.0, 2.0). If not provided, auto-calculates.")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "bandaid_output"
    bandaid_scale = float(sys.argv[3]) if len(sys.argv) > 3 else None

    result, info = process_image_with_bandaid(
        image_path,
        output_dir=output_dir,
        debug=True,
        bandaid_scale=bandaid_scale
    )

    if result is not None:
        print("\nProcessing complete!")
        print(f"Info: {info}")
    else:
        print("\nProcessing failed!")


if __name__ == "__main__":
    main()
