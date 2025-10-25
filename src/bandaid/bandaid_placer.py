"""
Automated bandaid placement on detected skin regions.
"""

import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
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


def find_widest_breadth_point(mask, contour, major_axis_vec, minor_axis_vec, center):
    """
    Find the point along the limb where the breadth (width perpendicular to length) is widest.

    Args:
        mask: Binary mask of the limb
        contour: Contour of the limb
        major_axis_vec: Unit vector along the major axis (length)
        minor_axis_vec: Unit vector along the minor axis (breadth)
        center: Center point of the limb

    Returns:
        widest_point: (x, y) coordinates of the center of the widest section
        widest_breadth: The width value at the widest point
        breadth_angle: Angle of the breadth direction in degrees
    """
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Determine scan range along major axis
    # We'll scan from -half_length to +half_length along the major axis
    points = contour.reshape(-1, 2)

    # Project all points onto major axis to find extent
    center_np = np.array(center)
    projections = np.dot(points - center_np, major_axis_vec)
    min_proj, max_proj = projections.min(), projections.max()

    # Sample points along the major axis
    num_samples = 100
    scan_positions = np.linspace(min_proj, max_proj, num_samples)

    max_breadth = 0
    widest_point = center

    for proj in scan_positions:
        # Position along major axis
        scan_center = center_np + proj * major_axis_vec

        # Measure breadth perpendicular to major axis at this position
        # Cast rays in both directions along minor axis
        breadth = 0

        # Check multiple ray lengths to find the actual breadth
        for ray_len in range(1, max(w, h)):
            pos_point = (scan_center + ray_len * minor_axis_vec).astype(int)
            neg_point = (scan_center - ray_len * minor_axis_vec).astype(int)

            # Check if points are within bounds
            if (0 <= pos_point[0] < mask.shape[1] and 0 <= pos_point[1] < mask.shape[0] and
                0 <= neg_point[0] < mask.shape[1] and 0 <= neg_point[1] < mask.shape[0]):

                # If both points are in the mask, breadth continues
                if mask[pos_point[1], pos_point[0]] > 0 and mask[neg_point[1], neg_point[0]] > 0:
                    breadth = ray_len * 2
                else:
                    break
            else:
                break

        if breadth > max_breadth:
            max_breadth = breadth
            widest_point = tuple(scan_center.astype(int))

    # Calculate the angle of the breadth direction (perpendicular to major axis)
    breadth_angle = np.arctan2(minor_axis_vec[1], minor_axis_vec[0]) * 180 / np.pi

    return widest_point, max_breadth, breadth_angle


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


def process_image_with_bandaid(image_path, bandaid_path, output_dir=None, debug=False, bandaid_scale=None):
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

    # Find widest breadth point
    print("Finding widest breadth point...")
    widest_point, max_breadth, breadth_angle = find_widest_breadth_point(
        largest_mask, contour, major_axis_vec, minor_axis_vec, center
    )
    print(f"Widest point: {widest_point}")
    print(f"Max breadth: {max_breadth} pixels")
    print(f"Breadth angle: {breadth_angle:.2f}°")

    # Calculate appropriate bandaid scale based on breadth with padding
    bandaid_img = Image.open(bandaid_path)

    if bandaid_scale is not None:
        scale_factor = bandaid_scale
        print(f"Using manual bandaid scale factor: {scale_factor:.3f}")
    else:
        # Auto-calculate with padding constraint
        # The bandaid's length (width) extends along the breadth direction
        # We want: bandaid_length/2 (half extends from center) + bandaid_length/3 (padding)
        # This means: 0.5L + 0.333L = 0.833L should fit within the mask extent
        # So: 0.833L <= extent from center in one direction
        # Therefore: L <= extent / 0.833

        # Measure actual mask extent from widest point in both directions along breadth
        extent_positive = 0
        extent_negative = 0

        for ray_len in range(1, max(largest_mask.shape)):
            pos_point = (int(widest_point[0] + ray_len * minor_axis_vec[0]),
                        int(widest_point[1] + ray_len * minor_axis_vec[1]))
            neg_point = (int(widest_point[0] - ray_len * minor_axis_vec[0]),
                        int(widest_point[1] - ray_len * minor_axis_vec[1]))

            # Check positive direction
            if (0 <= pos_point[0] < largest_mask.shape[1] and
                0 <= pos_point[1] < largest_mask.shape[0] and
                largest_mask[pos_point[1], pos_point[0]] > 0):
                extent_positive = ray_len

            # Check negative direction
            if (0 <= neg_point[0] < largest_mask.shape[1] and
                0 <= neg_point[1] < largest_mask.shape[0] and
                largest_mask[neg_point[1], neg_point[0]] > 0):
                extent_negative = ray_len

        min_extent = min(extent_positive, extent_negative)
        print(f"Mask extent from widest point: +{extent_positive}px, -{extent_negative}px (using {min_extent}px)")

        # Calculate max bandaid length that allows 1/3 padding
        # min_extent = 0.5L + 0.333L = 0.833L
        max_bandaid_length = min_extent / 0.833

        scale_factor = max_bandaid_length / bandaid_img.width
        print(f"Auto-calculated bandaid scale factor: {scale_factor:.3f} (with 1/3 length padding)")

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
        widest_point,
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
        'widest_point': widest_point,
        'max_breadth': float(max_breadth),
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

        # Widest point visualization
        vis_widest = image_array.copy()
        cv2.circle(vis_widest, widest_point, 10, (0, 255, 0), -1)

        # Draw line across the widest breadth
        line_len = int(max_breadth / 2) + 20
        line_end1 = (int(widest_point[0] + minor_axis_vec[0] * line_len),
                     int(widest_point[1] + minor_axis_vec[1] * line_len))
        line_end2 = (int(widest_point[0] - minor_axis_vec[0] * line_len),
                     int(widest_point[1] - minor_axis_vec[1] * line_len))
        cv2.line(vis_widest, line_end1, line_end2, (255, 255, 0), 3)

        axes[1, 1].imshow(vis_widest)
        axes[1, 1].set_title(f"Widest Point\nBreadth={max_breadth:.0f}px")
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

    return result_image, info


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python bandaid_placer.py <image_path> <bandaid_path> [output_dir] [scale_factor]")
        print("  scale_factor: Optional manual scale (e.g., 0.5, 1.0, 2.0). If not provided, auto-calculates.")
        sys.exit(1)

    image_path = sys.argv[1]
    bandaid_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "bandaid_output"
    bandaid_scale = float(sys.argv[4]) if len(sys.argv) > 4 else None

    result, info = process_image_with_bandaid(
        image_path,
        bandaid_path,
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
