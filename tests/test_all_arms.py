"""
Test script to process all arm images and generate a single comparison visualization.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandaid.bandaid_placer import process_image_with_bandaid


def test_all_arms():
    """Process all arm images and create a single comparison visualization."""

    # Define test images
    assets_dir = Path(__file__).parent.parent / "assets" / "sample_arms"
    test_images = [
        assets_dir / "new_arm2.jpg",
    ]

    # Output directory
    output_dir = Path(__file__).parent.parent / "test_output_all"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("TESTING ALL ARM IMAGES")
    print("=" * 80)
    print()

    results = []

    # Process each image
    for image_path in test_images:
        if not image_path.exists():
            print(f"⚠ Skipping {image_path.name} - file not found")
            continue

        print(f"\nProcessing: {image_path.name}")
        print("-" * 80)

        try:
            # Create individual output directory for debug visualizations
            individual_output = output_dir / image_path.stem
            individual_output.mkdir(exist_ok=True)

            result_image, info = process_image_with_bandaid(
                str(image_path),
                output_dir=str(individual_output),  # Save individual outputs with debug
                debug=True
            )

            if result_image is not None:
                # Load original image
                original_image = np.array(Image.open(image_path).convert("RGB"))

                results.append({
                    'name': image_path.stem,
                    'original': original_image,
                    'result': result_image,
                    'info': info
                })
                print(f"✓ Success - Q1 breadth: {info['q1_breadth']:.0f}px, "
                      f"Wrist detected: {info['wrist_point'] is not None}")
                print(f"  Debug files saved to: {individual_output}")
            else:
                print(f"✗ Failed to process {image_path.name}")

        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Create combined comparison visualization
    if results:
        print("\n" + "=" * 80)
        print("GENERATING COMBINED COMPARISON")
        print("=" * 80)

        num_images = len(results)
        fig, axes = plt.subplots(num_images, 2, figsize=(16, 8 * num_images))

        # Handle case of single image
        if num_images == 1:
            axes = axes.reshape(1, -1)

        for idx, result in enumerate(results):
            # Original image
            axes[idx, 0].imshow(result['original'])
            axes[idx, 0].set_title(f"{result['name']} - Original",
                                   fontsize=14, fontweight='bold')
            axes[idx, 0].axis('off')

            # Result image
            axes[idx, 1].imshow(result['result'])
            info = result['info']
            wrist_status = "Wrist detected" if info['wrist_point'] else "No wrist"
            axes[idx, 1].set_title(
                f"{result['name']} - With Bandaid\n"
                f"Q1 breadth: {info['q1_breadth']:.0f}px | {wrist_status}",
                fontsize=14, fontweight='bold'
            )
            axes[idx, 1].axis('off')

        plt.tight_layout()

        # Save comparison
        comparison_path = output_dir / 'all_arms_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Combined comparison saved to: {comparison_path}")
        print(f"  - Processed {len(results)} images successfully")

    else:
        print("\n✗ No images were processed successfully")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_all_arms()
