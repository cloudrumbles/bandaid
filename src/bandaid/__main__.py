"""
Main entry point for the Band-Aid Overlay Program
"""
from .placer import BandAidPlacer
import os


def main():
    """Main function to demonstrate the band-aid placer"""
    print("=" * 60)
    print("Band-Aid Overlay Program")
    print("=" * 60)

    # Initialize the band-aid placer (auto-generates bandaid if needed)
    try:
        placer = BandAidPlacer()
        print("✓ Band-aid placer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing band-aid placer: {e}")
        return

    # Check for sample image
    sample_image = "sample_arm.jpg"

    if not os.path.exists(sample_image):
        print(f"\n✗ Sample image '{sample_image}' not found!")
        print("\nTo use this program:")
        print("1. Place an image containing a person with a visible arm in the project directory")
        print("2. Name it 'sample_arm.jpg' or update the 'sample_image' variable in main()")
        print("3. Run this program again")
        print("\nAlternatively, you can use the BandAidPlacer class directly:")
        print("  from bandaid import BandAidPlacer")
        print("  placer = BandAidPlacer()")
        print("  original, result, detected = placer.process_image('your_image.jpg', arm_side='right')")
        print("  placer.display_results(original, result)")
        return

    print(f"✓ Processing image: {sample_image}")

    # Process the image
    original, result, detected = placer.process_image(
        sample_image,
        arm_side='right',  # Change to 'left' for left arm
        show_landmarks=False  # Set to True to see pose landmarks
    )

    if detected:
        print("✓ Arm detected and band-aid applied successfully!")

        # Display and save results
        placer.display_results(original, result, save_path="result_comparison.png")

        # Also save the individual result
        import cv2
        cv2.imwrite("result_with_bandaid.jpg", result)
        print("✓ Result saved to: result_with_bandaid.jpg")
    else:
        print("✗ Could not detect arm in the image")
        print("  Try with a different image showing a clear view of the arm")


if __name__ == "__main__":
    main()