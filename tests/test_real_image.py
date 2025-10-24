"""
Test script demonstrating the band-aid placement with downloaded image
"""
from bandaid import BandAidPlacer
import os

def test_with_online_image():
    """Test the band-aid placer with a real downloaded image"""
    
    print("=" * 70)
    print("BAND-AID PLACEMENT TEST - Real Image")
    print("=" * 70)
    
    # Initialize placer
    placer = BandAidPlacer('bandaid.png')
    print("✓ Band-aid placer initialized")
    
    # Check if we have an image
    if not os.path.exists('sample_arm.jpg'):
        print("\n✗ No sample image found")
        print("Downloading sample image...")
        import subprocess
        result = subprocess.run([
            'curl', '-L', '-o', 'sample_arm.jpg',
            'https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=800&q=80'
        ], capture_output=True)
        
        if result.returncode == 0:
            print("✓ Sample image downloaded")
        else:
            print("✗ Failed to download image")
            return
    
    # Process the image - try both arms
    print("\n" + "=" * 70)
    print("Testing with RIGHT arm...")
    print("=" * 70)
    
    original, result_right, detected_right = placer.process_image(
        'sample_arm.jpg',
        arm_side='right',
        show_landmarks=False
    )
    
    if detected_right:
        print("✓ Right arm detected and band-aid applied!")
        placer.display_results(original, result_right, 
                              save_path='result_right_arm.png')
    else:
        print("✗ Right arm not clearly detected")
    
    print("\n" + "=" * 70)
    print("Testing with LEFT arm...")
    print("=" * 70)
    
    original, result_left, detected_left = placer.process_image(
        'sample_arm.jpg',
        arm_side='left',
        show_landmarks=False
    )
    
    if detected_left:
        print("✓ Left arm detected and band-aid applied!")
        placer.display_results(original, result_left, 
                              save_path='result_left_arm.png')
    else:
        print("✗ Left arm not clearly detected")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Right arm detection: {'✓ SUCCESS' if detected_right else '✗ FAILED'}")
    print(f"Left arm detection:  {'✓ SUCCESS' if detected_left else '✗ FAILED'}")
    print("\nGenerated files:")
    if detected_right:
        print("  - result_right_arm.png")
    if detected_left:
        print("  - result_left_arm.png")
    print("=" * 70)

if __name__ == "__main__":
    test_with_online_image()
