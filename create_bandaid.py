"""
Simple script to create a bandaid image for testing.
"""
from PIL import Image, ImageDraw
import numpy as np

def create_bandaid_image(width=400, height=120, output_path="bandaid.png"):
    """
    Create a simple bandaid image with transparency.

    Args:
        width: Width of the bandaid
        height: Height of the bandaid
        output_path: Where to save the image
    """
    # Create RGBA image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Main bandaid color (beige/tan)
    bandaid_color = (245, 222, 179, 255)

    # Draw rounded rectangle for main bandaid body
    padding = 10
    draw.rounded_rectangle(
        [(padding, padding), (width - padding, height - padding)],
        radius=20,
        fill=bandaid_color,
        outline=(220, 200, 160, 255),
        width=2
    )

    # Draw center pad (white/gauze color)
    pad_width = width // 3
    pad_height = height - 2 * padding - 20
    pad_x = (width - pad_width) // 2
    pad_y = (height - pad_height) // 2

    draw.rounded_rectangle(
        [(pad_x, pad_y), (pad_x + pad_width, pad_y + pad_height)],
        radius=5,
        fill=(255, 255, 255, 230)
    )

    # Add some texture dots to the adhesive areas (left and right)
    dot_color = (210, 190, 150, 180)
    dot_size = 3

    # Left adhesive area
    for y in range(padding + 15, height - padding - 15, 10):
        for x in range(padding + 15, pad_x - 10, 12):
            draw.ellipse([x, y, x + dot_size, y + dot_size], fill=dot_color)

    # Right adhesive area
    for y in range(padding + 15, height - padding - 15, 10):
        for x in range(pad_x + pad_width + 10, width - padding - 15, 12):
            draw.ellipse([x, y, x + dot_size, y + dot_size], fill=dot_color)

    img.save(output_path)
    print(f"Bandaid image created: {output_path}")
    return img

if __name__ == "__main__":
    create_bandaid_image()
