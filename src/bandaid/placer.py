"""
Band-Aid Overlay Program
This module contains the BandAidPlacer class for detecting arms and placing band-aids.
"""
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os


def create_bandaid_image(width=200, height=80, filename="bandaid.png"):
    """
    Create a band-aid image with transparent background
    
    Args:
        width: Width of the band-aid
        height: Height of the band-aid
        filename: Output filename
    """
    # Create image with transparent background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Band-aid beige/tan color
    bandaid_color = (245, 222, 179, 255)  # Wheat/tan color
    pad_color = (255, 255, 240, 255)  # Ivory color for the pad
    hole_color = (210, 180, 140, 255)  # Tan color for breathable holes
    
    # Draw main band-aid shape (rounded rectangle)
    padding = 5
    draw.rounded_rectangle(
        [(padding, padding), (width - padding, height - padding)],
        radius=15,
        fill=bandaid_color
    )
    
    # Draw central white pad
    pad_width = width // 3
    pad_x = (width - pad_width) // 2
    pad_y = padding + 5
    draw.rounded_rectangle(
        [(pad_x, pad_y), (pad_x + pad_width, height - padding - 5)],
        radius=5,
        fill=pad_color
    )
    
    # Draw small holes on the adhesive parts for texture
    hole_radius = 2
    spacing = 10
    
    # Left side holes
    for x in range(padding + 10, pad_x - 5, spacing):
        for y in range(padding + 10, height - padding - 5, spacing):
            draw.ellipse(
                [(x - hole_radius, y - hole_radius), 
                 (x + hole_radius, y + hole_radius)],
                fill=hole_color
            )
    
    # Right side holes
    for x in range(pad_x + pad_width + 5, width - padding - 5, spacing):
        for y in range(padding + 10, height - padding - 5, spacing):
            draw.ellipse(
                [(x - hole_radius, y - hole_radius), 
                 (x + hole_radius, y + hole_radius)],
                fill=hole_color
            )
    
    # Save the image
    img.save(filename)
    print(f"Band-aid image created: {filename}")
    print(f"Dimensions: {width}x{height}")


class BandAidPlacer:
    def __init__(self, bandaid_path="bandaid.png"):
        """
        Initialize the Band-Aid Placer with MediaPipe Pose detection

        Args:
            bandaid_path: Path to the band-aid image with transparent background
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Load or create band-aid image
        if not os.path.exists(bandaid_path):
            print(f"Band-aid image not found at {bandaid_path}, generating...")
            create_bandaid_image(filename=bandaid_path)
        
        self.bandaid_img = cv2.imread(bandaid_path, cv2.IMREAD_UNCHANGED)
        if self.bandaid_img is None:
            raise FileNotFoundError(f"Failed to load band-aid image: {bandaid_path}")

    def detect_arm(self, image):
        """
        Detect arm landmarks in the image using MediaPipe Pose

        Args:
            image: Input image (BGR format)

        Returns:
            Pose landmarks if detected, None otherwise
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.pose.process(image_rgb)

        return results.pose_landmarks

    def calculate_bandaid_transform(self, landmarks, image_shape, arm_side='right'):
        """
        Calculate position, scale, and rotation for band-aid placement

        Args:
            landmarks: Pose landmarks from MediaPipe
            image_shape: Shape of the input image (height, width, channels)
            arm_side: Which arm to place band-aid on ('left' or 'right')

        Returns:
            Dictionary with position, scale, and angle
        """
        h, w = image_shape[:2]

        # Get arm landmarks based on side
        if arm_side == 'right':
            # Right arm: shoulder (12), elbow (14), wrist (16)
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        else:
            # Left arm: shoulder (11), elbow (13), wrist (15)
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

        # Convert normalized coordinates to pixel coordinates
        elbow_x, elbow_y = int(elbow.x * w), int(elbow.y * h)
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

        # Place band-aid on forearm (between elbow and wrist)
        # Position it at 40% from elbow to wrist
        position_ratio = 0.4
        bandaid_x = int(elbow_x + (wrist_x - elbow_x) * position_ratio)
        bandaid_y = int(elbow_y + (wrist_y - elbow_y) * position_ratio)

        # Calculate angle of the arm
        dx = wrist_x - elbow_x
        dy = wrist_y - elbow_y
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate scale based on arm length
        arm_length = np.sqrt(dx**2 + dy**2)
        scale = arm_length / 400  # Scale band-aid relative to arm length
        scale = max(0.3, min(scale, 2.0))  # Clamp scale between 0.3 and 2.0

        return {
            'position': (bandaid_x, bandaid_y),
            'angle': angle,
            'scale': scale
        }

    def overlay_bandaid(self, image, transform):
        """
        Overlay the band-aid on the image with specified transform

        Args:
            image: Input image (BGR format)
            transform: Dictionary with position, scale, and angle

        Returns:
            Image with band-aid overlay
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Get transform parameters
        position = transform['position']
        angle = transform['angle']
        scale = transform['scale']

        # Resize band-aid
        bandaid_h, bandaid_w = self.bandaid_img.shape[:2]
        new_w = int(bandaid_w * scale)
        new_h = int(bandaid_h * scale)
        bandaid_resized = cv2.resize(self.bandaid_img, (new_w, new_h))

        # Rotate band-aid
        center = (new_w // 2, new_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w_rotated = int((new_h * sin) + (new_w * cos))
        new_h_rotated = int((new_h * cos) + (new_w * sin))

        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w_rotated / 2) - center[0]
        rotation_matrix[1, 2] += (new_h_rotated / 2) - center[1]

        # Apply rotation
        bandaid_rotated = cv2.warpAffine(
            bandaid_resized,
            rotation_matrix,
            (new_w_rotated, new_h_rotated),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Calculate position for overlay (center the band-aid at the position)
        x = position[0] - new_w_rotated // 2
        y = position[1] - new_h_rotated // 2

        # Determine visible overlap between the band-aid and the image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + new_w_rotated, w)
        y2 = min(y + new_h_rotated, h)

        if x1 >= x2 or y1 >= y2:
            return result

        bandaid_x1 = x1 - x
        bandaid_y1 = y1 - y
        bandaid_x2 = bandaid_x1 + (x2 - x1)
        bandaid_y2 = bandaid_y1 + (y2 - y1)

        # Overlay the band-aid using alpha blending
        if bandaid_rotated.shape[2] == 4:  # Has alpha channel
            alpha = bandaid_rotated[bandaid_y1:bandaid_y2, bandaid_x1:bandaid_x2, 3] / 255.0
            overlay_rgb = bandaid_rotated[bandaid_y1:bandaid_y2, bandaid_x1:bandaid_x2, :3]

            for c in range(3):
                result[y1:y2, x1:x2, c] = (
                    alpha * overlay_rgb[:, :, c] +
                    (1 - alpha) * result[y1:y2, x1:x2, c]
                )
        else:
            result[y1:y2, x1:x2] = bandaid_rotated[bandaid_y1:bandaid_y2, bandaid_x1:bandaid_x2, :3]

        return result

    def process_image(self, image_path, arm_side='right', show_landmarks=False):
        """
        Process an image: detect arm and place band-aid

        Args:
            image_path: Path to input image
            arm_side: Which arm to place band-aid on ('left' or 'right')
            show_landmarks: Whether to show pose landmarks on the result

        Returns:
            Tuple of (original_image, result_image, landmarks_detected)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Detect arm
        landmarks = self.detect_arm(image)

        if landmarks is None:
            print("No pose detected in the image!")
            return image, image, False

        # Calculate band-aid transform
        transform = self.calculate_bandaid_transform(landmarks, image.shape, arm_side)

        # Overlay band-aid
        result = self.overlay_bandaid(image, transform)

        # Optionally draw landmarks
        if show_landmarks:
            annotated = result.copy()
            self.mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2
                )
            )
            result = annotated

        return image, result, True

    def display_results(self, original, result, save_path=None):
        """
        Display original and result images side by side

        Args:
            original: Original image
            result: Image with band-aid
            save_path: Optional path to save the comparison image
        """
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(result_rgb)
        axes[1].set_title('With Band-Aid', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")

        plt.show()

    def __del__(self):
        """Cleanup MediaPipe resources"""
        self.pose.close()