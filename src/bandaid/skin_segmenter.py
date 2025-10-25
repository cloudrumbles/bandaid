"""
MediaPipe-based skin segmentation module for extracting body-skin regions from images.
"""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path


class SkinSegmenter:
    """Extracts body-skin regions from images using MediaPipe MultiClass Selfie Segmentation."""

    # Category mapping from MediaPipe
    CATEGORY_BACKGROUND = 0
    CATEGORY_HAIR = 1
    CATEGORY_BODY_SKIN = 2  # Arms, legs, torso skin
    CATEGORY_FACE_SKIN = 3
    CATEGORY_CLOTHES = 4
    CATEGORY_OTHERS = 5

    def __init__(self, model_path="selfie_multiclass_256x256.tflite"):
        """
        Initialize the skin segmenter with MediaPipe model.

        Args:
            model_path: Path to the MediaPipe segmentation model file.
                       Will download if not present.
        """
        self.model_path = model_path
        self._ensure_model_exists()
        self._initialize_segmenter()

    def _ensure_model_exists(self):
        """Download the model if it doesn't exist locally."""
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe model to {self.model_path}...")
            model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
            import urllib.request
            urllib.request.urlretrieve(model_url, self.model_path)
            print(f"Model downloaded successfully.")

    def _initialize_segmenter(self):
        """Initialize the MediaPipe image segmenter."""
        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False
        )

        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def get_body_skin_mask(self, image_array):
        """
        Extract body-skin mask from an image.

        Args:
            image_array: Input image as numpy array in RGB format (H, W, 3).
                        Values should be uint8 (0-255).

        Returns:
            body_skin_mask: Binary mask (uint8) where 255 represents body-skin pixels
                           and 0 represents non-skin areas. Shape: (H, W)
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_array
        )

        # Run segmentation
        segmentation_result = self.segmenter.segment(mp_image)

        # Extract category mask
        category_mask = segmentation_result.category_mask
        mask_array = category_mask.numpy_view()

        # Extract body-skin (category 2) as binary mask
        body_skin_mask = (mask_array == self.CATEGORY_BODY_SKIN).astype(np.uint8) * 255

        return body_skin_mask

    def get_all_categories(self, image_array):
        """
        Extract all semantic categories from an image.

        Args:
            image_array: Input image as numpy array in RGB format (H, W, 3).

        Returns:
            category_mask: Integer array where each pixel value (0-5) represents
                          the semantic category. Shape: (H, W)
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_array
        )

        # Run segmentation
        segmentation_result = self.segmenter.segment(mp_image)

        # Extract and return category mask
        category_mask = segmentation_result.category_mask
        return category_mask.numpy_view()

    def get_category_statistics(self, image_array):
        """
        Get pixel statistics for each semantic category.

        Args:
            image_array: Input image as numpy array in RGB format (H, W, 3).

        Returns:
            statistics: Dictionary with category names as keys and pixel counts as values.
        """
        mask_array = self.get_all_categories(image_array)
        total_pixels = mask_array.shape[0] * mask_array.shape[1]

        category_names = {
            0: "Background",
            1: "Hair",
            2: "Body-skin",
            3: "Face-skin",
            4: "Clothes",
            5: "Others"
        }

        statistics = {}
        for category_id in range(6):
            category_pixels = np.sum(mask_array == category_id)
            coverage_percent = (category_pixels / total_pixels) * 100
            statistics[category_names[category_id]] = {
                "pixels": int(category_pixels),
                "percentage": round(coverage_percent, 2)
            }

        return statistics

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'segmenter'):
            self.segmenter.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_body_skin_mask(image_array, model_path="selfie_multiclass_256x256.tflite"):
    """
    Convenience function to extract body-skin mask from an image.

    Args:
        image_array: Input image as numpy array in RGB format (H, W, 3).
        model_path: Path to the MediaPipe segmentation model.

    Returns:
        body_skin_mask: Binary mask (uint8) where 255 represents body-skin pixels.
    """
    with SkinSegmenter(model_path) as segmenter:
        return segmenter.get_body_skin_mask(image_array)
