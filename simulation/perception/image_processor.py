"""
Image Processing Module
Performs preprocessing, resizing, normalization, and filtering on RGB and Depth streams.
"""

from typing import Tuple
import numpy as np


class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (84, 84)):
        self.target_width, self.target_height = target_size

    def resize_rgb(self, image: np.ndarray) -> np.ndarray:
        """Resizes an image using bilinear interpolation (Pure NumPy/SciPy compatible)."""
        try:
            import cv2
            return cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback fast downsampling via slicing if cv2 not installed
            h_step = max(1, image.shape[0] // self.target_height)
            w_step = max(1, image.shape[1] // self.target_width)
            return image[::h_step, ::w_step][:self.target_height, :self.target_width]

    def normalize_rgb(self, image: np.ndarray) -> np.ndarray:
        """Converts uint8 [0, 255] to float32 [0.0, 1.0]."""
        return (image.astype(np.float32) / 255.0)

    def filter_depth(self, depth: np.ndarray, max_range: float = 5.0) -> np.ndarray:
        """Clips depth to max sensor range and normalizes to [0.0, 1.0]."""
        clipped = np.clip(depth, 0.0, max_range)
        normalized = clipped / max_range
        return normalized.astype(np.float32)

    def to_grayscale(self, rgb: np.ndarray) -> np.ndarray:
        """Converts RGB image to single-channel luminance."""
        # Standard Rec.601 luma coefficients
        return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)

    def process_for_rl(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Prepares a stacked (4, H, W) visual observation for neural network consumption."""
        resized_rgb = self.resize_rgb(rgb)
        norm_rgb = self.normalize_rgb(resized_rgb)
        
        try:
            import cv2
            resized_depth = cv2.resize(depth, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            h_step = max(1, depth.shape[0] // self.target_height)
            w_step = max(1, depth.shape[1] // self.target_width)
            resized_depth = depth[::h_step, ::w_step][:self.target_height, :self.target_width]
            
        norm_depth = self.filter_depth(resized_depth)
        
        # Stack channels: R, G, B, Depth -> Shape (H, W, 4)
        stacked = np.dstack([norm_rgb, norm_depth])
        return stacked
