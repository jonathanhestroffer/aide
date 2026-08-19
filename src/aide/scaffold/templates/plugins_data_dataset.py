from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from aide.registry.registries import DatasetRegistry


@DatasetRegistry.register("procedural_shapes")
class ProceduralShapesDataset(Dataset):
    """Synthetic multi-class dataset with randomized shapes, positions, scale,
    rotation, colors, and background clutter.
    """

    def __init__(self, num_samples: int = 500, img_size: int = 32, noise_level: float = 0.15):
        """Initialize the ProceduralShapesDataset.

        Args:
            num_samples (int): Number of samples in the dataset.
            img_size (int): Size of the square image (img_size x img_size).
            noise_level (float): Standard deviation of Gaussian noise added to the images.
        """
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.noise_level = noise_level
        self.num_classes = 3  # 0: Circle, 1: Square, 2: Triangle

    def __len__(self) -> int:
        return self.num_samples

    def _draw_shape(
        self,
        canvas: np.ndarray,
        shape_type: int,
        center: tuple[int, int],
        size: int,
        color: np.ndarray,
    ):
        """Draw a shape on the given canvas.

        Args:
            canvas (np.ndarray): The image canvas to draw on.
            shape_type (int): Type of shape (0: Circle, 1: Square, 2: Triangle).
            center (tuple[int, int]): Center coordinates (y, x) of the shape.
            size (int): Size of the shape.
            color (np.ndarray): RGB color of the shape.
        """
        cy, cx = center
        y, x = np.ogrid[: self.img_size, : self.img_size]

        if shape_type == 0:  # Circle
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= (size // 2) ** 2

        elif shape_type == 1:  # Square (Rotated)
            # Apply 2D rotation matrix coordinates
            angle = np.random.uniform(0, np.pi / 2)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = cos_a * (x - cx) - sin_a * (y - cy)
            yr = sin_a * (x - cx) + cos_a * (y - cy)
            half = size // 2
            mask = (np.abs(xr) <= half) & (np.abs(yr) <= half)

        elif shape_type == 2:  # Triangle
            # Bounding box approximation for an upright/tilted triangle
            half = size // 2
            h = int(size * 0.866)
            mask = (
                (y >= cy - half)
                & (y <= cy + half)
                & (np.abs(x - cx) <= (cy + half - y) * (half / h))
            )

        # Paint RGB color onto mask
        canvas[mask] = color

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # 1. Random background color (RGB)
        bg_color = np.random.uniform(0.1, 0.4, size=(1, 1, 3))
        canvas = np.ones((self.img_size, self.img_size, 3)) * bg_color

        # 2. Random shape, position, size, and distinct foreground color
        label = idx % self.num_classes
        size = np.random.randint(6, self.img_size // 2)
        margin = size // 2 + 2
        center = (
            np.random.randint(margin, self.img_size - margin),
            np.random.randint(margin, self.img_size - margin),
        )

        fg_color = np.random.uniform(0.5, 1.0, size=(3,))
        self._draw_shape(canvas, label, center, size, fg_color)

        # 3. Add background clutter (random distractor line segments)
        num_distractors = np.random.randint(1, 4)
        for _ in range(num_distractors):
            lx, ly = np.random.randint(0, self.img_size, size=(2,))
            canvas[
                max(0, ly - 1) : min(self.img_size, ly + 2),
                max(0, lx - 1) : min(self.img_size, lx + 2),
            ] = np.random.uniform(0.2, 0.8, size=(3,))

        # 4. Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, canvas.shape)
        canvas = np.clip(canvas + noise, 0.0, 1.0)

        # Convert HWC (32, 32, 3) -> CHW PyTorch FloatTensor (3, 32, 32)
        tensor_img = torch.from_numpy(canvas).permute(2, 0, 1).type(torch.float32)
        return tensor_img, label
