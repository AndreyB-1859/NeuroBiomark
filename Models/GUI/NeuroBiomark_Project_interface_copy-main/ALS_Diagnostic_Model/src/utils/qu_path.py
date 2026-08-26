import cv2
import numpy as np
from PIL import Image


def extract_color_masks(qpath_image):
    """Extract blue, yellow, and orange masks from a QuPath-annotated image."""

    # Convert to RGB numpy array
    image_np = np.array(qpath_image.convert("RGB"))

    # Define BGR-like color bounds (tune these as needed)
    masks = {
        "blue": cv2.inRange(image_np, (90, 90, 200), (150, 150, 255)),
        "yellow": cv2.inRange(image_np, (180, 180, 0), (255, 255, 150)),
        #"orange": cv2.inRange(image_np, (200, 100, 0), (255, 160, 80))
    }

    # Convert to PIL images for visualization
    return {k: Image.fromarray((v > 0).astype(np.uint8) * 255) for k, v in masks.items()}


def quantify_focus(circles, cells_mask, border_exclude=10):
    # Convert PIL image to binary NumPy mask
    cell_mask_np = np.array(cells_mask.convert("L")) > 0  # shape: (H, W)

    # Exclude borders
    h, w = cell_mask_np.shape
    cell_mask_np[:border_exclude, :] = 0              # Top
    cell_mask_np[-border_exclude:, :] = 0             # Bottom
    cell_mask_np[:, :border_exclude] = 0              # Left
    cell_mask_np[:, -border_exclude:] = 0             # Right

    hit_count, miss_count = 0, 0
    for cx, cy, r in circles:
        circle_mask = np.zeros_like(cell_mask_np, dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 1, -1)

        if np.any(circle_mask & cell_mask_np):
            hit_count += 1
        else:
            miss_count += 1

    precision = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0.0
    return hit_count, miss_count, precision
