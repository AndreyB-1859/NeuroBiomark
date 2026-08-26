from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw


def extract_output_cam_and_image(model, image_tensor, class_idx, target_layer="features.denseblock4"):

    cam_extractor = GradCAM(model.model, target_layer=target_layer)
    output = model(image_tensor)
    cam = cam_extractor(class_idx, output)[0]

    # De-normalize image
    denorm_img = image_tensor.squeeze().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_img = denorm_img * std + mean
    original_image = to_pil_image(denorm_img.clamp(0, 1))

    return output, cam, original_image

def generate_gradcam_overlay(cam, original_image):

    overlay = overlay_mask(original_image, to_pil_image(cam, mode="F"))
    return overlay

def generate_gradcam_circles(cam, original_image, threshold=0.8):

    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()

    # Resize and normalize CAM
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized) / 255.0
    binary_mask = (cam_array >= threshold).astype(np.uint8)

    # Detect contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles
    circles = []
    draw = ImageDraw.Draw(original_image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        r = max(w, h) // 2 + 4
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="red", width=3)
        circles.append((cx,cy,r))

    return original_image, circles


def generate_gradcam_focus_mask(cam, original_image, threshold=0.8, alpha=0.6):

    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()

    # Resize and normalize CAM
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized).astype(np.float32) / 255.0

    # Create alpha mask: 0 for high activation, alpha for low
    alpha_mask = np.where(cam_array < threshold, int(255 * alpha), 0).astype(np.uint8)

    # Convert to RGBA
    image_rgba = original_image.convert("RGBA")

    # Create black image with variable alpha (based on CAM)
    black_overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    black_overlay.putalpha(Image.fromarray(alpha_mask))

    # Composite: overlay the transparent black mask on top of original image
    result = Image.alpha_composite(image_rgba, black_overlay)

    return result


import torch
import numpy as np
import cv2
from PIL import Image

def generate_gradcam_boundary(cam, original_image, threshold=0.8, thickness=2, smooth=True):
    """
    Draw a red boundary along the threshold contour separating focused (>= threshold)
    and non-focused (< threshold) regions of the CAM, leaving the image otherwise unchanged.

    Args:
        cam: torch.Tensor [1,H,W] or numpy array [H,W] with values in [0,1] (or any float range).
        original_image: PIL.Image (RGB or RGBA); the underlying image remains fully visible.
        threshold: float in [0,1] at which the boundary is drawn.
        thickness: contour line thickness in pixels.
        smooth: if True, lightly blur CAM before thresholding to reduce jagged edges.

    Returns:
        PIL.Image with red boundary overlayed.
    """
    # Ensure numpy CAM in [0,1] and resize to image size
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()

    # Resize CAM to match the original image
    cam_resized = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(
        original_image.size, Image.BILINEAR
    )
    cam_array = np.asarray(cam_resized).astype(np.float32) / 255.0

    if smooth:
        # Light blur to make cleaner contours (tweak kernel if needed)
        cam_array = cv2.GaussianBlur(cam_array, (3, 3), 0)

    # Binary mask at threshold
    mask = (cam_array >= threshold).astype(np.uint8) * 255

    # Option A (nice smooth lines): find contours and draw them
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert image to OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Draw red contours (BGR: (0,0,255))
    cv2.drawContours(img_bgr, contours, contourIdx=-1, color=(0, 0, 255),
                     thickness=thickness, lineType=cv2.LINE_AA)

    # Back to PIL RGB
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
