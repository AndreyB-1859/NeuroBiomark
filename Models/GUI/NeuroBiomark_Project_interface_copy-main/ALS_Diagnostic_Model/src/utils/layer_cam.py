from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

def extract_output_cam_and_image(model, image_tensor, class_idx, target_layers):
    """
    Extract model output, LayerCAM maps fused from all target_layers,
    and denormalized PIL image for visualization.
    
    Args:
        model: Your AttentionDenseNet121 model instance
        image_tensor: Input tensor to pass to model (1,C,H,W)
        class_idx: Target class index for CAM explanation
        target_layers: list of layers from model.model.features.denseblockX to analyze
    
    Returns:
        output: model raw output (logits)
        fused_cam: averaged CAM map across target layers (numpy array HxW)
        original_image: denormalized original PIL image
    """
    model.eval()
    cam_extractor = LayerCAM(model.model, target_layers)
    
     # Forward pass
    output = model(image_tensor)

    # Define target for CAM: class index
    targets = [ClassifierOutputTarget(class_idx)]

    # Get CAMs for each layer: tensor of shape (num_layers, H, W)
    layer_cams = cam_extractor(input_tensor=image_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)
    
    # Suppose layer_cams is list of numpy arrays from LayerCAM output
    layer_cams_tensors = [torch.tensor(cam) if isinstance(cam, np.ndarray) else cam for cam in layer_cams]
    # Fuse CAMs by averaging across layers
    fused_cam = torch.mean(torch.stack(layer_cams_tensors), dim=0)
    
    # Denormalize image tensor to PIL image for overlay
    denorm_img = image_tensor.squeeze(0).detach().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_img = denorm_img * std + mean
    denorm_img = denorm_img.clamp(0, 1)
    original_image = to_pil_image(denorm_img)
    
    return output, fused_cam.cpu().numpy(), original_image

def generate_layercam_overlay(cam, original_image):
    """
    Overlay LayerCAM map onto the original image.
    """
    # Convert original_image PIL to numpy float32 [0,1]
    original_np = np.array(original_image).astype(np.float32) / 255.0

    # Ensure cam is numpy array with values in [0,1]
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()
    cam = np.clip(cam, 0, 1)

    overlay = show_cam_on_image(original_np, cam, use_rgb=True)
    return Image.fromarray(overlay)

def generate_layercam_circles(cam, original_image, threshold=0.8):
    """
    Draw circles around highly activated LayerCAM regions.
    """
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized) / 255.0
    binary_mask = (cam_array >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    draw = ImageDraw.Draw(original_image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        r = max(w, h) // 2 + 4
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="red", width=3)
        circles.append((cx, cy, r))
    return original_image, circles

def generate_layercam_focus_mask(cam, original_image, threshold=0.8, alpha=0.6):
    """
    Mask/fade less important areas in the LayerCAM map.
    """
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized).astype(np.float32) / 255.0
    alpha_mask = np.where(cam_array < threshold, int(255 * alpha), 0).astype(np.uint8)
    image_rgba = original_image.convert("RGBA")
    black_overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    black_overlay.putalpha(Image.fromarray(alpha_mask))
    result = Image.alpha_composite(image_rgba, black_overlay)
    return result

def generate_layercam_boundary(cam, original_image, threshold=0.8, thickness=2, smooth=True):
    """
    Draw boundary lines for highly-activated LayerCAM regions.
    """
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()
    cam_resized = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.asarray(cam_resized).astype(np.float32) / 255.0
    if smooth:
        cam_array = cv2.GaussianBlur(cam_array, (3, 3), 0)
    mask = (cam_array >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bgr = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_bgr, contours, contourIdx=-1, color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
