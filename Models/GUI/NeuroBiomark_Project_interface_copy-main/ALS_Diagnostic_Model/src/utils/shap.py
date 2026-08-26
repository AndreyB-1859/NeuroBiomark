# import shap
# import torch
# from torchvision.transforms.functional import to_pil_image
# from torchcam.utils import overlay_mask
#
# def generate_shap_heatmap(model, image_tensor, class_idx=None):
#     image_tensor.requires_grad_()
#
#     # Background: black image
#     background = torch.zeros_like(image_tensor)
#
#     explainer = shap.GradientExplainer(model, background)
#     shap_values = explainer.shap_values(image_tensor, nsamples=50)
#
#     # Get SHAP for target class
#     shap_map = shap_values[class_idx][0]  # shape (3, H, W)
#     shap_map = shap_map.mean(0).cpu().numpy()  # (H, W)
#
#     # Normalize to [0, 1]
#     shap_map -= shap_map.min()
#     shap_map /= shap_map.max() + 1e-6
#
#     # De-normalize original image
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#     img = image_tensor[0].detach().cpu() * std + mean
#     original_img = to_pil_image(img.clamp(0,1))
#
#     # Overlay SHAP heatmap
#     shap_img = overlay_mask(original_img, to_pil_image(shap_map, mode='F'))
#
#     return shap_img
