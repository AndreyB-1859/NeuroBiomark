import torch
from ALS_Diagnostic_Model.src.model.DenseNet121 import AttentionDenseNet121

def load_model(weights_path):
    model = AttentionDenseNet121(freeze_until="denseblock4")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
