import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from ALS_Diagnostic_Model.src.model.attention import SEBlock

class AttentionDenseNet121(nn.Module):
    def __init__(self, num_classes=3, attention=SEBlock, reduction=16, pretrained=True, freeze_until=None):
        super().__init__()

        # Decide whether to use pretrained ImageNet weights
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = densenet121(weights=weights)
        self.attention = attention  # store attention type for internal use

        # Inject attention blocks after each dense block
        self.model.features.denseblock1 = nn.Sequential(
            self.model.features.denseblock1, attention(256, reduction)
        )
        self.model.features.denseblock2 = nn.Sequential(
            self.model.features.denseblock2, attention(512, reduction)
        )
        self.model.features.denseblock3 = nn.Sequential(
            self.model.features.denseblock3, attention(1024, reduction)
        )
        self.model.features.denseblock4 = nn.Sequential(
            self.model.features.denseblock4, attention(1024, reduction)
        )

        # Replace classifier (no dropout)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        # Optionally freeze early layers (except attention modules)
        if freeze_until is not None:
            self._freeze_model_features(up_to=freeze_until)

    def _freeze_model_features(self, up_to="denseblock2"):
        freeze = True
        for name, module in self.model.features.named_children():
            if freeze:
                for child in module.children() if isinstance(module, nn.Sequential) else [module]:
                    if not isinstance(child, self.attention):
                        for param in child.parameters():
                            param.requires_grad = False
            if name == up_to:
                freeze = False

    def forward(self, x):
        return self.model(x)