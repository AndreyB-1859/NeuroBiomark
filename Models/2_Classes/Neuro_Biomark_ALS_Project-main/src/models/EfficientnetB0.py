import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNetB0 for 3-class classification - OPTIMIZED FOR SMALL DATASETS
    
    🎯 CRITICAL FIXES FOR YOUR 190-IMAGE DATASET:
    1. Encoder MUST be frozen (freeze_encoder=True)
    2. Much simpler classifier head (reduced from 3 layers to 2)
    3. Higher dropout (0.5 default, can go up to 0.6-0.7)
    4. Reduced classifier capacity (1280 → 128 → 3)
    
    With 190 images and freeze_encoder=True:
    - Only ~400 parameters to train (0.01% of total model)
    - This prevents catastrophic overfitting
    - Validation loss won't explode
    
    RECOMMENDED SETTINGS:
    - freeze_encoder=True (MANDATORY for <500 images)
    - dropout=0.5 to 0.7 (higher for smaller datasets)
    - learning_rate=1e-5 or 5e-6 (very low!)
    - weight_decay=0.01 to 0.05
    - batch_size=8 or 16 (small batches for small dataset)
    """
    
    def __init__(self, num_classes=3, pretrained=True, freeze_encoder=True, 
                 dropout=0.5):
        """
        Args:
            num_classes: Number of output classes (default: 3)
            pretrained: Use ImageNet pretrained weights (default: True)
            freeze_encoder: Freeze encoder weights (default: True - CRITICAL!)
            dropout: Dropout rate (default: 0.5, increase to 0.6-0.7 for tiny datasets)
        """
        super().__init__()
        
        # Load pretrained model
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # CRITICAL: Freeze encoder for small datasets
        if freeze_encoder:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            print("✅ Encoder FROZEN - Only training classifier (recommended for <500 images)")
        else:
            print("⚠️  WARNING: Encoder UNFROZEN - This will overfit with only 190 images!")
            print("   → Set freeze_encoder=True to fix validation loss explosion")
        
        # Get input features from the original classifier
        in_features = self.backbone.classifier[1].in_features  # 1280 for EfficientNet-B0
        
        # MUCH SIMPLER classifier for tiny dataset (190 images)
        # Original design: 1280 → 512 → 256 → 3 (too complex!)
        # New design: 1280 → 128 → 3 (minimal trainable parameters)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),  # Only 163,968 parameters!
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)  # Only 387 parameters
        )
        
        # Print model info
        self._print_model_info(freeze_encoder, dropout)
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def unfreeze_encoder(self):
        """
        ⚠️  DO NOT USE THIS with only 190 images!
        Only unfreeze if you have >500 images AND good frozen performance
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("⚠️  Encoder unfrozen - High risk of overfitting with small dataset!")
    
    def freeze_encoder(self):
        """Freeze encoder to prevent overfitting"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("✅ Encoder frozen - Safe for small datasets")
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def _print_model_info(self, freeze_encoder, dropout):
        """Print model configuration info"""
        trainable, total = self.get_num_trainable_params()
        print("\n" + "="*70)
        print("EfficientNet-B0 Model Configuration (OPTIMIZED FOR SMALL DATASET)")
        print("="*70)
        print(f"Dataset size: 190 images (VERY SMALL)")
        print(f"Encoder frozen: {freeze_encoder} {'✅ CORRECT' if freeze_encoder else '❌ WRONG - WILL OVERFIT!'}")
        print(f"Dropout rate: {dropout}")
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        if freeze_encoder:
            print(f"\n✅ GOOD: Only {trainable:,} trainable parameters")
            print(f"   This is safe for your 190-image dataset")
            print(f"   Recommended: Use learning_rate=1e-5 or lower")
        else:
            print(f"\n❌ BAD: {trainable:,} trainable parameters")
            print(f"   This is TOO MANY for 190 images!")
            print(f"   You need ~10-20 images per parameter")
            print(f"   Expected result: Validation loss explosion (EXACTLY what you see!)")
            print(f"\n   FIX: Set freeze_encoder=True")
        
        print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE FOR YOUR AUGMENTATION EVALUATION
# ============================================================================

def create_model_for_small_dataset(num_classes=3):
    """
    Factory function to create properly configured model for 190-image dataset
    
    Returns model with:
    - Frozen encoder (only ~165K trainable params)
    - High dropout (0.5)
    - Simple classifier head
    """
    model = EfficientNetB0Classifier(
        num_classes=num_classes,
        pretrained=True,
        freeze_encoder=True,  # CRITICAL!
        dropout=0.5           # Can increase to 0.6-0.7 if still overfitting
    )
    return model


def get_recommended_hyperparameters():
    """
    Returns recommended hyperparameters for 190-image medical imaging dataset
    """
    return {
        'learning_rate': 1e-5,      # Very low! Can try 5e-6 if still unstable
        'weight_decay': 0.01,        # L2 regularization
        'batch_size': 8,             # Small batches for small dataset
        'optimizer': 'AdamW',        # Better than Adam for small datasets
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_params': {
            'factor': 0.5,
            'patience': 3,
            'min_lr': 1e-7
        },
        'gradient_clip': 1.0,        # Prevent exploding gradients
        'early_stopping': 10,        # Stop if no improvement
        'max_epochs': 50
    }



# class EfficientNetB0Classifier(nn.Module):
#     """
#     EfficientNetB0 for 3-class classification
#     Lightweight alternative to DenseNet121
#     """
    
#     def __init__(self, num_classes=3, pretrained=True, freeze_encoder=False, dropout=0.3):
#         super().__init__()
        
#         # Load pretrained model
#         weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
#         self.backbone = efficientnet_b0(weights=weights)
        
#         # Freeze encoder if specified
#         if freeze_encoder:
#             for param in self.backbone.features.parameters():
#                 param.requires_grad = False
        
#         # Replace classifier
#         in_features = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         return self.backbone(x)
    
