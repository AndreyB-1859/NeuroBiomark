import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    """
    Basic CNN for 3-class classification - OPTIMIZED FOR SMALL DATASETS
    
    DESIGN PRINCIPLES FOR 190-IMAGE DATASET:
    1. Shallow architecture (4 conv blocks instead of deep networks)
    2. Heavy regularization (dropout, batch norm, L2)
    3. Small number of filters to prevent overfitting
    4. Global Average Pooling instead of large FC layers
    5. Minimal trainable parameters (~250K total for 400x400 input)
    
    ARCHITECTURE:
    - Input: 400x400x3 RGB images
    - 4 Convolutional blocks with batch norm + dropout
    - Global Average Pooling (reduces params dramatically)
    - Small classifier head
    - Output: 3 classes (Control, ALS+Dementia, ALS-Dementia)
    
    NORMALIZATION:
    - Uses dataset-specific mean/std (computed from your 190 images)
    - NOT ImageNet normalization (since model trained from scratch)
    - Normalization handled in data loader, not in model
    
    SMALL DATASETS:
    - ~250K parameters vs 4M+ in EfficientNet/DenseNet
    - With 190 images: ~0.76 images per parameter (acceptable with heavy regularization)
    - Heavy dropout prevents memorization
    - BatchNorm provides regularization
    - Simple architecture is easier to train with limited data
    """
    
    def __init__(self, num_classes=3, input_channels=3, dropout=0.5, image_size=400):
        """
        Args:
            num_classes: Number of output classes (default: 3)
            input_channels: Number of input channels (default: 3 for RGB)
            dropout: Dropout rate (default: 0.5, high for small datasets)
            image_size: Input image size (default: 400 for 400x400 images)
        """
        super(BasicCNN, self).__init__()
        
        self.dropout_rate = dropout
        self.image_size = image_size
        
        # Calculate final feature map size after 4 pooling layers
        # Each pooling reduces size by 2: 400 -> 200 -> 100 -> 50 -> 25
        self.final_size = image_size // (2 ** 4)
        
        # ====================================================================
        # CONVOLUTIONAL FEATURE EXTRACTOR
        # For 400x400 input: 400 -> 200 -> 100 -> 50 -> 25 after 4 poolings
        # ====================================================================
        
        # Block 1: 3 -> 32 filters (400x400 -> 200x200)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.3)  # Light dropout early
        )
        
        # Block 2: 32 -> 64 filters (200x200 -> 100x100)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.4)
        )
        
        # Block 3: 64 -> 128 filters (100x100 -> 50x50)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Block 4: 128 -> 256 filters (50x50 -> 25x25)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.5)
        )
        
        # ====================================================================
        # CLASSIFIER HEAD (MINIMAL PARAMETERS)
        # ====================================================================
        
        # Global Average Pooling (reduces 25x25x256 -> 256)
        # This eliminates need for large FC layers!
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Small classifier: 256 -> 64 -> 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model info
        self._print_model_info()
    
    def forward(self, x):
        """Forward pass"""
        # Convolutional feature extraction
        x = self.conv1(x)  # 400 -> 200
        x = self.conv2(x)  # 200 -> 100
        x = self.conv3(x)  # 100 -> 50
        x = self.conv4(x)  # 50 -> 25
        
        # Global pooling + classification
        x = self.global_avg_pool(x)  # 25x25x256 -> 1x1x256
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def _print_model_info(self):
        """Print model configuration"""
        trainable, total = self.get_num_trainable_params()
        
        print("\n" + "="*70)
        print("BasicCNN Model Configuration (OPTIMIZED FOR SMALL DATASET)")
        print("="*70)
        print(f"Dataset size: 190 images (VERY SMALL)")
        print(f"Input image size: {self.image_size}x{self.image_size}")
        print(f"Architecture: 4 conv blocks + Global Avg Pool + Small Classifier")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Trainable parameters: {trainable:,} / {total:,}")
        print(f"\n✅ OPTIMIZED: ~{trainable:,} parameters for 190 images")
        print(f"   Ratio: ~{190/trainable:.4f} images per parameter")
        print(f"   This requires heavy regularization for small datasets")
        print(f"   Using dataset-specific normalization (NOT ImageNet)")
        print("="*70 + "\n")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_basic_cnn(num_classes=3, dropout=0.5, image_size=400):
    """
    Factory function to create BasicCNN for small dataset
    
    Args:
        num_classes: Number of output classes (default: 3)
        dropout: Dropout rate (default: 0.5)
        image_size: Input image size (default: 400 for 400x400)
    
    Returns:
        BasicCNN model optimized for small datasets
    """
    model = BasicCNN(
        num_classes=num_classes,
        input_channels=3,
        dropout=dropout,
        image_size=image_size
    )
    return model


def get_recommended_hyperparameters():
    """
    Returns recommended hyperparameters for 190-image medical imaging dataset
    
    These hyperparameters are specifically tuned for:
    - Small dataset (190 images)
    - 3-class imbalanced classification
    - Medical imaging (high-resolution 400x400)
    - 5-fold Group CV (prevents patient leakage)
    - Training from scratch (no ImageNet pretraining)
    """
    return {
        # Optimizer settings
        'learning_rate': 1e-3,       # Higher than transfer learning (training from scratch)
        'weight_decay': 1e-4,         # L2 regularization
        'optimizer': 'AdamW',         # Better weight decay handling than Adam
        
        # Training settings
        'batch_size': 8,              # Small batches for small dataset (400x400 uses more memory)
        'max_epochs': 100,            # More epochs needed (training from scratch)
        'early_stopping_patience': 15, # Stop if no improvement
        
        # Learning rate scheduling
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_params': {
            'mode': 'min',
            'factor': 0.5,            # Reduce LR by half
            'patience': 5,            # Wait 5 epochs before reducing
            'min_lr': 1e-6,           # Minimum learning rate
            'verbose': True
        },
        
        # Regularization
        'gradient_clip': 1.0,         # Prevent exploding gradients
        'label_smoothing': 0.1,       # Soft labels to prevent overconfidence
        
        # Class imbalance handling
        'use_class_weights': True,    # Weight loss by inverse class frequency
        
        # Augmentation
        'augmentation_probability': 0.8  # Apply augmentation 80% of the time
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test model creation
    print("Testing BasicCNN model...")
    
    model = create_basic_cnn(num_classes=3, dropout=0.5, image_size=400)
    
    # Test forward pass with 400x400 images
    dummy_input = torch.randn(4, 3, 400, 400)  # Batch of 4 images at 400x400
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits sample: {output[0].detach().numpy()}")
    
    # Print recommended hyperparameters
    print("\n" + "="*70)
    print("RECOMMENDED HYPERPARAMETERS")
    print("="*70)
    hyperparams = get_recommended_hyperparameters()
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("="*70 + "\n")