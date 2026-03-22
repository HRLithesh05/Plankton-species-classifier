"""
CNN model using EfficientNetV2 for plankton classification.
Optimized for RTX 4060 Ti (8.59 GB VRAM).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


class PlanktonCNN(nn.Module):
    """
    EfficientNetV2-based CNN for plankton classification.

    Uses transfer learning from ImageNet pretrained weights.
    Two-phase training:
        1. Frozen backbone - only train classifier head
        2. Fine-tuning - unfreeze last N layers
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = 'efficientnet_v2_s',
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained backbone
        self.backbone = self._create_backbone(model_name, pretrained)

        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(512, num_classes)
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

    def _create_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """Create the backbone model."""
        weights = 'DEFAULT' if pretrained else None

        if model_name == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(weights=weights)
            # Remove the classifier
            model.classifier = nn.Identity()
        elif model_name == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(weights=weights)
            model.classifier = nn.Identity()
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Identity()
        elif model_name == 'resnet50':
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
        elif model_name == 'convnext_tiny':
            model = models.convnext_tiny(weights=weights)
            model.classifier = nn.Sequential(
                model.classifier[0],
                model.classifier[1],
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    def _get_feature_dim(self) -> int:
        """Get the output feature dimension of the backbone."""
        # Use a dummy input to get the feature dimension
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            dim = features.shape[1]
        self.backbone.train()
        return dim

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone frozen. Trainable params: {self.count_trainable_params():,}")

    def unfreeze_backbone(self, num_layers: int = -1):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end.
                       -1 means unfreeze all.
        """
        # First, make sure all params require grad
        for param in self.backbone.parameters():
            param.requires_grad = True

        if num_layers == -1:
            print(f"Full backbone unfrozen. Trainable params: {self.count_trainable_params():,}")
            return

        # Get all modules
        modules = list(self.backbone.modules())

        # Freeze all except last num_layers
        modules_to_freeze = modules[:-num_layers] if num_layers > 0 else []

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        print(f"Unfroze last {num_layers} layers. Trainable params: {self.count_trainable_params():,}")

    def count_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get features from backbone
        features = self.backbone(x)

        # Flatten if needed (for some backbones)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings (before classification)."""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return features


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)

        # Compute softmax probabilities
        probs = torch.softmax(inputs, dim=-1)

        # Get probability of true class
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Compute focal weight
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_model(
    num_classes: int,
    config_dict: dict,
    device: torch.device = None
) -> Tuple[PlanktonCNN, nn.Module]:
    """
    Create CNN model and loss function.

    Returns:
        model, criterion
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PlanktonCNN(
        num_classes=num_classes,
        model_name=config_dict['model_name'],
        pretrained=config_dict['pretrained'],
        dropout=config_dict['dropout'],
        freeze_backbone=config_dict['freeze_backbone']
    )

    model = model.to(device)

    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config_dict.get('label_smoothing', 0.0)
    )

    print(f"\nModel: {config_dict['model_name']}")
    print(f"  Total parameters: {model.count_total_params():,}")
    print(f"  Trainable parameters: {model.count_trainable_params():,}")
    print(f"  Device: {device}")

    return model, criterion


def create_optimizer(
    model: nn.Module,
    config_dict: dict,
    phase: str = 'frozen'
) -> torch.optim.Optimizer:
    """
    Create optimizer for the model.

    Args:
        model: The model
        config_dict: Configuration dictionary
        phase: 'frozen' or 'finetune'
    """
    if phase == 'frozen':
        lr = config_dict['learning_rate_frozen']
    else:
        lr = config_dict['learning_rate_finetune']

    # Only optimize parameters that require gradients
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config_dict['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=config_dict['weight_decay']
        )
    elif config_dict['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=config_dict['weight_decay']
        )
    elif config_dict['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=config_dict['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config_dict['optimizer']}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config_dict: dict,
    num_epochs: int,
    steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = config_dict['warmup_epochs'] * steps_per_epoch

    if config_dict['scheduler'] == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config_dict['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )

    elif config_dict['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

    else:
        # No scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)

    return scheduler


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    import config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, criterion = create_model(
        num_classes=94,
        config_dict=config.CNN_CONFIG,
        device=device
    )

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
