import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional


class HybridDRClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        lesion_feature_dim: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 512,
        in_channels: int = 3,
    ):
        
        super().__init__()
        self.num_classes = num_classes
        self.lesion_feature_dim = lesion_feature_dim
        self.in_channels = in_channels

        if pretrained:
            try:
                weights = models.EfficientNet_B4_Weights.DEFAULT
                self.backbone = models.efficientnet_b4(weights=weights)
            except AttributeError:
                self.backbone = models.efficientnet_b4(pretrained=True)
        else:
            self.backbone = models.efficientnet_b4(pretrained=False)

            
        if self.in_channels != 3:
            old_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if self.in_channels > 3:
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
            self.backbone.features[0][0] = new_conv

        self.num_features = 1792

        self.backbone.classifier = nn.Identity()
        in_features = self.num_features + lesion_feature_dim
        self.classifier_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, lesion_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        features = self.backbone(x)

        if lesion_counts is not None and self.lesion_feature_dim > 0:
            merged = torch.cat([features, lesion_counts], dim=1)
        else:
            merged = features

        logits = self.classifier_head(merged)
        return logits


def build_classifier(config: Dict[str, Any], in_channels: int = 3) -> HybridDRClassifier:
    
    model_cfg = config.get("classification_model", {})
    
    return HybridDRClassifier(
        num_classes=model_cfg.get("num_classes", 5),
        lesion_feature_dim=model_cfg.get("lesion_feature_dim", 4),
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        in_channels=in_channels,
    )

