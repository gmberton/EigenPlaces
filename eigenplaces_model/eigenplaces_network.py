
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple

from eigenplaces_model.layers import Flatten, L2Norm, GeM

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


class GeoLocalizationNet_(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int):
        """Return a model_ for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = _get_backbone(backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def _get_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    return getattr(torchvision.models, backbone_name.lower())()


def _get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    backbone = _get_torchvision_model(backbone_name)

    logging.info("Loading pretrained backbone's weights from CosPlace")
    cosplace = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=backbone_name, fc_output_dim=512)
    new_sd = {k1: v2 for (k1, v1), (k2, v2) in zip(backbone.state_dict().items(), cosplace.state_dict().items())
              if v1.shape == v2.shape}
    backbone.load_state_dict(new_sd, strict=False)

    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

