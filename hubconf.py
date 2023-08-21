
dependencies = ['torch', 'torchvision']

import torch
from eigenplaces_model import eigenplaces_network


AVAILABLE_TRAINED_MODELS = {
    # backbone : list of available fc_output_dim, which is equivalent to descriptors dimensionality
    "VGG16":     [          512],
    "ResNet18":  [     256, 512],
    "ResNet50":  [128, 256, 512, 1024, 2048],
    "ResNet101": [128, 256, 512, 1024, 2048],
}


def get_trained_model(backbone : str = "ResNet50", fc_output_dim : int = 2048) -> torch.nn.Module:
    """Return a model trained with EigenPlaces on San Francisco eXtra Large.
    
    Args:
        backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
        fc_output_dim (int): the output dimension of the last fc layer, equivalent to
            the descriptors dimension. Must be between 32 and 2048, depending on model's availability.
    
    Return:
        model (torch.nn.Module): a trained model.
    """
    print(f"Returning EigenPlaces model with backbone: {backbone} with features dimension {fc_output_dim}")
    if backbone not in AVAILABLE_TRAINED_MODELS:
        raise ValueError(f"Parameter `backbone` is set to {backbone} but it must be one of {list(AVAILABLE_TRAINED_MODELS.keys())}")
    try:
        fc_output_dim = int(fc_output_dim)
    except:
        raise ValueError(f"Parameter `fc_output_dim` must be an integer, but it is set to {fc_output_dim}")
    if fc_output_dim not in AVAILABLE_TRAINED_MODELS[backbone]:
        raise ValueError(f"Parameter `fc_output_dim` is set to {fc_output_dim}, but for backbone {backbone} "
                         f"it must be one of {list(AVAILABLE_TRAINED_MODELS[backbone])}")
    model = eigenplaces_network.GeoLocalizationNet_(backbone, fc_output_dim)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            f'https://github.com/gmberton/EigenPlaces/releases/download/v1.0/{backbone}_{fc_output_dim}_eigenplaces.pth',
        map_location=torch.device('cpu'))
    )
    return model