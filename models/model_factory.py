"""
Model factory: creates any model by name string.
Makes it easy to swap models in experiments via config.
"""

from .efficientnet import EfficientNetB4
from .resnet import ResNet50
from .densenet import DenseNet169


_MODEL_REGISTRY = {
    'efficientnet_b4': EfficientNetB4,
    'resnet50': ResNet50,
    'densenet169': DenseNet169,
}


def create_model(name, num_classes=7, dropout_rate=0.3, pretrained=True):
    """
    Factory function to create a model by name.

    Parameters
    ----------
    name : str
        One of: 'efficientnet_b4', 'resnet50', 'densenet169'.
    num_classes : int
        Number of output classes.
    dropout_rate : float
        Dropout rate for the MC Dropout head.
    pretrained : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    BaseModel
        The initialized model.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    model_class = _MODEL_REGISTRY[name]
    return model_class(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )


def list_models():
    """Return list of available model names."""
    return list(_MODEL_REGISTRY.keys())
