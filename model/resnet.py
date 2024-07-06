from torch import nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

from .template import FoundationModel


class ResNet(FoundationModel):
    """
    ResNet class that inherits from FoundationModel,
        representing a ResNet model with three parts: input part, backbone, and head.
    """

    def __init__(self, architecture, num_classes):
        """
        Initialize the ResNet model.

        Args:
            architecture (str): The architecture of the ResNet model.
                Supported options: "resnet18", "resnet34", "resnet50".
            num_classes (int): The number of output classes.
        """
        # Load the pretrained ResNet model based on the specified architecture
        if architecture == "resnet18":
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif architecture == "resnet34":
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif architecture == "resnet50":
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported architecture: {}".format(architecture))

        # Part 1: Input processing part (the first layer of the network)
        input_layer = lambda x: x  # Only supports images for now

        # Part 2: Backbone (feature extractor part with AdaptiveAvgPool2d and Flatten)
        first = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        other_layers = nn.Sequential(
            *list(resnet.children())[4:-1], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        backbone_layers = nn.Sequential(first, other_layers)

        # Part 3: Head (linear head for predictions)
        head = nn.Linear(resnet.fc.in_features, num_classes)

        # Initialize the parent class (FoundationModel) with the three parts
        super().__init__(input_layer, backbone_layers, head)
