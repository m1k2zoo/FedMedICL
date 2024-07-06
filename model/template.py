from torch import nn


class FoundationModel(nn.Module):
    """
    Template for models with three parts: input processing part, backbone layers, and output layer.

    Attributes:
        input_layer (nn.Module): The input processing part of the model.
        backbone_layers (nn.Module): The backbone layers (feature extractor) of the model.
        output_layer (nn.Module): The output layer (head) of the model.
    """

    def __init__(self, input_layer, backbone_layers, output_layer):
        """
        Args:
            input_layer (nn.Module): The input processing part of the model.
            backbone_layers (nn.Module): The backbone layers (feature extractor) of the model.
            output_layer (nn.Module): The output layer (head) of the model.
        """

        super().__init__()
        self.input_layer = input_layer
        self.backbone_layers = backbone_layers
        self.output_layer = output_layer

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.input_layer(x)
        x = self.backbone_layers(x)
        x = self.output_layer(x)
        return x

    def replace_backbone_layers(self, backbone_layers):
        """
        Replaces the backbone layers of the model with the provided backbone_layers.

        Args:
            backbone_layers (OrderedDict): State dict of the new backbone layers.

        Returns:
            FoundationModel: The modified model instance.
        """
        # Load the state_dict of the new backbone layers into the current model's backbone_layers
        self.backbone_layers.load_state_dict(backbone_layers)

        # Return the modified model instance
        return self

    def replace_all_layers(self, new_backbone_layer_dict, new_output_layer_dict):
        """
        Replaces all layers of the model with the provided layers.

        Args:
            new_backbone_layer_dict (dict): State dict of the new backbone layers.
            new_output_layer_dict (dict): State dict of the new output layer.

        Returns:
            FoundationModel: The modified model instance.
        """
        self.backbone_layers.load_state_dict(new_backbone_layer_dict)
        self.output_layer.load_state_dict(new_output_layer_dict)

        return self  # Return the modified instance
