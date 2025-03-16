import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.activation = None

        ############################################################################
        # Student code begin
        ############################################################################

        # Load pretrained ResNet18 model
        resnet = resnet18(pretrained=True)

        # Freeze all convolutional and first two fully connected layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Modify the last fully connected layer for 7 output labels
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 7)

        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])  # All layers except final FC
        self.fc_layers = resnet.fc  # Last FC layer (trainable)
        self.activation = nn.Sigmoid()  # Sigmoid activation for multi-label classification

        self.loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')  # Binary cross-entropy loss

        """ raise NotImplementedError(
            "`__init__` function in "
            + "`multi_resnet.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################

        # Extract features using frozen convolutional layers
        features = self.conv_layers(x)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten

        # Pass through the FC layer 
        logits = self.fc_layers(features)  # Get raw scores

        # Apply sigmoid activation for multi-label classification
        model_output = self.activation(logits)  # Apply sigmoid for multi-label output
        
        """ raise NotImplementedError(
            "`forward` function in "
            + "`multi_resnet.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
