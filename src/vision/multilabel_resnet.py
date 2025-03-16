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

        #Load the pretrained ResNet18 model
        model = resnet18(pretrained=True)

        # Retrieve all layers except the final FC layer
        self.conv_layers = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer

        # Freeze all convolutional layers
        #for param in self.conv_layers.parameters():
            #param.requires_grad = False  

        # Get the number of input features for the new FC layer
        num_features = model.fc.in_features  # Typically 512 for ResNet18

        # Replace the final fully connected layer with one that outputs 7 classes
        self.fc_layers = nn.Linear(num_features, 7)

        # Loss function for multi-label classification
        self.loss_criterion = nn.BCEWithLogitsLoss(reduction="mean")  # Binary Cross-Entropy loss


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
        x_features = self.conv_layers(x)
        x_flatten = torch.flatten(x_features, 1)  # Flatten (batch_size, num_features)

        # Pass through the FC layer
        model_output = self.fc_layers(x_flatten)  

        # Apply sigmoid activation for multi-label classification
        model_output = torch.sigmoid(model_output)
        
        """ raise NotImplementedError(
            "`forward` function in "
            + "`multi_resnet.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
