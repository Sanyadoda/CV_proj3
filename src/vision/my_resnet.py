import torch
import torch.nn as nn
from torchvision.models import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        model=resnet18(pretrained= True)

        # Retrieve all layers except the final FC layer from the pretrained model
        self.conv_layers = nn.Sequential(*list(model.children())[:-1])  # Exclude the final FC layer

        #Freeze all parameters in the convolutional layers
        for param in self.conv_layers.parameters():
            param.requires_grad = False  # Freeze convolution layers

        # Get the number of input features to the FC layer
        num_features = model.fc.in_features  # Typically 512 for ResNet18

        # Replace the final FC layer with a new one that outputs 15 classes
        self.fc_layers = nn.Linear(num_features, 15)

        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")


        """ raise NotImplementedError(
            "`__init__` function in "
            + "`my_resnet.py` needs to be implemented"
        )
        """
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
        

        # Pass through the convolutional layers (no gradient updates here)
        x_features = self.conv_layers(x)  
        x_flatten = torch.flatten(x_features, 1)  # Flatten the output from Conv layers (batch size, num_features)
        
        # Pass through the new FC layer to get the final classification
        model_output = self.fc_layers(x_flatten)  # Output layer with 15 classes

        """ raise NotImplementedError(
            "`forward` function in "
            + "`my_resnet.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
