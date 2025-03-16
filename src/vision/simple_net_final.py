import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None
        

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2),  # Conv1
            nn.BatchNorm2d(10),  # BatchNorm after Conv1
            nn.ReLU(), # Activation
            nn.MaxPool2d(kernel_size=3, stride=3),  # Pool1

            nn.Dropout(p=0.5),  # Dropout layer before the last convoltion layer Conv2

            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding=0),  # Conv2
            nn.BatchNorm2d(10),  # BatchNorm after Conv2
            nn.ReLU(),  # Activation
            #nn.MaxPool2d(kernel_size=3, stride=3),  # Pool2      
            
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=1, padding=0),  # Conv3
            nn.ReLU(),  # Activation
            nn.MaxPool2d(kernel_size=3, stride=2)  # Pool3 
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(15 * 5 * 5, 100),  # Fully connected 1
            nn.ReLU(),
            nn.Linear(100, 15)  # Fully connected 
        )

        for layer in self.conv_layers:
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        """ raise NotImplementedError(
            "`__init__` function in "
            + "`simple_net.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        x = self.conv_layers(x)  # Pass through conv layers
        x = x.view(x.shape[0], -1)  # Flatten
        model_output = self.fc_layers(x)  # Pass through FC layers

        """ raise NotImplementedError(
            "`forward` function in "
            + "`simple_net.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################

2        return model_output