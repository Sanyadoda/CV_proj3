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

        # Load pretrained model without the last FC layer
        resnet = resnet18(pretrained=False)
        pretrained_model_path = "./CV_proj3/trained_MyResNet18_final.pt"
        checkpoint = torch.load(pretrained_model_path, map_location="cpu")

        if checkpoint["class_name"] != "MyResNet18":
            raise ValueError(f"Expected MyResNet18 model weights, found {checkpoint['class_name']}")

        state_dict = checkpoint["state_dict"]
        state_dict.pop("fc_layers.weight", None)  # Remove final FC layer
        state_dict.pop("fc_layers.bias", None)

        # Load pretrained weights (excluding the last FC layer)
        resnet.load_state_dict(state_dict, strict=False)

        # Freeze convolutional layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Extract convolutional layers
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-2])  # Keep all but last FC

        #Ensure feature size is (batch, 512) using adaptive pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  

        # Ensure input to FC is correct (512 features)
        num_features = 512  
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

        # Freeze first two FC layers
        for param in list(self.fc_layers.children())[:4]:  
            param.requires_grad = False

        self.activation = nn.Sigmoid()
        
        self.loss_criterion = nn.BCEWithLogitsLoss(reduction="mean")


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
        # Extract convolutional features
        features = self.conv_layers(x)

        # Ensure the output is (batch_size, 512)
        features = self.avg_pool(features)
        features = torch.flatten(features, start_dim=1)  # Flatten properly

        # Pass through FC layers
        x = self.fc_layers(features)

        # Apply sigmoid activation for multi-label classification
        model_output= self.activation(x)
        
        """ raise NotImplementedError(
            "`forward` function in "
            + "`my_resnet.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
