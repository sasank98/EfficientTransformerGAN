import torch.nn as nn

# Function for creating a default convolutional layer
def defaultConv(inChannels, outChannels, kernelSize, bias=True):
    return nn.Conv2d(
        inChannels, outChannels, kernelSize,
        padding=(kernelSize//2), bias=bias)


# Class definition for Squeeze-and-Excitation (SE) Layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        # Fully connected layers for SE Block
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), # Reduce dimensionality
            nn.ReLU(inplace=True), # ReLU activation
            nn.Linear(channel // reduction, channel, bias=False), # Restore dimensionality
            nn.Sigmoid() # Sigmoid activation for scale factors
        )

    def forward(self, x):

        # Get dimensions of the input tensor
        b, c, _, _ = x.size()

        # Apply adaptive average pooling to reduce spatial dimensions
        y = self.avg_pool(x).view(b, c)

         # Apply fully connected layers to compute scale factors
        y = self.fc(y).view(b, c, 1, 1)

         # Scale the input tensor element-wise by the computed factors
        return x * y.expand_as(x)
