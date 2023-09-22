import torch
import torch.nn as nn

from .comm import defaultConv

# Import defaultConv and other necessary components from the .comm module
class ResidualUnit(nn.Module):
    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True):
        super().__init__()

        # Define the reduction and expansion convolution layers
        self.reduction = defaultConv(
            inChannel, outChannel//2, kernelSize, bias)
        self.expansion = defaultConv(
            outChannel//2, inChannel, kernelSize, bias)
        
        # Store the scaling factors for residual connections
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        # Apply the reduction convolution
        res = self.reduction(x)

        # Scale and apply the expansion convolution
        res = self.lamRes * self.expansion(res)

        # Add the scaled residual to the input
        x = self.lamX * x + res

        return x


class ARFB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()

        # Create two ResidualUnit instances for the ARFB module
        self.RU1 = ResidualUnit(inChannel, outChannel, reScale)
        self.RU2 = ResidualUnit(inChannel, outChannel, reScale)

        # Define convolution layers
        self.conv1 = defaultConv(2*inChannel, 2*outChannel, kernelSize=1)
        self.conv3 = defaultConv(2*inChannel, outChannel, kernelSize=3)

        # Store scaling factors for residual connections
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        # Apply Residual Units

        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)

        # Concatenate the outputs of the Residual Units
        x_ru = torch.cat((x_ru1, x_ru2), 1)

        # Apply convolution layers
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)


        # Scale the output and add to the input
        x_ru = self.lamRes * x_ru
        x = x*self.lamX + x_ru
        return x
