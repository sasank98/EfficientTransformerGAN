# Ajaysriram Muthuraman & Sasank Potluri, Date: 09/22/2023, CS 7180 Advanced Perception
import torch
import torch.nn as nn

from .hpb import HPB, Config
from .et import EfficientTransformer
from .debug import PrintLayer


class BackBoneBlock(nn.Module):
    def __init__(self, num, fm, **args):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm(**args))

    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x


class ESRT(nn.Module):
    def __init__(self, hiddenDim=32, mlpDim=128, scaleFactor=2):
        super().__init__()
        self.conv3 = nn.Conv2d(3, hiddenDim,
                               kernel_size=3, padding=1)
        
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)

        self.path1 = nn.Sequential(
            BackBoneBlock(3, HPB, inChannel=hiddenDim,
                          outChannel=hiddenDim, reScale=self.adaptiveWeight),
            BackBoneBlock(1, EfficientTransformer,
                          mlpDim=mlpDim, inChannels=hiddenDim),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      3, kernel_size=3, padding=1),
        )

        self.path2 = nn.Sequential(
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1 + x2
    

class ETGAN(nn.Module):
    def __init__(self, hiddenDim=32, mlpDim=128, scaleFactor=2):
        super().__init__()
        self.conv3 = nn.Conv2d(3, hiddenDim,
                               kernel_size=3, padding=1)
        
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        
        lamRes_main = torch.nn.Parameter(torch.ones(1))
        lamX_main = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight_main = (lamRes_main,lamX_main)


        self.path1 = nn.Sequential(
            BackBoneBlock(2, HPB, inChannel=hiddenDim,
                          outChannel=hiddenDim, reScale=self.adaptiveWeight),
            BackBoneBlock(1, EfficientTransformer,
                          mlpDim=mlpDim, inChannels=hiddenDim),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      3, kernel_size=3, padding=1),
        )

        self.path2 = nn.Sequential(
            # nn.PixelShuffle(scaleFactor),
            # nn.Conv2d(hiddenDim // (scaleFactor**2),
            #           3, kernel_size=3, padding=1),

            BackBoneBlock(1, HPB, inChannel=hiddenDim,
                          outChannel=hiddenDim, reScale=self.adaptiveWeight),
            nn.Conv2d(hiddenDim,hiddenDim*scaleFactor*scaleFactor,kernel_size=3, padding=1),
            nn.PixelShuffle(scaleFactor),
            nn.PReLU(),
            nn.Conv2d(hiddenDim,3,kernel_size=3,padding=1)
            

        )

    def forward(self, x):
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1*self.adaptiveWeight_main[0] + torch.clamp_(x2, 0.0, 1.0)*self.adaptiveWeight_main[1]


if __name__ == '__main__':
    x = torch.tensor([float(i+1)
                     for i in range(3*48*48)]).reshape((1, 3, 48, 48))

    model = ESRT(mlpDim=128, scaleFactor=4)
    y = model(x)
    print(y.shape)
