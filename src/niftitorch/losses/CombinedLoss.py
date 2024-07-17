import torch.nn as nn
from .PerceptualLoss import PerceptualLoss


class CombinedLoss(nn.Module):
    def __init__(self, criterion_1=nn.MSELoss(), criterion_2=PerceptualLoss(),
                 alpha=1.0, beta=0.1):
        super(CombinedLoss, self).__init__()
        self.criterion_1 = criterion_1
        self.criterion_2 = criterion_2
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        criterion_loss = self.criterion_1(output, target)
        perceptual_loss = self.criterion_2(output, target)

        total_loss = self.alpha * criterion_loss + self.beta * perceptual_loss
        return total_loss
