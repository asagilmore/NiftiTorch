import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.weights = []

    def add_loss(self, loss, weight):
        self.losses.append(loss)
        self.weights.append(weight)

    def forward(self, output, target):
        combined_loss = 0.0
        for i, loss in enumerate(self.losses):
            combined_loss += self.weights[i] * loss(output, target)

        return combined_loss

    def forward_validate(self, output, target):
        combined_loss = 0.0
        loss_dict = {}
        for i, loss in enumerate(self.losses):
            loss_val = loss(output, target)
            loss_dict[type(loss).__name__] = loss_val
            combined_loss += self.weights[i] * loss_val

        return combined_loss, loss_dict
