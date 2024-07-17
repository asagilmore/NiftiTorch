import torch


class RandomRotation90(torch.nn.Module):
    def __init__(self, axis=[1, 2]):
        super(RandomRotation90, self).__init__()
        self.axis = axis

    def forward(self, img, mask):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, self.axis), torch.rot90(mask, k, self.axis)
