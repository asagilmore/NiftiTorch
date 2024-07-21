import torch.nn as nn
import torch


class HistogramLoss(nn.Module):
    def __init__(self, num_bins=64, min_val=None, max_val=None,
                 squared_diff=False):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.squared_diff = squared_diff

    def forward(self, output, target):
        if self.min_val is None:
            min_val = torch.min(output.min(), target.min())
        else:
            min_val = self.min_val
        if self.max_val is None:
            max_val = torch.max(output.max(), target.max())
        else:
            max_val = self.max_val

        hist_output = torch.histc(output, bins=self.num_bins, min=min_val,
                                  max=max_val)
        hist_target = torch.histc(target, bins=self.num_bins, min=min_val,
                                  max=max_val)

        diff = torch.abs(hist_output - hist_target)

        if self.squared_diff:
            diff = diff ** 2

        loss = torch.mean(diff)

        return loss
