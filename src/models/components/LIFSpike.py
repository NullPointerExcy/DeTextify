import norse.torch as snn
from torch import nn


class LIFSpike(nn.Module):
    def __init__(self):
        super(LIFSpike, self).__init__()
        self.lif_cell = snn.LIFCell()

    def forward(self, x):
        spikes, _ = self.lif_cell(x)
        return spikes
