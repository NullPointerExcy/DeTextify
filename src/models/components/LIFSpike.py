import norse.torch as snn
from norse.torch import LIFParameters
from torch import nn


class LIFSpike(nn.Module):
    def __init__(self, p: LIFParameters = None):
        super(LIFSpike, self).__init__()
        if p is None:
            self.lif_cell = snn.LIFCell()
        else:
            self.lif_cell = snn.LIFCell(p)

    def forward(self, x):
        spikes, _ = self.lif_cell(x)
        return spikes
