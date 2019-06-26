import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
import numpy as np
import math
import pdb
import time
import pdb
from model.utils.config import cfg

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class _Collection_Unit(nn.Module):
    def __init__(self, dim):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim, dim)
        normal_init(self.fc, 0, 0.001, cfg.TRAIN.TRUNCATED)
    def forward(self, source, adjacent):
        #assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
        fc_out = self.fc(source)
        #collect = fc_out
        #todo: change to sparse matrix
        collect = torch.mm(adjacent, fc_out)
        collect_avg = collect /(adjacent.sum(1).view(collect.size(0), 1) + 1e-7)

        return collect_avg

class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()
        # self.fc = nn.Linear(dim, dim, bias=True)
        # normal_init(self.fc, 0, 0.001, cfg.TRAIN.TRUNCATED)
    def forward(self, target, source):
        #assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """

    def __init__(self, dim):
        super(_GraphConvolutionLayer_Collect, self).__init__()

        self.collect_units = nn.ModuleList([_Collection_Unit(dim) for i in range(2)])  # 5 with out attribute

    def forward(self, source, attention, unit_id):
        collection = self.collect_units[unit_id](source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """

    def __init__(self, dim):
        super(_GraphConvolutionLayer_Update, self).__init__()

        self.update_units = nn.ModuleList([_Update_Unit(dim) for i in range(2)])  # 5 with out attribute

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update
