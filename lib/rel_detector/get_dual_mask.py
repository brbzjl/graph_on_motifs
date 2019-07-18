"""
credits to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/network.py#L91
"""

import torch
from torch.autograd import Variable

from matplotlib import pyplot as plt

from torch import nn
from config import BATCHNORM_MOMENTUM, IM_SCALE
from lib.model.roi_layers import ROIAlign

class DualMaskFeats(nn.Module):
    def __init__(self, im_size = IM_SCALE, pooling_size=7, dim=512, concat = True ):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(DualMaskFeats, self).__init__()


        self.im_size = im_size
        self.pooling_size = pooling_size
        self.dim = dim
        self.concat = concat
        self.conv = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=7, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(dim*49), 2048),#20736 25088
            nn.ReLU(inplace=True)
        )

    def forward(self, rois, union_inds):
        """
        :param fmap: (batch_size, d, IM_SIZE / stride, IM_SIZE / stride)
        :param rois: (num_rois, 5)with [im_ind, x1, y1, x2, y2]
        :param union_inds: (num_urois, 2) with [roi_ind1, roi_ind2]
        """
        rois = rois.cpu()
        union_inds = union_inds.cpu()
        uni_size = 32.0
        dual_masks = []
        for i in range(union_inds.size(0)):
            xy11,xy12 = rois[union_inds[i, 0], 1:3], rois[union_inds[i, 0], 3:5]
            xy21,xy22 = rois[union_inds[i, 1], 1:3], rois[union_inds[i, 1], 3:5]
            # u1 = torch.min(xy11, xy21)
            # u2 = torch.max(xy12, xy22)

            w,h = 592,592#u2-u1

            # xy11 -= u1
            # xy12 -= u1
            # xy21 -= u1
            # xy22 -= u1

            sub_mask = getMask(uni_size, h, w, torch.cat((xy11,xy12)))
            obj_mask = getMask(uni_size, h, w, torch.cat((xy21,xy22)))
            dual_mask = torch.cat((sub_mask,obj_mask), 0).type(torch.float)


            dual_masks.append(dual_mask)

            # plt.figure(1)
            # plt.imshow(dual_mask[0,:,:].cpu())
            # plt.figure(2)
            # plt.imshow(dual_mask[1,:,:].cpu())
            # plt.show()
            # plt.pause(0.1)

        dual_masks = Variable(torch.stack(dual_masks)).cuda()
        dual_feats = self.conv(dual_masks)
        dual_feats = self.fc(dual_feats.view(dual_masks.size(0),-1))

        return dual_feats


def getMask(uni_size, ih, iw, bb):
    rh = uni_size / ih
    rw = uni_size / iw
    x1 = torch.max(torch.tensor(0).int(),  torch.floor(bb[0] * rw).int())
    x2 = torch.min(torch.tensor(32).int(), torch.ceil( bb[2] * rw).int())
    y1 = torch.max(torch.tensor(0).int(),  torch.floor(bb[1] * rh).int())
    y2 = torch.min(torch.tensor(32).int(), torch.ceil( bb[3] * rh).int())
    mask = torch.zeros((1, int(uni_size), int(uni_size)),dtype=torch.int)
    mask[:, y1 : y2, x1 : x2] = 1
    try:
        assert(mask.sum().int() == (y2 - y1) * (x2 - x1))
    except:
        print('let see what happens')
    return mask