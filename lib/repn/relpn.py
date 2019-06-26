import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.network import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RelPN(nn.Module):
    """ region proposal network """
    def __init__(self, dim=512, n_obj_classes=151):
        super(_RelPN, self).__init__()
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.feat_stride = cfg.FEAT_STRIDE[0]

        roi_feat_dim = n_obj_classes

        if cfg.TRAIN.RELPN_WITH_BBOX_INFO:
            roi_feat_dim += 4

        # define proposal layer
        self.RelPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales)

        # define anchor target layer
        self.RelPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, rois, roi_feat, nlp_feat, im_info, gt_boxes, gt_relation, num_boxes, use_gt_boxes=False):


        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'########
        #roi_proposals is the index of sub and obj rois, roi_pairs_scores is the score of these roi pairs.
        #the first dim of all of these three is batch size
        #roi_pairs, roi_proposals, roi_pairs_scores = self.RelPN_proposal((rois, relpn_cls_score.data, im_info, cfg_key))
        roi_pairs, roi_proposals, roi_pairs_scores = self.RelPN_proposal((rois, im_info, roi_feat, nlp_feat, cfg_key))
        relpn_loss_cls = 0

        # generating training labels and build the rpn loss
        relpn_eval = [0,0,0]
        if self.training:
            assert gt_boxes is not None, "gt_boxes should not be none"

            relpn_label,relpn_eval = self.RelPN_anchor_target((roi_pairs, roi_pairs_scores.data, gt_boxes, gt_relation, im_info, num_boxes))

            # compute classification loss
            relpn_keep = Variable(relpn_label.view(-1).ne(-1).nonzero().view(-1))

            relpn_cls_score = roi_pairs_scores.view(-1,1)[relpn_keep]
            relpn_label = relpn_label.view(-1)[relpn_keep.data]
            relpn_label = Variable(relpn_label.long())
            try:
                assert((relpn_cls_score >=0.) & (relpn_cls_score <= 1.)).all()
            except:
                print('omg sigmoid')
            relpn_loss_cls = F.mse_loss(relpn_cls_score.squeeze(), relpn_label.float())
            #relpn_loss_cls = torch.sqrt(relpn_loss_cls)
            #print('relpn_loss_cls {}'.format(relpn_loss_cls))
            # if relpn_eval[2] > 0.1:
            #     relpn_loss_cls /= relpn_eval[2]
            # else:
            #     relpn_loss_cls *= 5
        return roi_pairs, roi_proposals, roi_pairs_scores, relpn_loss_cls, relpn_eval
