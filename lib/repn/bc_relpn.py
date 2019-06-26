# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import yaml
from lib.model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.co_nms.co_nms_wrapper import co_nms
from model.roi_layers import nms
import pdb
from torch.autograd import Variable

DEBUG = False


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales):
        roi_feat_dim = 4096
        super(_ProposalLayer, self).__init__()
        if cfg.TRAIN.RELPN_WITH_BBOX_INFO:
            roi_feat_dim += 4
        self.sub_feat_size = 256
        self.sub_feat = nn.Linear(roi_feat_dim, self.sub_feat_size)

        self.select_NN = nn.Sequential(
            nn.Linear(self.sub_feat_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )

        self.lstm_encoder = nn.LSTM(
            input_size=self.sub_feat_size,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,

        )
        self.lstm_out = nn.Linear(2 * 64, 64)

    def forward(self, input, use_gt_boxes=False):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        rois = input[0]
        im_info = input[1]
        roi_feat = input[2]
        nlp_features = input[3]
        cfg_key = input[4]

        ####################################################
        ### ###
        ####################################################
        assert roi_feat.dim() == 3, "roi_feat must be B x N x D shape"
        B = roi_feat.size(0)
        N = roi_feat.size(1)
        D = roi_feat.size(2)

        if cfg.TRAIN.RELPN_WITH_BBOX_INFO:
            rois_nm = rois.new(rois.size(0), rois.size(1), 4)
            xx = im_info[:, 1]
            yy = im_info[:, 0]
            rois_nm[:, :, :2] = rois[:, :, 1:3]  # / xx[:,None]
            rois_nm[:, :, 2:] = rois[:, :, 3:5]  # / yy[:,None]
            roi_feat4prop = torch.cat((roi_feat, Variable(rois_nm)), 2)
            D += 4
        else:
            roi_feat4prop = roi_feat  #
        # roi_feat4prop = roi_feat4prop.view(B * N, D)
        roi_feat4prop = self.sub_feat(roi_feat4prop)  # feat dim reduction to 256
        batch_size = rois.size(0)
        pre_nms_topN = cfg[cfg_key].RELPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RELPN_POST_NMS_TOP_N  # cfg[cfg_key].RELPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RELPN_NMS_THRESH
        min_size = cfg[cfg_key].RELPN_MIN_SIZE

        if DEBUG:
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))

        ####################################################
        ###  Method 1.Use coorelation with nlp fv to compute the scores   ###
        ####################################################
        map_x = np.arange(0, rois.size(1))
        map_y = np.arange(0, rois.size(1))
        map_x_g, map_y_g = np.meshgrid(map_x, map_y)
        map_yx = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose()).cuda()
        proposals = map_yx.expand(batch_size, rois.size(1) * rois.size(1), 2)  # B x (N * N) x 2

        # filter diagnal entries
        keep = self._filter_diag(proposals)
        proposals = proposals.contiguous().view(-1, 2)[keep.nonzero().squeeze(), :].contiguous().view(batch_size, -1,
                                                                                                      2).contiguous()
        # -------------using NN to encode the pair feature----------------
        nlp_features_repeated = nlp_features.unsqueeze(1).repeat(1, proposals.size(1), 1)
        # TODO: add new score method:
        all_box_pairs_fet = []  # bs x pairs_Num x 151
        all_box_pairs_roi = []  # bs x pairs_Num x 8
        all_box_pairs_score = []
        for b in range(batch_size):
            # torch.cuda.empty_cache()
            # proposals_subject_roi_i = rois[b][proposals[b, :, 0], :][:, 1:5]
            # proposals_object_roi_i = rois[b][proposals[b, :, 1], :][:, 1:5]
            proposals_subject_fet_i = roi_feat4prop[b][proposals[b, :, 0], :]  # [:, 1:5]
            proposals_object_fet_i = roi_feat4prop[b][proposals[b, :, 1], :]  # [:, 1:5]

            # -------------using NN to encode the pair feature----------------
            # all_box_pairs_fet.append(torch.cat((proposals_subject_fet_i, proposals_object_fet_i), 1))

            # -------------using bi-lstm to encode the pair feature----------------
            # Get the output from the LSTM. inputs pairNum x 2 x 256

            box_pairs_fet = (torch.cat((proposals_subject_fet_i, proposals_object_fet_i), 0)).view(2, -1,
                                                                                                   self.sub_feat_size)
            outputs, state = self.lstm_encoder(box_pairs_fet, None)
            # Return the Encoder's output.
            # sequence x minibatch x features length
            select_features = self.lstm_out(outputs[-1, :, :])
            scores_i = F.cosine_similarity(select_features, nlp_features_repeated[b], dim=1, eps=1e-6)

        # -------------using NN to encode the pair feature----------------
        # select_features = self.select_NN(all_box_pairs_fet)
        # caluculate cls score of fg and bg prediction
        # caluculate cos similarity
        # nlp_features = torch.unsqueeze(nlp_features, dim=1)
        # scores_all = []
        # for i in range(1):#nlp_features.size(1)
        #     nlp_features_repeated_i = nlp_features[:,i,:].unsqueeze(1).repeat(1, all_box_pairs_fet.size(1), 1)
        #
        #     scores_i = F.cosine_similarity(select_features, nlp_features_repeated_i, dim=2, eps=1e-6)
        #     scores_all.append(scores_i.unsqueeze(1))
        # scores_all = torch.cat(scores_all,1) # bs x regions_num x scores
        # scores,_ = torch.max(scores_all,1) # bs x  scores
        # nlp_features_repeated_i = nlp_features.unsqueeze(1).repeat(1, all_box_pairs_fet.size(1), 1)
        #
        # scores = F.cosine_similarity(select_features, nlp_features_repeated_i, dim=2, eps=1e-6)
        #     scores_all.append(scores_i.unsqueeze(1))
        _, order = torch.sort(scores, 1, True)
        # _, order = torch.sort(scores)
        post_nms_topN = proposals.size(1)
        output = scores.new(batch_size, post_nms_topN, 9).zero_()
        output_score = scores.new(batch_size, post_nms_topN, 1).zero_()
        output_proposals = proposals.new(batch_size, post_nms_topN, 2).zero_()

        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_i = proposals[i]
            scores_i = scores[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_i = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores.numel():
                order_single = order_i[:pre_nms_topN]
            else:
                order_single = order_i

            proposals_single = proposals_i[order_single, :]
            scores_single = scores_i[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            if not use_gt_boxes:
                xx = proposals_single[:, 0]
                xxx = rois[i][xx, :]
                proposals_subject = rois[i][proposals_single[:, 0], :][:, 1:5]
                proposals_object = rois[i][proposals_single[:, 1], :][:, 1:5]

                rel_rois_final = torch.cat((proposals_subject, proposals_object), 1)

                keep_idx_i = nms(rel_rois_final, scores_single.squeeze(1), nms_thresh).long().view(-1)

                keep_idx_i = keep_idx_i.long().view(-1)

                if post_nms_topN > 0:
                    keep_idx_i = keep_idx_i[:post_nms_topN]
                proposals_single = proposals_single[keep_idx_i, :]
                scores_single = scores_single[keep_idx_i, :]
            else:
                proposals_single = proposals_single[:post_nms_topN, :]
                scores_single = scores_single[:post_nms_topN, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i, :num_proposal, 0] = i
            output[i, :num_proposal, 1:5] = rois[i][proposals_single[:, 0], :][:, 1:5]
            output[i, :num_proposal, 5:] = rois[i][proposals_single[:, 1], :][:, 1:5]
            output_score[i, :num_proposal, 0] = scores_single.squeeze()
            output_proposals[i, :num_proposal, :] = proposals_single
        return output, output_proposals, output_score

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep

    def _filter_diag(self, roi_pairs):
        """Remove all boxes with any side smaller than min_size."""
        keep = roi_pairs[:, :, 0] != roi_pairs[:, :, 1]
        return keep.view(-1)
