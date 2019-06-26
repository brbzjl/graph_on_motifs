import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.autograd import Variable
import numpy as np
import time
import pdb
from matplotlib import  pyplot as plt
from lib.model.utils.config import cfg
from lib.model.roi_layers import ROIAlign, ROIPool
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from lib.model.repn_new.relpn import _RelPN
from lib.model.repn_new.relpn_target_layer import _RelProposalTargetLayer


from lib.model.graph_conv.graph_conv import _GraphConvolutionLayer as _GCN_2
from lib.model.graph_conv.graph_attention import _GraphAttentionLayer as _GCN_ATT
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

from config import NUM_phrases, FIX_length

class lstm_encoder_spinn(nn.Module):
    def __init__(self):
        super(lstm_encoder_spinn, self).__init__()
        self.rnn = nn.LSTM(
            input_size=300, hidden_size=64,#300 is the size of spinn model
            num_layers=1)#batch_first=True if you need to switch the seq_len and batch_size dimensions of the input and output.
        self.out = nn.Linear(64,64)
    # def reset_state(self):
    #     self.state = None
    def forward(self, spinn_res):
        #
        # # initial lstm state
        # self.reset_state()
        # if self.state is None:
        #     self.state = 2 * [Variable(spinn_res.data.new(50, 64).zero_())]  # hx,cx
        spinn_res = spinn_res.permute(1,0,2).float()
        r_out, state = self.rnn(spinn_res, None)# none means 0 inital state
        return self.out(r_out[-1,:,:])


class EncoderLSTM_w2v(nn.Module):

    def __init__(
        self, embed_dim=60, hidden_size=64, dropout=0.1,
    ):
        # input_size : length of sequence
        # embed_dim : embedding size
        # hidden_size : the size of hidden layer of lstm
        super(EncoderLSTM_w2v, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False,

        )
        self.out = nn.Linear(hidden_size, 64)

    def forward(self, inputs):
        # Get the output from the LSTM. inputs 10 x 60 x 5
        inputs = inputs.permute(2, 0, 1).float()
        outputs,state = self.lstm(inputs,None)
        # Return the Encoder's output.
        # sequence x minibatch x features length
        return  self.out(outputs[-1, :, :])


class graphRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, n_obj_classes, n_rel_classes,pooled_feat_dim=4096):
        '''

        :param obj_classes:
        :param att_classes:
        :param rel_classes:
        :param dout_base_model: ## todo???
        :param pooled_feat_dim: ## todo???
        '''
        super(graphRCNN, self).__init__()

        self.n_obj_classes = n_obj_classes
        self.n_rel_classes = n_rel_classes

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_obj_classes)
        # define aGCN
        if cfg.HAS_RELATIONS:
            self.encoder = EncoderLSTM_w2v()
            self.RELPN = _RelPN(pooled_feat_dim, self.n_obj_classes)
            self.RELPN_proposal_target = _RelProposalTargetLayer(self.n_rel_classes)

            self.RELPN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
            self.RELPN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)
            self.RELPN_grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE


            num_rois = cfg['TRAIN'].BATCH_SIZE * 5
            self.bn_obj = nn.BatchNorm1d(4096 ,affine=False)
            self.bn_rel = nn.BatchNorm1d(4096, affine=False)

        reduced_pooled_feat_dim = pooled_feat_dim
        #################////////////////////////////////////////////////////////////////////take care
        self.gcn_head_rel_sub = nn.Linear(pooled_feat_dim, int(pooled_feat_dim/2))
        self.GRCNN_obj_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_obj_classes)
        self.GRCNN_rel_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_rel_classes)


        if cfg.GCN_LAYERS > 0:

            if cfg.GCN_ON_FEATS and not cfg.GCN_SHARE_FEAT_PARAMS:  # true
                self.GRCNN_gcn_feat = _GCN_2(reduced_pooled_feat_dim, dropout=0.1)

            if cfg.GCN_HAS_ATTENTION:
                self.GRCNN_gcn_att1 = _GCN_ATT(pooled_feat_dim)  # self.n_obj_classes)

            self.fcsub = nn.Linear(reduced_pooled_feat_dim, reduced_pooled_feat_dim)
            self.fcrel = nn.Linear(reduced_pooled_feat_dim, reduced_pooled_feat_dim)

        self.GRCNN_loss_obj_cls = 0
        self.GRCNN_loss_rel_cls = 0
        self.res_spinn = []
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.GRCNN_obj_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.GRCNN_rel_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.gcn_head_rel_sub, 0, 0.01, cfg.TRAIN.TRUNCATED)


    def create_architecture(self, ext_feat=False):

        self._init_weights()
        self.ext_feat = ext_feat

    def forward(self, rois, bbox_pred, im_info, obj_cls_score, obj_cls_feat, spinn_res, rois_obj_label, gt_boxes, gt_relation, use_gt_boxes=False):

        batch_size = rois.size(0)
        num_boxes = min(gt_boxes.size(0), cfg.MAX_NUM_GT_BOXES)

        #batch normalization
        _obj_cls_feat = obj_cls_feat#self.bn_obj(obj_cls_feat)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED and not use_gt_boxes:
            box_deltas = bbox_pred.data
            # conversly normalize targets by a precomputed mean and stdev this is done in RCNN_proposal_target
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(bbox_pred.size(0), -1, 4)
            boxes = rois.data[:, :, 1:5]
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info, 1)
            rois.data[:, :, 1:5] = pred_boxes

        if spinn_res is not None:
            #todo: when the spinn_res are all zero matrix
            encoder_res = self.encoder(spinn_res)
            self.res_spinn.append(encoder_res)
        else:
            encoder_res = None

        relpn_feats = _obj_cls_feat.view(rois.size(0), rois.size(1), _obj_cls_feat.size(1)) #todo: last size wether 1 or 2

        roi_rel_pairs, roi_pair_proposals, roi_rel_pairs_score, relpn_loss_cls, relpn_eval= \
            self.RELPN(rois.data, relpn_feats, encoder_res, im_info, gt_boxes.data, gt_relation.data, num_boxes, use_gt_boxes)

        if not self.training:
            if batch_size == 1:
                valid = roi_rel_pairs.sum(2).view(-1).nonzero().view(-1)
                roi_rel_pairs = roi_rel_pairs[:, valid, :]
                roi_pair_proposals = roi_pair_proposals[:, valid, :]
                roi_rel_pairs_score = roi_rel_pairs_score[:, valid, :]

        size_per_batch = _obj_cls_feat.size(0) / batch_size
        # xxx = torch.arange(0, batch_size).view(batch_size, 1, 1).type_as(roi_pair_proposals) * size_per_batch
        roi_pair_proposals = roi_pair_proposals + \
                             torch.arange(0, batch_size).view(batch_size, 1, 1).type_as(roi_pair_proposals) * size_per_batch

        roi_pair_proposals_v = roi_pair_proposals.view(-1, 2)
        ind_subject = roi_pair_proposals_v[:, 0]
        ind_object = roi_pair_proposals_v[:, 1]

        if self.training:
            roi_rel_pairs, rois_rel_label, roi_pair_keep = \
                self.RELPN_proposal_target(roi_rel_pairs, gt_boxes.data, gt_relation.data, num_boxes)

            rois_rel_label = Variable(rois_rel_label.view(-1))
            xxx = torch.arange(0, roi_pair_keep.size(0)).view(roi_pair_keep.size(0), 1).cuda() * roi_pair_proposals_v.size(0)
            x = xxx / batch_size
            # roi_pair_keep = roi_pair_keep + torch.arange(0, roi_pair_keep.size(0)).view(roi_pair_keep.size(0), 1).cuda() \
            #                                 * roi_pair_proposals_v.size(0) / batch_size
            roi_pair_keep = roi_pair_keep + x.float()
            roi_pair_keep = roi_pair_keep.view(-1).long()

            ind_subject = roi_pair_proposals_v[roi_pair_keep][:, 0]
            ind_object = roi_pair_proposals_v[roi_pair_keep][:, 1]

        _obj_cls_feat_sub = self.gcn_head_rel_sub(_obj_cls_feat)
        x_sobj = _obj_cls_feat_sub[ind_subject] #1500 x 4096 , 640
        x_oobj = _obj_cls_feat_sub[ind_object]

        pred_feat = torch.cat((x_sobj, x_oobj), 1)

        # compute object classification probability
        #pred_feat = self.gcn_head_rel_fc(_pred_feat)
        # pred_feat = self.bn_rel(pred_feat)

        ## ============================================================================================================================
        ##                  GCN
        ## ============================================================================================================================
        if cfg.GCN_ON_FEATS and cfg.GCN_LAYERS > 0:  # true

            x_obj_gcn, x_pred_gcn = self.GRCNN_gcn_feat(_obj_cls_feat, pred_feat, ind_subject, ind_object)
            # x_obj_gcn = self.fcsub(_obj_cls_feat)
            # #x_obj_gcn = self.bn_obj(x_obj_gcn)
            #
            # x_pred_gcn = self.fcrel(pred_feat)
            # x_pred_gcn = self.bn_rel(x_pred_gcn)
            ## ============================================================================================================================
            ## LSTM endoder Layer
            ## ============================================================================================================================
            # self.encoder.reset_state()

            # if cfg.GCN_HAS_ATTENTION:  # true
            #
            #     attend_score = self.GRCNN_gcn_att1(x_sobj, x_oobj, None)  # N_rel x 1
            #     attend_score = attend_score.view(1, x_pred_relpn.size(0))
            #
            #
            # # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
            # # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
            # # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
            # # NOTE exclude one roi feature from another.
            # # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
            # # cooccurence, such as bus are more frequently appear on the road.
            # # assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"
            #
            # size_per_batch = obj_cls_feat.size(0) / batch_size
            #
            # map_obj_obj = obj_cls_feat.data.new(obj_cls_feat.size(0), obj_cls_feat.size(0)).fill_(0.0)
            # eye_mat = torch.eye(int(size_per_batch)).type_as(obj_cls_feat.data)
            # for i in range(batch_size):
            #     a = int(i * size_per_batch)  # size_per_batch 128
            #     b = int((i + 1) * size_per_batch)
            #     c = map_obj_obj[a:b, a:b]
            #     c.fill_(1.0)
            #     map_obj_obj[a:b, a:b].fill_(1.0)
            #     map_obj_obj[a:b, a:b] = map_obj_obj[a:b,
            #                             a:b] - eye_mat  # 256x256 block diagnal matrix, diagnal elements are 0
            # #todo: change adjacent matrix
            # map_obj_obj = Variable(map_obj_obj)
            #
            # map_sobj_rel = Variable(obj_cls_feat.data.new(obj_cls_feat.size(0), x_pred_relpn.size(0)).zero_())
            # map_sobj_rel.scatter_(0, Variable(ind_subject.contiguous().view(1, x_pred_relpn.size(0))), attend_score)
            # map_oobj_rel = Variable(obj_cls_feat.data.new(obj_cls_feat.size(0), x_pred_relpn.size(0)).zero_())
            # map_oobj_rel.scatter_(0, Variable(ind_object.contiguous().view(1, x_pred_relpn.size(0))), attend_score)
            # map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 2)
            #
            # gcnstart = time.time()
            #
            # x_obj_gcn = obj_cls_feat
            # x_pred_gcn = x_pred_relpn
            # for i in range(cfg.GCN_LAYERS):  # cfg.GCN_LAYERS
            #     # pass graph representation to gcn
            #     x_obj_gcn, x_pred_gcn = self.GRCNN_gcn_feat(x_obj_gcn, x_pred_gcn, map_obj_obj, map_obj_rel)
            #
            #     x_sobj = x_obj_gcn[ind_subject]
            #     x_oobj = x_obj_gcn[ind_object]
            #     attend_score = self.GRCNN_gcn_att1(x_sobj, x_oobj, None)  # N_rel x 1
            #     attend_score = attend_score.view(1, x_pred_gcn.size(0))
            #
            #     map_sobj_rel = Variable(obj_cls_feat.data.new(obj_cls_feat.size(0), x_pred_gcn.size(0)).zero_())
            #     map_sobj_rel.scatter_(0, Variable(ind_subject.contiguous().view(1, x_pred_gcn.size(0))), attend_score)
            #     map_oobj_rel = Variable(obj_cls_feat.data.new(obj_cls_feat.size(0), x_pred_gcn.size(0)).zero_())
            #     map_oobj_rel.scatter_(0, Variable(ind_object.contiguous().view(1, x_pred_gcn.size(0))), attend_score)
            #     map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 2)
                # 256x4096
                # pdb.set_trace()
                # compute object classification loss
            gcn_obj_cls_score = self.GRCNN_obj_cls_score(x_obj_gcn)
            gcn_obj_cls_prob = F.softmax(gcn_obj_cls_score, 1)


            gcn_rel_cls_score = self.GRCNN_rel_cls_score(x_pred_gcn)
            gcn_rel_cls_prob = F.softmax(gcn_rel_cls_score, dim=1)
            ## ============================================================================================================================
            ##                  LOSS function
            ## ============================================================================================================================
            if self.training:


                if cfg.GCN_LAYERS > 0:
                    # object classification los
                    self.GRCNN_loss_obj_cls = F.cross_entropy(gcn_obj_cls_score, rois_obj_label.long())
                    # relation classification los
                    self.rel_fg_cnt = torch.sum(rois_rel_label.data.ne(0))
                    self.rel_bg_cnt = rois_rel_label.data.numel() - self.rel_fg_cnt
                    self.GRCNN_loss_rel_cls = F.cross_entropy(gcn_rel_cls_score, rois_rel_label.long())
                    grcnn_loss = self.GRCNN_loss_obj_cls + self.GRCNN_loss_rel_cls  # used only for rpn relpn training

            relpn_loss = relpn_loss_cls
            gcn_obj_cls_prob = gcn_obj_cls_prob.view(batch_size, rois.size(1), -1)


            gcn_rel_cls_prob = gcn_rel_cls_prob.view(batch_size, int(gcn_rel_cls_prob.size(0) / batch_size), -1)
            ## ============================================================================================================================
            ##                  Return Values
            ## ============================================================================================================================

            if cfg.HAS_RELATIONS:
                if self.training:  # true use this option
                    return  rois, gcn_obj_cls_prob, gcn_rel_cls_prob, self.GRCNN_loss_obj_cls, relpn_loss, grcnn_loss, relpn_eval
                    # return rois, bbox_pred_frcnn, obj_cls_prob_frcnn, att_cls_prob, rel_cls_prob, rpn_loss, relpn_loss, grcnn_loss
                else:

                    return rois, roi_pair_proposals, gcn_obj_cls_prob, gcn_rel_cls_prob, roi_rel_pairs_score, 0, 0
        else:
            return relpn_eval, relpn_loss_cls