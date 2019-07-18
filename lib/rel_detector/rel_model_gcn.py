"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
# from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM

from lib.model.box_utils import bbox_preds, center_size, bbox_overlaps, apply_nms
# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from .lstm.decoder_rnn import DecoderRNN
# from .lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM

from .get_union_boxes import UnionBoxesAndFeats
from .get_dual_mask import DualMaskFeats
from lib.model.proposal_assignments.rel_assignments import rel_assignments
from lib.model.object_dectector.object_detector_motifs import ObjectDetector, gather_res, load_vgg
from lib.model.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, \
    Flattener
from lib.model.sparse_targets import FrequencyBias
from lib.model.surgery import filter_dets
from lib.model.word_vectors import obj_edge_vectors
from lib.model.roi_layers import ROIAlign
from lib.model.graph_conv.graph_conv_attention import _GraphConvolutionLayer as GCN
import math
import copy
from lib.model.rel_model.triplet_loss import SemihardNegativeTripletSelector, OnlineTripletLoss, pdist
def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1
    rois_per_image = scores.new(num_im.item())
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


MODES = ('sgdet', 'sgcls', 'predcls')

class obj_semamtic_feat(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, predicate, obj_feat_size, embed_dim=200, concat_vis_feat = False):
        super(obj_semamtic_feat, self).__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.predicate = predicate
        self.num_predicate = len(self.predicate)
        self.embed_dim = embed_dim
        # EMBEDDINGS
        obj_embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        obj_embed.weight.data = obj_embed_vecs.clone()
        cls = torch.LongTensor([i for i in range(self.num_classes)])
        self.obj_semantic_embed =  obj_embed(cls).cuda()

        pred_embed_vecs = obj_edge_vectors(self.predicate, wv_dim=self.embed_dim)
        pred_embed = nn.Embedding(self.num_predicate, self.embed_dim)
        pred_embed.weight.data = pred_embed_vecs.clone()
        pred = torch.LongTensor([i for i in range(self.num_predicate)])
        self.pred_semantic_embed = pred_embed(pred).cuda()
        self.concat_vis_feat = concat_vis_feat

        # if self.concat_vis_feat:
        #     self.com_dual_mask = nn.Linear(obj_feat_size, int(obj_feat_size/2))
        #     self.com_obj = nn.Linear(obj_feat_size, int(obj_feat_size/4)-self.embed_dim)
        #     self.com_uni = nn.Linear(obj_feat_size, int(obj_feat_size / 2) )
        # else:
        #     self.com_dual_mask = nn.Linear(int((self.embed_dim)), int(obj_feat_size/2))
        #     self.com_obj = nn.Linear(int((self.embed_dim)), int(obj_feat_size/2))
        self.rel_semantic_out = nn.Linear(int((self.embed_dim)), int(self.embed_dim))
        self.obj_semantic_out = nn.Linear(int((self.embed_dim)), int(self.embed_dim))
        # from new paper large scale vrd
        concat_size = int(3 * embed_dim)
        inter_size = int(obj_feat_size / 2)
        self.s_o_linears  = nn.Sequential(
            nn.Linear(obj_feat_size, inter_size),  #
            nn.ReLU(inplace=True),
            nn.Linear(inter_size, embed_dim),  #
            nn.ReLU(inplace=True),
        )
        self.s_o_linears_out = nn.Linear(embed_dim, embed_dim) #
        self.pred_linears  = nn.Sequential(
            nn.Linear(obj_feat_size, inter_size),
            nn.ReLU(inplace=True),
            nn.Linear(inter_size, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pred_linears_inter = nn.Sequential(
            nn.Linear(concat_size, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pred_linears_out = nn.Sequential(
            nn.Linear(concat_size, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),  #
        )

    def forward(self, obj_preds_label,  obj_feats, rel_feats, im_inds, rel_inds, gt_rel_label, gt_obj_label=None):
        """
        Object context and object classification.
        :param obj_preds_label: one hot obj class
        :param im_inds: [num_obj] the indices of the images

        :return: edge_ctx: [num_obj, #feats] For later!
        """
        #Only use hard embeddings
        sub_ind, obj_ind = rel_inds[:, 1], rel_inds[:,2]
        invalid_obj = obj_preds_label == -1
        # try:
        if invalid_obj.sum():
            obj_preds_label[invalid_obj] = 0
        with torch.no_grad():
            rel_semantic = self.pred_semantic_embed
            obj_semantic = self.obj_semantic_embed
        rel_semantic_gt = self.rel_semantic_out(rel_semantic)
        obj_semantic_gt = self.obj_semantic_out(obj_semantic)

        if gt_rel_label is not None:
            rel_semantic = rel_semantic_gt[gt_rel_label]
            obj_semantic = obj_semantic_gt[gt_obj_label]
        else:
            rel_semantic = None
            obj_semantic = None

                # obj_feat = self.obj_embed(gt_obj_label)
        #   with torch.no_grad():
        #         sub_feat = self.obj_embed(obj_preds_label)[sub_ind]
        #         obj_feat = self.obj_embed(obj_preds_label)[obj_ind]
        #       oobj_feat = self.obj_embed(obj_preds_label)
        # except:
        #     print('wtf')
        #
        # if self.concat_vis_feat:# num_objs x embed_dim
        #     com_feat = self.com_obj(obj_feats)
        #     sub_feat = torch.cat((sub_feat, com_feat[sub_ind]), 1)
        #     obj_feat = torch.cat((com_feat[obj_ind], obj_feat), 1)

        # if self.concat_vis_feat:# num_objs x embed_dim
        #
        #     return torch.cat((obj_feats, dual_mask_feats), 1)

        # sub_feat = self.com_sub(sub_feat)
        #comp_dual_mask_feats = self.com_dual_mask(dual_mask_feats)
        #pred_feat = torch.cat((sub_feat, obj_feat, comp_dual_mask_feats), 1)
        #return self.out(sub_feat - obj_feat)#obj_feats + dual_mask_feats
        #return sub_feat - obj_feat ,  oobj_feat # obj_feats + dual_mask_feats

        # from new paper large scale vrd


        s_o_inter =  self.s_o_linears(obj_feats)
        s_o_out = self.s_o_linears_out(s_o_inter)
        sub_inter = s_o_inter[sub_ind]
        obj_inter = s_o_inter[sub_ind]
        sub_out = s_o_out[sub_ind]
        obj_out = s_o_out[obj_ind]


        pred_inter = self.pred_linears(rel_feats)
        pred_inter_bf = torch.cat((sub_inter,pred_inter,obj_inter),dim=1)
        pred_inter_af = self.pred_linears_inter(pred_inter_bf)
        pred_inter_out = torch.cat((sub_out,pred_inter_af,obj_out),dim=1)
        pred_out = self.pred_linears_out(pred_inter_out)
        return s_o_out, pred_out, rel_semantic, obj_semantic, rel_semantic_gt, obj_semantic_gt
class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes,
                 mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.0, use_bias=True, use_tanh=True, limit_vision=True,
                 ):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 200#2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.obj_detector = ObjectDetector(
            classes=classes,
            mode='gtbox',#('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )
        self.obj_semamtic_feat = obj_semamtic_feat(self.classes,self.rel_classes, obj_feat_size=4096, embed_dim=self.embed_dim,
                                                   concat_vis_feat=True)
        ###################################
        self.post_fc = nn.Linear(self.embed_dim, self.pooling_dim)
        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                               dim=1024 if use_resnet else 512, use_feats=False)
        self.dual_mask_feat = DualMaskFeats(dim=64, concat = True)
        roi_fmap = [
            Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096,
                     pretrained=False).classifier,
        ]
        #if pooling_dim != 4096:
        roi_fmap.append(nn.Linear(4096, 2048))
        self.roi_fmap = nn.Sequential(*roi_fmap)
        self.dual_mask_compress = nn.Linear(self.obj_dim, len(self.classes))
        self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        self.GRCNN_gcn_feat = GCN( self.obj_dim, dropout=0.05)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_fc.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_fc.bias.data.zero_()

        # if nl_edge == 0:
        #     self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim * 2)
        #     self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        self.rel_compress = nn.Linear(self.obj_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()
        self.obj_compress = nn.Linear(self.obj_dim, len(self.classes))

        _TripletSelector = SemihardNegativeTripletSelector(0.2, cpu=False)
        self.OnlineTripletLoss = OnlineTripletLoss(0.2, _TripletSelector)
    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds, dual_mask_feats=None):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """

        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        # if self.concat_vis_feat:  # num_objs x embed_dim
        #     return self.roi_fmap(torch.cat((uboxes, dual_mask_feats), 1))
        if dual_mask_feats is None:
            return self.roi_fmap(uboxes)# + dual_mask_feats)
        else:
            uni_feat = self.roi_fmap(uboxes)
            vis_feat = torch.cat((uni_feat,dual_mask_feats),dim=1)
            return vis_feat


    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            #rel_inds = rel_labels[:, :3].data.clone()
            rel_inds = rel_labels.data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = ROIAlign((self.pooling_size, self.pooling_size), 1.0 / 16.0, 0)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, train_anchor_inds=None, return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        result = self.obj_detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        fmap = result.fmap.detach()
        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)
        # sub and obj inds of triplets
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        try:
            gt_rel_label = rel_inds[:, 3]
        except:
            gt_rel_label = None
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        #todo : fine tune fmap, comes from faster rcnn
        result.obj_fmap = self.obj_feature_map(fmap, rois)
        dual_mask_feat = self.dual_mask_feat(rois, rel_inds[:, 1:])
        pred_fmap = self.visual_rep(fmap, rois, rel_inds[:, 1:3], dual_mask_feats=dual_mask_feat)
        # get obj feature used for the rel prediction
        if self.mode in ['predcls','sgcls']:
            # todo: using gt obj_labels
            # vr = None
            dual_mask_feat = None
            obj_feat, pred_feat, rel_semantic, obj_semantic, rel_semantic_gt, obj_semantic_gt \
                = self.obj_semamtic_feat(result.rm_obj_labels, result.obj_fmap, pred_fmap, result.im_inds, rel_inds, gt_rel_label,result.rm_obj_labels)

        else:
            pred_feat = self.obj_semamtic_feat(result.obj_preds, result.obj_fmap, dual_mask_feat, result.im_inds, rel_inds)
        #edge_rep = self.post_fc(edge_ctx)

        # Split into subject and object representations
        # sub_feat = obj_ctx[rel_inds[:, 1]]  # 1500 x 4096 , 640
        # obj_feat = obj_ctx[rel_inds[:, 2]]
        #
        # pred_feat = torch.cat((sub_feat, obj_feat), 1)

        vis = False if self.training else True
        #result.obj_fmap.
        obj_gcn, pred_gcn = self.GRCNN_gcn_feat(obj_feat, pred_feat, rel_inds[:, 1], rel_inds[:, 2], gt_rel_label, self.classes, self.rel_classes,result.rm_obj_labels, vis )#self.classes, self.rel_classes,result.rm_obj_labels
        #obj_gcn, pred_gcn = obj_feat, pred_feat
        # result.rel_dists = self.rel_compress(pred_gcn)
        # result.obj_dists = self.obj_compress(obj_gcn)
        result.rel_dists = 1/pdist(pred_gcn, rel_semantic_gt)
        result.obj_dists = 1/pdist(obj_gcn,  obj_semantic_gt)
        valid_obj_idx = (result.rm_obj_labels != -1)
        if self.training:
            loss_rel = self.OnlineTripletLoss(pred_gcn, rel_semantic, gt_rel_label)
            loss_obj = self.OnlineTripletLoss(obj_gcn[valid_obj_idx], obj_semantic[valid_obj_idx], result.rm_obj_labels[valid_obj_idx])
            result.trip_loss = loss_rel + loss_obj

        # if self.use_vision:
        #
        #     vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
        #     if self.limit_vision:
        #         # exact value TBD
        #         prod_rep = torch.cat((prod_rep[:, :2048] * vr[:, :2048], prod_rep[:, 2048:]), 1)
        #     else:
        #         prod_rep = prod_rep * vr
        #
        # if self.use_tanh:re
        #     prod_rep = F.tanh(prod_rep)
        #
        #result.rel_dists = self.rel_compress(prod_rep)

        #
        # if self.use_bias:
        #     result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
        #         result.obj_preds[rel_inds[:, 1]],
        #         result.obj_preds[rel_inds[:, 2]],
        #     ), 1))

        if self.training:
            return result

        _, result.obj_preds = torch.max( result.obj_dists , 1)  # None
        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_scores = F.softmax(result.rel_dists, dim=1)
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:3], rel_scores), result

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0][:7])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
