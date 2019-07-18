import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from lib.model.graph_conv.graph_conv_unit import _GraphConvolutionLayer_Collect, _MessagePassingLayer
from matplotlib import pyplot as plt
import math
import copy
def show_heatmap(fig_ind, num, fig_name, map, x_name, y_name):

    plt.figure(fig_ind)
    plot_num = len(map)
    for i in range(plot_num):
        ax = plt.subplot(2,int(plot_num/2),i+1)
        # ax = plt.gca()
        # #ax = plt.subplots()
        ax.cla()
        ax.imshow(map[i].detach().cpu().numpy())

        # We want to show all ticks...
        ax.set_xticks(np.arange(map[i].size(1)))
        ax.set_yticks(np.arange(map[i].size(0)))

        ax.set_xticklabels(x_name)
        ax.set_yticklabels(y_name)
        ax.set_title(fig_name+'-'+str(i))
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig('/home/bai/MA/grcnn_based_onMotifs/attention_map/'+fig_name+str(num))
    #plt.show(block=False)
    plt.pause(0.01)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, query_v, key_v , mask=None, dropout=None, concat = True):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    p_attn_all_head1 = []
    p_attn_all_head2 = []
    scores_all = []
    for i in range(query.size(0)):
        scores = torch.matmul(query[i], torch.t(key[i])) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        p_attn1 = F.elu(torch.matmul(p_attn, key_v[i]))
        p_attn2 = F.elu(torch.matmul(torch.t(p_attn), query_v[i]))
        p_attn_all_head1.append(p_attn1)
        p_attn_all_head2.append(p_attn2)
        scores_all.append(p_attn)
    if concat:
        p_attn_all_head1 = torch.cat(p_attn_all_head1, 1)
        p_attn_all_head2 = torch.cat(p_attn_all_head2, 1)
    else:
        p_attn_all_head1 = torch.sum(torch.stack(p_attn_all_head1), dim=0)
        p_attn_all_head2 = torch.sum(torch.stack(p_attn_all_head2), dim=0)

    if dropout is not None:
        p_attn_all_head1 = dropout(p_attn_all_head1)
        p_attn_all_head2 = dropout(p_attn_all_head2)
    return  p_attn_all_head1, p_attn_all_head2, scores_all

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, concat = True, ):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()

        if concat:
            assert int(d_model / 2) % h == 0
            # We assume d_v always equals d_k
            out_dim = int(d_model / 2)
            self.linears = clones(nn.Linear(d_model, out_dim), 4)
        else:
            out_dim = int(d_model / 2) * h
            self.linears = clones(nn.Linear(d_model, out_dim), 4)
        self.h = h
        self.d_k = out_dim // h

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat

    def forward(self, query, key, mask=None):
        "Implements Figure 2"
        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)
        
        # 1) Do all the linear projections in batch from d_model => num x d_k x h
        # transpose(1, 2) purpose?
        query, key, query_v, key_v =  [l(x).view(-1, self.h, self.d_k).transpose(1, 0) \
                       for l, x in zip(self.linears, (query, key, query, key))]

        # 2) Apply attention on all the projected vectors in batch.
        x1, x2, self.attn = attention(query, key,  query_v, key_v, mask=mask, dropout=self.dropout,concat = self.concat)

        return x1, x2, self.attn #

class _GraphConvolutionLayer(nn.Module):
    """ graph convolutional layer """
    def __init__(self, dim, dropout):
        super(_GraphConvolutionLayer, self).__init__()

        self.gcn_collect_1 = _GraphConvolutionLayer_Collect(dim,6)
        self.gcn_collect_2 = _GraphConvolutionLayer_Collect(dim,6)
        self.dropout = dropout
        self.bn_obj = nn.BatchNorm1d(int(dim/2), affine=False)
        self.bn_rel = nn.BatchNorm1d(int(dim/2), affine=False)
        self.MultiHeadedAttentions = nn.ModuleList([copy.deepcopy(MultiHeadedAttention(4, dim, dropout,concat=True)) for _ in range(3)])
        self.MultiHeadedAttentions_out = nn.ModuleList([copy.deepcopy(MultiHeadedAttention(6, dim, dropout,concat=False)) for _ in range(3)])
        self.MessagePassingLayer =  nn.ModuleList([copy.deepcopy( _MessagePassingLayer(dim, dropout)) for _ in range(2)])

        self.num_vis = 0
    def forward(self, feat_obj, feat_rel, sub_indexs, obj_indexs, rel_label, classes, rel_classes,rm_obj_labels, vis ):
    # def forward(self, feat_obj, feat_att, feat_rel, map_obj_att, map_obj_obj, map_obj_rel):

        ### todo: rel_label only be used to verify the ability of new graph


        # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
        # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
        # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
        # NOTE exclude one roi feature from another.
        # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
        # cooccurence, such as bus are more frequently appear on the road.
        # assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"


        #size_per_batch = feat_obj.size(0) / batch_size
        rel_classes[0] = 'BG'
        obj_name = [classes[i] for i in rm_obj_labels]
        if  rel_label is not None:
            rel_name = [rel_classes[i] for i in rel_label]
        # =====================================================================
        # ==========================attention adj matrix=======================
        # =====================================================================
        mask = (1-torch.eye(feat_obj.size(0))).cuda()
        sub_obj, obj_sub, map_sub_obj = self.MultiHeadedAttentions[0](feat_obj, feat_obj, mask)
        sub_rel, rel_sub, map_sub_rel = self.MultiHeadedAttentions[1](feat_obj, feat_rel)
        rel_obj, obj_rel, map_rel_obj = self.MultiHeadedAttentions[2](feat_rel, feat_obj)

        obj_feat = (feat_obj + torch.cat((sub_obj, obj_sub), dim=1) + torch.cat((sub_rel, obj_rel), dim=1)) / 3
        rel_feat = (feat_rel + torch.cat((rel_obj, rel_sub), dim=1)) / 2

        #sub_obj, obj_sub, map_sub_obj1 = self.MultiHeadedAttentions_out[0](obj_feat, obj_feat, mask)
        sub_rel, rel_sub, map_sub_rel1 = self.MultiHeadedAttentions_out[1](obj_feat, rel_feat)
        rel_obj, obj_rel, map_rel_obj1 = self.MultiHeadedAttentions_out[2](rel_feat, obj_feat)

        #obj_feat = (feat_obj + torch.cat((sub_obj, obj_sub), dim=1) + torch.cat((sub_rel, obj_rel), dim=1)) / 3
        rel_feat = (feat_rel + torch.cat((rel_obj, rel_sub), dim=1)) / 2

        #
        # if self.num_vis % 3000 == 0 and self.num_vis > 0:
        #     show_heatmap(0, self.num_vis,'map_sub_obj', map_sub_obj, obj_name, obj_name)
        #     show_heatmap(1, self.num_vis,'map_sub_rel', map_sub_rel, rel_name, obj_name)
        #     show_heatmap(2, self.num_vis,'map_rel_obj', map_rel_obj, obj_name, rel_name)
        #
        #     show_heatmap(3, self.num_vis, 'map_sub_obj_1_', map_sub_obj1, obj_name, obj_name)
        #     show_heatmap(4, self.num_vis, 'map_sub_rel_1_', map_sub_rel1, rel_name, obj_name)
        #     show_heatmap(5, self.num_vis, 'map_rel_obj_1_', map_rel_obj1, obj_name, rel_name)
        #     # =====================================================================
        #     # ============================gt adj matrix=======================
        #     # =====================================================================
        #     map_sub_obj_ = feat_obj.data.new(feat_obj.size(0), feat_obj.size(0)).fill_(0.0)
        #     map_sub_rel_ = feat_obj.data.new(feat_obj.size(0), feat_rel.size(0)).fill_(0.0)
        #     map_rel_obj_ = feat_obj.data.new(feat_rel.size(0), feat_obj.size(0)).fill_(0.0)
        #     # !!!!!!!!! sparse matrix will cause the 0 gradient
        #     for i in range(len(obj_indexs)):
        #
        #         sub_index = sub_indexs[i]
        #         obj_index = obj_indexs[i]
        #         if rel_label[i] is not 0:
        #             map_sub_obj_[sub_index, obj_index] = 1
        #
        #         map_sub_rel_[sub_index, i] = 1
        #         map_rel_obj_[i, obj_index] = 1
        #
        #     if rel_label is not None:
        #
        #         rel_unique = np.unique(rel_label.cpu().numpy())
        #         print('rel_unique {}'.format(rel_unique))
        #
        #         for v in rel_unique:
        #             if v is not 0:
        #                 idx = (rel_label == v)
        #                 sub_rel_sum = torch.sum(map_sub_rel_[:, idx], 1)
        #                 map_sub_rel_[:, idx] = sub_rel_sum[:, None]
        #
        #                 rel_obj_sum = torch.sum(map_rel_obj_[idx, :], 0)
        #                 map_rel_obj_[idx, :] = rel_obj_sum[None, :]
        #
        #     # viz the attention map:
        #     show_heatmap(4,  self.num_vis,'map_sub_obj_gt', map_sub_obj_, obj_name, obj_name)
        #     show_heatmap(5,  self.num_vis,'map_sub_rel_gt', map_sub_rel_, rel_name, obj_name)
        #     show_heatmap(6,  self.num_vis,'map_rel_obj_gt', map_rel_obj_, obj_name, rel_name)
        self.num_vis += 1
        return obj_feat, rel_feat
####
'''

only subtration of embedding vectors of sub and obj, all trainset , all testset, wo fine tune
without attention, use the info from gt_rel to cluster same labeled predicate
1 epoch, batch 10, 40 min
loss: cls: 0.0049, rel:about 0.07
model name: vgrel_03.tar
======================predcls============================
R@100: 0.608532
Max_right_num R@100: 45.000000
all_mean R@100: 0.621248
R@50: 0.588207
Max_right_num R@50: 41.000000
all_mean R@50: 0.601753
R@20: 0.562835
Max_right_num R@20: 41.000000
all_mean R@20: 0.572527


with attention, 1 iteration
3 epoch, 60 min
loss: cls: 2.06, rel:about 0.47
======================predcls============================
R@100: 0.55
Max_right_num R@100: 77.000000
all_mean R@100: 0.57
R@50: 0.54
Max_right_num R@50: 77.000000
all_mean R@50: 0.56
R@20: 0.45
Max_right_num R@20: 77.000000
all_mean R@20: 0.47

feat from new paper without dual mask, all trainset , all testset, wo fine tune
9 epoch, batch 10, 40 min
loss: cls: 1, rel:about 0.20
model name: vgrel_04.tar
======================sgcls============================
R@100: 0.325112
Max_right_num R@100: 55.000000
all_mean R@100: 0.337986
R@50: 0.315476
Max_right_num R@50: 55.000000
all_mean R@50: 0.322779
R@20: 0.283603
Max_right_num R@20: 55.000000
all_mean R@20: 0.278737
======================predcls============================
R@100: 0.602202
Max_right_num R@100: 56.000000
all_mean R@100: 0.602039
R@50: 0.571515
Max_right_num R@50: 55.000000
all_mean R@50: 0.557946
R@20: 0.489548
Max_right_num R@20: 55.000000
all_mean R@20: 0.454171


all trainset , all testset, with one epoch fine tune
2.749s/batch, 275.7m/epoch
======================sgcls============================
R@100: 0.291272
Max_right_num R@100: 55.000000
all_mean R@100: 0.304422
R@50: 0.282718
Max_right_num R@50: 55.000000
all_mean R@50: 0.290892
R@20: 0.253956
Max_right_num R@20: 55.000000
all_mean R@20: 0.250280
======================predcls============================
R@100: 0.586275
Max_right_num R@100: 56.000000
all_mean R@100: 0.584330
R@50: 0.553567
Max_right_num R@50: 56.000000
all_mean R@50: 0.538649
R@20: 0.468989
Max_right_num R@20: 55.000000
all_mean R@20: 0.432971
'''
'''
feat from new paper with dual mask, all trainset , all testset, wo fine tune
12 epoch, batch 10, 90 min
loss: cls: 0.9, rel:about 0.14
model name: vgrel_05.tar
======================sgcls============================
R@100: 0.334439
Max_right_num R@100: 55.000000
all_mean R@100: 0.348558
R@50: 0.326340
Max_right_num R@50: 55.000000
all_mean R@50: 0.335490
R@20: 0.296891
Max_right_num R@20: 55.000000
all_mean R@20: 0.292421
======================predcls============================
R@100: 0.609784
Max_right_num R@100: 56.000000
all_mean R@100: 0.613897
R@50: 0.584634
Max_right_num R@50: 56.000000
all_mean R@50: 0.575660
R@20: 0.507676
Max_right_num R@20: 55.000000
all_mean R@20: 0.472737
'''