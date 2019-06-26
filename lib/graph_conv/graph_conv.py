import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.graph_conv.graph_conv_unit import _GraphConvolutionLayer_Collect


class _GraphConvolutionLayer(nn.Module):
    """ graph convolutional layer """
    def __init__(self, dim, dropout):
        super(_GraphConvolutionLayer, self).__init__()

        self.gcn_collect_1 = _GraphConvolutionLayer_Collect(dim)
        self.gcn_collect_2 = _GraphConvolutionLayer_Collect(dim)
        self.dropout = dropout
        self.bn_obj = nn.BatchNorm1d(4096, affine=False)
        self.bn_rel = nn.BatchNorm1d(4096, affine=False)

    def forward(self, feat_obj, feat_rel, obj_indexs, sub_indexs ):
    # def forward(self, feat_obj, feat_att, feat_rel, map_obj_att, map_obj_obj, map_obj_rel):


        # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
        # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
        # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
        # NOTE exclude one roi feature from another.
        # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
        # cooccurence, such as bus are more frequently appear on the road.
        # assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"

        #size_per_batch = feat_obj.size(0) / batch_size

        map_sub_rel = feat_obj.data.new(feat_obj.size(0), feat_rel.size(0)).fill_(0.0)

        map_rel_obj = feat_obj.data.new(feat_rel.size(0), feat_obj.size(0)).fill_(0.0)

        #eye_mat = torch.eye(int(size_per_batch)).type_as(feat_obj.data)
        #!!!!!!!!! sparse matrix will cause the 0 gradient
        for i in range(len(obj_indexs)):
            obj_index = obj_indexs[i]
            sub_index = sub_indexs[i]
            map_sub_rel[sub_index, i] = 1
            map_rel_obj[i, obj_index] = 1

        source_sub = self.gcn_collect_1(feat_rel, map_sub_rel, 0)
        source_rel = self.gcn_collect_1(feat_obj, map_rel_obj, 1)

        source_sub = self.bn_obj(source_sub)
        _source_sub_updated =F.tanh(source_sub)#self.gcn_update(feat_obj, source2obj_all, torch.tensor(0))
        source_sub_updated = F.dropout(_source_sub_updated,self.dropout)


        source_rel = self.bn_rel(source_rel)
        _source_rel_updated =F.tanh(source_rel)#self.gcn_update(feat_rel, source2rel_all, torch.tensor(1))
        source_rel_updated = F.dropout(_source_rel_updated, self.dropout)

        #add secend gcn layer to updata nodes features
        source_sub = self.gcn_collect_2(source_rel_updated, map_sub_rel, 0)
        source_rel = self.gcn_collect_2(source_sub_updated, map_rel_obj, 1)

        source_sub = self.bn_obj(source_sub)
        source_rel = self.bn_rel(source_rel)
        feat_obj_updated = F.tanh(feat_obj + source_sub)  # self.gcn_update(feat_obj, source2obj_all, torch.tensor(0))
        feat_rel_updated = F.tanh(feat_rel + source_rel)  #
        # return feat_obj_updated, feat_att_updated, feat_rel_updated
        return feat_obj_updated, feat_rel_updated

