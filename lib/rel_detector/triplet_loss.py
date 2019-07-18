from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings_x, embeddings_y, labels):
        if self.cpu:
            embeddings_x = embeddings_x.cpu()
            embeddings_y = embeddings_y.cpu()
        distance_matrix = pdist(embeddings_x, embeddings_y)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets_y = []
        triplets_x = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                #print('len(label_indices) < 2')
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = label_indices  # All anchor-positive pairs
            #anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives, anchor_positives]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values_x = ap_distance - distance_matrix[anchor_positive, torch.LongTensor(negative_indices)] + self.margin # size: 1xlen(negative_indices)
                loss_values_y = ap_distance - distance_matrix[torch.LongTensor(negative_indices), anchor_positive] + self.margin # size: len(negative_indices)x1

                loss_values_x = loss_values_x.data.cpu().numpy()
                loss_values_y = loss_values_y.data.cpu().numpy()
                hard_negative_x = self.negative_selection_fn(loss_values_x)
                hard_negative_y = self.negative_selection_fn(loss_values_y)
                if hard_negative_x is not None :
                    hard_negative_x = negative_indices[hard_negative_x]
                    triplets_x.append([anchor_positive, anchor_positive, hard_negative_x])

                if hard_negative_y is not None:
                    hard_negative_y = negative_indices[hard_negative_y]
                    triplets_y.append([anchor_positive, anchor_positive, hard_negative_y])

        if len(triplets_x) == 0:
            triplets_x.append([anchor_positives[0], anchor_positives[0], negative_indices[0]])
        if len(triplets_y) == 0:
            triplets_y.append([anchor_positives[0], anchor_positives[0], negative_indices[0]])

        triplets_x, triplets_y  = np.array(triplets_x),  np.array(triplets_y)

        return torch.LongTensor(triplets_x), torch.LongTensor(triplets_y)

def pdist(vectors_x, vectors_y):
    '''

    :param vectors:
    :return: ||x-y||^2 = x^2 -2xy + y^2 NxN
    '''
    Nx = vectors_x.size(0)
    Ny = vectors_y.size(0)
    distance_matrix = -2 * vectors_x.mm(torch.t(vectors_y)) + vectors_x.pow(2).sum(dim=1).view(-1, 1).repeat(1, Ny) + vectors_y.pow(2).sum(
        dim=1).view(1, -1).repeat(Nx, 1)
    return distance_matrix

def semihard_negative(loss_values, margin):
    #loss_values: 1xlen(negative_indices)
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                   negative_selection_fn=lambda x: semihard_negative(x, margin), cpu=cpu)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings_x, embeddings_y, labels):

        triplets_x, triplets_y = self.triplet_selector.get_triplets(embeddings_x, embeddings_y, labels)

        if embeddings_x.is_cuda:
            triplets_x, triplets_y = triplets_x.cuda(), triplets_y.cuda()

        ap_distances_x = (embeddings_x[triplets_x[:, 0]] - embeddings_y[triplets_x[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances_x = (embeddings_x[triplets_x[:, 0]] - embeddings_y[triplets_x[:, 2]]).pow(2).sum(1)  # .pow(.5)

        ap_distances_y = (embeddings_y[triplets_y[:, 0]] - embeddings_x[triplets_y[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances_y = (embeddings_y[triplets_y[:, 0]] - embeddings_x[triplets_y[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances_x + self.margin - an_distances_x).mean() + F.relu(ap_distances_y + self.margin - an_distances_y).mean()

        return losses
'''
use triplet loss, distance to detemine similarity, margin 1,  choice one  negative sample for each anchor-positive pair
 7 epoch, still going up, total loss:5, time 90m
======================sgcls============================
R@100: 0.248718
Max_right_num R@100: 42.000000
all_mean R@100: 0.258809
R@50: 0.205740
Max_right_num R@50: 40.000000
all_mean R@50: 0.196591
R@20: 0.142403
Max_right_num R@20: 22.000000
all_mean R@20: 0.122602
======================predcls============================
R@100: 0.479879
Max_right_num R@100: 150.000000
all_mean R@100: 0.439578
R@50: 0.382919
Max_right_num R@50: 150.000000
all_mean R@50: 0.329282
R@20: 0.260719
Max_right_num R@20: 150.000000
all_mean R@20: 0.207827
'''