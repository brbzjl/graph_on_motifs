"""
Training script 4 Detection
"""
# from dataloaders.mscoco import CocoDetection, CocoDataLoader

from dataloaders.visual_genome_1 import VGDataLoader, VG
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.model.roi_layers import nms
from lib.model.object_dectector.old_obj_detector import fasterRCNN
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.model.utils.network import save_net, load_net, vis_detections, vis_det_att, vis_relations, vis_gt_relations, eval_relations_recall
from lib.model.utils import network
from lib.model.scenegraph_detector.scenegraph_detector import graphRCNN

from lib.model.spinn.parser_predictor import Parser
from lib.model.spinn.util import get_args

start_epoch = -1


def draw_box(draw, class_name, dets, thresh=0.7):
    if '-GT' in class_name:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)
    for i in range(dets.shape[0]):
        box = tuple([float(b) for b in dets[i, :4]])
        score = dets[i, -1]
        if score > thresh:
            # draw the fucking box
            draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
            draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
            draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
            draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)
            w, h = draw.textsize(class_name)

            x1text = box[0]
            y1text = max(box[1] - h, 0)
            x2text = min(x1text + w, draw.im.size[0])
            y2text = y1text + h

            draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
            draw.text((x1text, y1text), class_name, fill='black')
            draw.text((x1text + w, y1text), str(score), fill='black')

    return draw



top_Ns = [20,50,100]#, 100, 500]

def val_epoch():

    obj_detector.eval()
    scene_detector.eval()
    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    vr = []
    rel_cnt_all = 0
    rel_correct_cnt_all = torch.zeros(len(top_Ns)).int()

    for val_b, batch in enumerate(test_loader):
        rel_cnt, rel_corrent_cnt = val_batch(val_b, batch)

        rel_cnt_all += (rel_cnt)
        rel_correct_cnt_all += (rel_corrent_cnt)

        print('rel_recall: {:d}/top-50: {:d} top-100: {:d}  \r' \
              .format(rel_cnt_all, rel_correct_cnt_all[0], rel_correct_cnt_all[1]))
        if rel_cnt_all is not 0:
            for itop in range(len(top_Ns)):
                print("rel recall@%d: %0.4f\r" % (top_Ns[itop], float(rel_correct_cnt_all[itop]) / float(rel_cnt_all)))

def val_batch(batch_num, batch):

    with torch.no_grad():
        if using_spinn:
            spinn_res = spinn(batch.phrases)
        else:
            spinn_res = batch.phrases
        result = obj_detector(batch.imgs, batch.im_sizes, batch.gt_boxes, use_gt_boxes = True)
        rois, bbox_pred, cls_prob, rois_label, rpn_label, pooled_feat = result

        # scene detection part
        res_scene = scene_detector(rois.data, bbox_pred, batch.im_sizes, cls_prob, pooled_feat, spinn_res, rois_label,
                                   batch.gt_boxes, batch.gt_rels, use_gt_boxes = True)
    rois, roi_pair_proposals, obj_cls_prob, rel_cls_prob, roi_rel_pairs_score, _, _ = res_scene

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    scale = batch.im_sizes[0][0][2]
    pred_boxes = boxes / scale
    pred_boxes = pred_boxes.squeeze()
    use_gt_boxes = True
    if not use_gt_boxes:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

            box_deltas = box_deltas.view(1, -1, 4)
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, batch.im_sizes, 1)

    scores = scores.squeeze()

    theimg = cv2.imread(batch.im_fn[0])

    # im = theimg.copy()
    # for n in range(len(batch.gt_classes[0])):
    #     box = batch.gt_boxes[0, n, :] / batch.im_sizes[0][0][2]  # batch.im_sizes[0][0][2]
    #     box = box.cpu().numpy()
    #     bbox = tuple(int(np.round(x)) for x in box[:4])
    #     cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
    #     class_name = val.ind_to_classes[batch.gt_classes[0][n]]
    #     cv2.putText(im, '%s:' % (class_name), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
    #                 1.0, (0, 0, 255), thickness=1)
    # plt.figure(0)
    # plt.imshow(im)
    # plt.show()
    # plt.pause(0.1)
    # index_fg = (rpn_label == 1).nonzero().squeeze()
    # for n in range(len(index_fg)):
    #     rois_boxes = boxes[0,index_fg[n],:4].cpu().numpy()
    #     rois_boxes /= batch.im_sizes[0][0][2]
    #     rois_boxes = tuple(int(np.round(x)) for x in rois_boxes)
    #     if (rois_boxes[0] or rois_boxes[2]) < 0 or (rois_boxes[0] or rois_boxes[2]) > im.shape[1]\
    #         or (rois_boxes[1] or rois_boxes[3]) < 0 or (rois_boxes[1] or rois_boxes[3]) > im.shape[0]:
    #         print('out of boundary')
    #     cv2.rectangle(im, rois_boxes[0:2], rois_boxes[2:4], (204, 0, 0), 2)
    #
    #     plt.imshow(im)
    #     plt.show()
    #     plt.pause(0.1)
    # theimg = cv2.resize(theimg,(int(im_scale * theimg.shape[0]), int(im_scale * theimg.shape[1])))
    im2show = np.copy(theimg)
    # draw2 = ImageDraw.Draw(theimg2)
    ## ============================================================================================================================
    # visualilze rois detection from Faster RCNN
    ## ============================================================================================================================
    # v, class_index = torch.max(scores, 1)
    # for j in range(1, len(val.ind_to_classes)):
    #     bboxs_index = (class_index == j).nonzero().squeeze()
    #     if bboxs_index.numel() > 0:
    #         cls_boxes = pred_boxes[bboxs_index]
    #         cls_scores = v[bboxs_index]
    #         _, order = torch.sort(cls_scores, 0, True)
    #
    #         if (len(cls_scores.size()) == 0):
    #             # print(cls_scores)
    #             cls_scores = cls_scores.unsqueeze(0)
    #             cls_boxes = cls_boxes.unsqueeze(0)
    #             continue
    #
    #         cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
    #         cls_dets = cls_dets[order]
    #         # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
    #         keep = nms(cls_boxes[order, :], cls_scores[order], 0.5)  # )cfg.TEST.NMS
    #         cls_dets = cls_dets[keep.view(-1).long()]
    #
    #         im2show_ = vis_detections(im2show, val.ind_to_classes[j], cls_dets.cpu().numpy(), 0.1)
    #         plt.figure(3)
    #         plt.clf()
    #         plt.imshow(im2show_)
    #         plt.title('vis_pre_detections')
    #         plt.pause(0.1)
    # theimg2.show('obj_detection')
    im = theimg.copy()
    gt_boxes = batch.gt_boxes.data[0]
    rel_cnt, rel_corrent_cnt, gt_rel_rois, gt_rel_labels = \
        eval_relations_recall(im, gt_boxes, scale, batch.gt_rels.data[0], pred_boxes.data, obj_cls_prob.data[0],roi_pair_proposals.view(-1, 2),
                               rel_cls_prob.data[0], top_Ns, roi_rel_pairs_score, val,vis=False)

    return rel_cnt, rel_corrent_cnt
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # cudnn.benchmark = True
    conf = ModelConfig()
    using_spinn = False
    # conf.val_size
    conf.batch_size = 1

    data_val_test = VG.splits(('val','test'),num_im=1000, num_val_im=1, filter_empty=True)
    val, test= data_val_test[0], data_val_test[1]
    test_loader, val_loader = VGDataLoader.splits(test, val, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers,
                                                  num_gpus=conf.num_gpus, mode='rel')

    if using_spinn:
        args_spinn = get_args()
        # torch.cuda.set_device(0)
        # --------------configure of setting-------------
        config_spinn = args_spinn
        config_spinn.n_embed = 36990  # len(inputs.vocab) 36990
        config_spinn.lr = 2e-3  # 3e-4
        config_spinn.lr_decay_by = 0.75
        config_spinn.lr_decay_every = 1  # 0.6
        config_spinn.regularization = 0  # 3e-6
        config_spinn.mlp_dropout = 0.07
        config_spinn.embed_dropout = 0.08  # 0.17
        config_spinn.n_mlp_layers = 2
        config_spinn.d_tracker = 64
        config_spinn.d_mlp = 1024
        config_spinn.d_hidden = 300
        config_spinn.d_embed = 300
        config_spinn.d_proj = 600
        config_spinn.is_training = True
        torch.backends.cudnn.enabled = False

        spinn = Parser(config_spinn)
        spinn.cuda()
        model_path = os.path.join('/home/bai/MA/my_graph_rcnn-master/lib/model/spinn/', args_spinn.save_path, '5.pt')
        # model_name = model_path + '_1.pt'
        spinn.load_state_dict(torch.load(model_path))
        spinn.load_state_dict(torch.load(model_path))
        for key, value in dict(spinn.named_parameters()).items():
            value.requires_grad = False

    load_name = '/home/bai/MA/grcnn_based_onMotifs/data/pretrained_models/vg-5-120-gcn-1.3693231344223022.tar'
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)

    n_obj_classes, n_rel_classes = len(test.ind_to_classes), len(test.ind_to_predicates)

    obj_detector = fasterRCNN(n_obj_classes)
    obj_detector.create_architecture()
    obj_detector.cuda()
    obj_detector.load_state_dict(checkpoint['obj_state_dict'])


    scene_detector = graphRCNN(n_obj_classes, n_rel_classes, pooled_feat_dim=4096)
    scene_detector.create_architecture()
    scene_detector.cuda()

    checkpoint = torch.load(load_name)
    scene_detector.load_state_dict(checkpoint['rel_state_dict'])

    del checkpoint
    print("teste starts now!")
    mAp = val_epoch()
    # scheduler.step(mAp)
