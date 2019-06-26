"""
Training script 4 Detection
"""
#from dataloaders.mscoco import CocoDetection, CocoDataLoader

from dataloaders.visual_genome_1 import VGDataLoader, VG

import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from textwrap import wrap
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from lib.model.utils.config import cfg
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.model.object_dectector.old_obj_detector import fasterRCNN
from lib.model.scenegraph_detector.scenegraph_detector import graphRCNN
from lib.model.utils import network

from lib.model.spinn.parser_predictor import Parser
from lib.model.spinn.util import get_args
from tensorboardX import SummaryWriter
logger = SummaryWriter("logs/obj_detector_rpn")


start_epoch = -1
# if conf.ckpt is not None:
#     ckpt = torch.load(conf.ckpt)
#     if optimistic_restore(detector, ckpt['state_dict']):
#         start_epoch = ckpt['epoch']

def plot_grad_flow(named_parameters, figure_idx):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.figure(figure_idx)
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) :
            # if 'gcn_collect' in n:
            #     print('wait check gcn')
            try:
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
                layers.append(n)
            except:
                pass
                #print(n)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), ['\n'.join(wrap(ln,30)) for ln in layers], rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=1)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([mlines.Line2D([0], [0], color="c", lw=4),
                mlines.Line2D([0], [0], color="b", lw=4),
                mlines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('gradient.jpeg')
    plt.show(block=False)
    plt.pause(0.01)



def train_batch(batch):

    # obj detection part
    with torch.no_grad():
        if using_spinn:
            spinn_res = spinn(batch.phrases)
        else:
            spinn_res = batch.phrases
    res_obj = obj_detector(batch.imgs,batch.im_sizes,batch.gt_boxes, use_gt_boxes=True)
    rois, bbox_pred, cls_prob,  rois_label, rpn_label, pooled_feat = res_obj
    # scene detection part
    res_scene = scene_detector(rois.data, bbox_pred, batch.im_sizes, cls_prob, pooled_feat, spinn_res, rois_label, batch.gt_boxes, batch.gt_rels)
    rois, obj_cls_prob, rel_cls_prob, loss_obj_cls, relpn_loss, grcnn_loss, relpn_eval = res_scene
    loss_scene =  grcnn_loss.mean()
    # relpn_eval, relpn_loss = res_scene
    # loss_scene = relpn_loss.mean() #+ grcnn_loss.mean()
    # grcnn_loss = 0
    loss_scene.backward()
    # plot_grad_flow(obj_detector.named_parameters(),figure_idx=0)
    # plot_grad_flow(scene_detector.named_parameters(),figure_idx=1)

    return loss_scene, relpn_loss, relpn_eval, grcnn_loss, loss_obj_cls

def train_epoch(epoch_num, max_acc = 0):

    obj_detector.eval()
    scene_detector.train()
    accumulation_steps = 10
    start = time.time()
    s ,e = 0,0
    for b, batch in enumerate(train_loader):
        s = time.time()
        #print('time for data loading {}'.format(s-e))
        loss_scene, relpn_loss, relpn_eval, grcnn_loss, loss_obj_cls = train_batch(batch)
        #print('step {}'.format(b))
        #if b % conf.print_interval == 0 and b >= conf.print_interval:
        e = time.time()
        #print('time for NN {}'.format(e - s))
        if b % accumulation_steps == 0:
            network.clip_gradient(scene_detector, 10.)
            optimizer.step()
            scene_detector.zero_grad()  ##
            optimizer.zero_grad()


        if b % 10 == 0:
            ## todo: what is the purpose of mn
            #mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d} b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print('loss is {}'.format(loss_scene))
            print('relpn_acc is recall_fg,recall_bg,acc_fg {}'.format(relpn_eval))
            #print('fg/bg is {}/{} '.format(fg_cnt,bg_cnt))
            print('-----------', flush=True)
            info = {
                'loss_scene': loss_scene,
                'relpn_loss': relpn_loss,
                'grcnn_loss': grcnn_loss,
                'loss_obj_cls': loss_obj_cls,
                'relpn_recall_fg': relpn_eval[0],
                'relpn_recall_bg': relpn_eval[1],
                'relpn_acc_fg': relpn_eval[2]

            }
            logger.add_scalars("logs_{}/losses".format('rel'), info,
                               (epoch_num ) * (len(train_loader)) + b)
            loss_acc = 1/loss_scene
            if loss_acc > max_acc:
                max_acc = loss_acc
                torch.save({
                    'epoch': epoch,
                    'rel_state_dict': scene_detector.state_dict(),
                    'obj_state_dict': obj_detector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(conf.save_dir, '{}-{}-{}-gcn-{}.tar'.format('vg', epoch_num, b, max_acc)))


    #plot_grad_flow(scene_detector.named_parameters())

    return max_acc

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    using_spinn = False
    MODEL = 'rel'
    TRAIN_REL = True if MODEL == 'rel' else False
    # cudnn.benchmark = True
    conf = ModelConfig()

    ##########################################
    # -------------data loader
    ##########################################
    train, val = VG.splits(('train','val'),num_im=10000, num_val_im=200, filter_empty=True)
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus, mode=MODEL)
    ########################################
    # -------------spinn model
    ##########################################
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
        spinn.eval()
    ##########################################
    #-------------obj detector
    ##########################################
    n_obj_classes, n_rel_classes = len(train.ind_to_classes), len(train.ind_to_predicates)
    obj_detector = fasterRCNN(n_obj_classes, train_rel=TRAIN_REL)
    obj_detector.create_architecture()
    obj_detector.cuda()

    #Note: if you're doing the stanford setup, you'll need to change this to freeze the lower layers
    load_name = '/home/bai/MA/grcnn_based_onMotifs/data/pretrained_models/vg-5-120-gcn-1.3693231344223022.tar'
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)

    obj_detector.load_state_dict(checkpoint['obi_state_dict'])

    for key, value in dict(obj_detector.named_parameters()).items():
        if ('RCNN_top' in key) :  #('base' in key) or  or('RELPN' in key): #or ('GRCNN' in key)
            value.requires_grad = True
        else:
            value.requires_grad = False
    ##########################################
    # ------------rel detector
    ##########################################
    scene_detector = graphRCNN(n_obj_classes, n_rel_classes,pooled_feat_dim=4096)
    scene_detector.create_architecture()
    # print(obj_detector)
    scene_detector.cuda()

    use_pretrained_relpn = 1
    if use_pretrained_relpn:
        dict_new = scene_detector.state_dict().copy()
        dict_trained = checkpoint["rel_state_dict"]
        trained_list = list(dict_trained)
        print("trained_dicht size {}".format(len(trained_list)))

        for i in range(len(trained_list)):
            if trained_list[i] in dict_new :
                #and "RCNN_base" not in trained_list[i]
                dict_new[trained_list[i]] = dict_trained[trained_list[i]]
        scene_detector.load_state_dict(dict_new)
        del dict_trained
        del dict_new


    for key, value in dict(scene_detector.named_parameters()).items():
        if ('RELPN' in key) or ('encoder' in key):  #('base' in key) or  or('RELPN' in key): #or ('GRCNN' in key)
            value.requires_grad = False
        else:
            value.requires_grad = True

    ########################################
    # ------------optimizier
    ##########################################
    scene_detector_parameters = [p for p in scene_detector.parameters() if p.requires_grad]
    obj_detector_parameters = [p for p in obj_detector.parameters() if p.requires_grad]
    optimizer = optim.SGD(scene_detector_parameters+obj_detector_parameters,
                          weight_decay=conf.l2, lr=conf.lr * conf.num_gpus * conf.batch_size, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
    #                               verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)
    optimizer.load_state_dict(checkpoint["optimizer"])
    del checkpoint

    print("Training starts now!")
    max_acc = 0
    for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
        if not os.path.exists(conf.save_dir):
            os.mkdir(conf.save_dir)
        max_acc = train_epoch(epoch, max_acc)
        # print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
        # mAp = val_epoch()
        # scheduler.step(mAp)
        #
        torch.save({
            'epoch': epoch,
            'rel_state_dict': scene_detector.state_dict(),
            'obj_state_dict': obj_detector.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}-gcn.tar'.format('vg', epoch)))
