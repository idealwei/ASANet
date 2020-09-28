import sys
import os
import os.path as osp
import random
from pprint import pprint
import timeit

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from networks.deeplab import Deeplab_Res101
from networks.discriminator import EightwayASADiscriminator
from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss
from datasets.gta5_dataset import GTA5DataSet
from datasets.cityscapes_dataset import cityscapesDataSet
from options import gta5asa_opt
from tensorboardX import SummaryWriter
args = gta5asa_opt.get_arguments()


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda()
    criterion = CrossEntropy2d().cuda()
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():
    """Create the model and start the training."""
    save_dir = osp.join(args.snapshot_dir, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    cudnn.enabled = True
    # Create network
    if args.backbone == 'resnet':
        model = Deeplab_Res101(num_classes=args.num_classes)
    if args.resume:
        print("Resuming from ==>>", args.resume)
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
    else:
        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)
    model.train()
    model.cuda()
    cudnn.benchmark = True

    # init D
    model_D = EightwayASADiscriminator(num_classes=args.num_classes)
    model_D.train()
    model_D.cuda()

    print(model_D)
    pprint(vars(args))
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    img_size=input_size),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.batch_size,
                                                     img_size=input_size_target,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)
    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(
    ), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    weight_bce_loss = WeightedBCEWithLogitsLoss()
    interp = nn.Upsample(
        size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(
        input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    source_label = 0
    target_label = 1
    start = timeit.default_timer()
    loss_seg_value = 0
    loss_adv_target_value = 0
    loss_D_value = 0
    for i_iter in range(args.num_steps):
        damping = (1 - i_iter/args.num_steps)
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train G
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        # train with source
        _, batch = next(trainloader_iter)
        src_img, labels, _, _ = batch
        src_img = Variable(src_img).cuda()
        pred = model(src_img)
        pred = interp(pred)
        loss_seg = loss_calc(pred, labels)
        loss_seg.backward()
        loss_seg_value += loss_seg.item()

        # train with target
        _, batch = next(targetloader_iter)
        tar_img, _, _, _ = batch
        tar_img = Variable(tar_img).cuda()
        pred_target = model(tar_img)
        pred_target = interp_target(pred_target)
        D_out = model_D(F.softmax(pred_target, dim=1))
        loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
        loss_adv = loss_adv_target * args.lambda_adv_target1 * damping
        loss_adv.backward()
        loss_adv_target_value += loss_adv_target.item()
        # train D
        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True
        # train with source
        pred = pred.detach()
        D_out = model_D(F.softmax(pred, dim=1))
        loss_D1 = bce_loss(D_out, torch.FloatTensor(
            D_out.data.size()).fill_(source_label).cuda())
        loss_D1 = loss_D1 / 2
        loss_D1.backward()
        loss_D_value += loss_D1.item()
        # train with target
        pred_target = pred_target.detach()
        D_out1 = model_D(F.softmax(pred_target, dim=1))
        loss_D1 = bce_loss(D_out1, torch.FloatTensor(
            D_out1.data.size()).fill_(target_label).cuda())
        loss_D1 = loss_D1 / 2
        loss_D1.backward()
        loss_D_value += loss_D1.item()
        optimizer.step()
        optimizer_D.step()
        current = timeit.default_timer()

        if i_iter % 50 == 0:
            print(
                'iter = {0:6d}/{1:6d}, loss_seg1 = {2:.3f}  loss_adv1 = {3:.3f}, loss_D1 = {4:.3f} ({5:.3f}/iter)'.format(
                    i_iter, args.num_steps, loss_seg_value/50,  loss_adv_target_value/50, loss_D_value/50, (current - start) / (i_iter+1))
            )
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalars("Loss", {
                               "Seg": loss_seg_value, "Adv": loss_adv_target_value, "Disc": loss_D_value}, i_iter)
            loss_seg_value = 0
            loss_adv_target_value = 0
            loss_D_value = 0

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(
                save_dir, 'GTA5KLASA_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(
                save_dir, 'GTA5KLASA_' + str(i_iter) + '_D.pth'))

        if (i_iter+1) >= args.num_steps_stop:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(
                save_dir, 'GTA5KLASA_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(
                save_dir, 'GTA5KLASA_' + str(args.num_steps_stop) + '_D.pth'))


if __name__ == '__main__':
    main()
