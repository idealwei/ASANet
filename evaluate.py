import os
import os.path as osp
from collections import OrderedDict
import sys
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from networks.deeplab import Deeplab_Res101
from networks.deeplab_vgg import DeeplabVGG
from networks.fcn8s import VGG16_FCN8s
from datasets.cityscapes_dataset import cityscapesDataSet
import timeit
BACKBONE = 'resnet'
IGNORE_LABEL = 255
NUM_CLASSES = 19
LOG_DIR = './logs'
DATA_DIRECTORY = '/share/zhouwei/datasets/cityscapes'
DATA_LIST_PATH = './datasets/cityscapes_list/val.txt'
RESTORE_FROM = 'pretrained/'
# imageNet mean

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall',
    'fence', 'pole', 'light', 'sign',
    'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train',
    'motocycle', 'bicycle'
]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help=".")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--img-width", type=int, default=1024)
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--list-path", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--is-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--split", type=str, default='val',
                        help="Whether to randomly mirror the inputs during the training.")
    return parser.parse_args()


def colorize_mask(mask):
    # mask: numpy array of the mask
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def scale_image(image, scale):
    _, _, h, w = image.size()
    scale_h = int(h*scale)
    scale_w = int(w*scale)
    image = F.interpolate(image, size=(scale_h, scale_w),
                          mode='bilinear', align_corners=True)
    return image


def predict(net, image, output_size, is_mirror=True, scales=[1]):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    outputs = []
    if is_mirror:
        # image_rev = image[:, :, :, ::-1]
        image_rev = torch.flip(image, dims=[3])
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image
                image_rev_scale = image_rev
            image_scale = torch.cat([image_scale, image_rev_scale], dim=0)
            with torch.no_grad():
                prediction = net(image_scale.cuda())
                prediction = interp(prediction)
            prediction_rev = prediction[1, :, :, :].unsqueeze(0)
            prediction_rev = torch.flip(prediction_rev, dims=[3])
            prediction = prediction[0, :, :, :].unsqueeze(0)
            prediction = (prediction + prediction_rev)*0.5
            outputs.append(prediction)
        outputs = torch.cat(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)
        outputs = outputs.permute(1, 2, 0)
    else:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
            else:
                image_scale = image
            with torch.no_grad():
                prediction = net(image_scale.cuda())
            prediction = interp(prediction)
            outputs.append(prediction)
        outputs = torch.cat(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)
        outputs = outputs.permute(1, 2, 0)
    probs, pred = torch.max(outputs, dim=2)
    pred = pred.cpu().data.numpy()
    return pred


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred_label] = label_count[cur_index]

    return confusion_matrix


def main():
    """Create the model and start the evaluation process."""
    start = timeit.default_timer()
    args = get_arguments()
    pprint(vars(args))
    print("=======================================")
    print("Use weights:", args.restore_from)
    h, w = args.img_height, args.img_width
    if args.backbone == 'resnet':
        model = Deeplab_Res101(num_classes=args.num_classes)
    elif args.backbone == 'vgg':
        model = DeeplabVGG(num_classes=args.num_classes)
    else:
        model = VGG16_FCN8s(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()

    test_loader = cityscapesDataSet(root=args.data_dir, list_path=args.list_path, set=args.split, img_size=(
        2048, 1024), norm=False, ignore_label=args.ignore_label)
    test_loader = DataLoader(test_loader, batch_size=1,
                             shuffle=False, num_workers=4)
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    pbar = tqdm(enumerate(test_loader))
    print("len loader:", len(test_loader))
    confusion_cost = 0
    for index, batch in pbar:
        if index % 100 == 0:
            print('%d processd' % (index))
        image, label, size, name = batch
        image = F.interpolate(image, size=(
            h, w), mode='bilinear', align_corners=True)
        pred = predict(model, image.cuda(), (1024, 2048),
                       is_mirror=args.is_mirror, scales=[1])
        # seg_pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
        seg_pred = np.asarray(pred, dtype=np.uint8)
        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_start = timeit.default_timer()
        confusion_matrix += get_confusion_matrix(
            seg_gt, seg_pred, args.num_classes)
        confusion_cost += timeit.default_timer() - confusion_start
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_accuracy = tp.sum()/pos.sum()
    mean_accuracy = (tp/np.maximum(1.0, pos)).mean()
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU_19 = round(np.nanmean(IU_array) * 100, 2)
    mean_IU_16 = round(np.mean(
        IU_array[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)
    mean_IU_13 = round(
        np.mean(IU_array[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %s \n' % str(mean_IU_19))
    print('16-class IU: %s \n' % str(mean_IU_16))
    print('13-class IU: %s \n' % str(mean_IU_13))
    class_iu_dict = {}
    for class_name, IU in zip(CLASS_NAMES, IU_array):
        print(class_name, str(round(IU*100, 2)))
        class_iu_dict[class_name] = str(round(IU*100, 2))
    rst_dict = {
        "Weights": args.restore_from,
        "Mean Accuracy": mean_accuracy,
        "Mean IoU": str(mean_IU_19),
        "16-class IoU": str(mean_IU_16),
        "13-class IoU": str(mean_IU_13),
        "class IoU": class_iu_dict,
    }
    json_obj = json.dumps(
        rst_dict,
        sort_keys=False,
        indent=4,
        separators=(',', ':')
    )
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(osp.join(args.log_dir, 'result512x1024.txt'), 'a+') as fobj:
        pprint(json_obj, fobj)
    end = timeit.default_timer()
    print("Total time:", end-start, 'seconds')
    print("Cofusion cost:", confusion_cost)


if __name__ == '__main__':
    main()
