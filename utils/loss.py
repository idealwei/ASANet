import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        if weight is not None:
            loss = loss * weight
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight):
        return self.weighted(input, target, weight)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(
            0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(
            1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(
            2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(
            n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(
            predict, target, weight=weight, size_average=self.size_average)
        return loss


def eightwayASCLoss(probs, size=1):
    _, _, h, w = probs.size()
    softmax = F.softmax(probs, dim=1)
    p = size
    softmax_pad = F.pad(softmax, [p]*4, mode='replicate')
    affinity_group = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if st_y == size and st_x == size:
                continue
            affinity_paired = torch.sum(
                softmax_pad[:, :, st_y:st_y+h, st_x:st_x+w] * softmax, dim=1)
            affinity_group.append(affinity_paired.unsqueeze(1))
    affinity = torch.cat(affinity_group, dim=1)
    loss = 1.0 - affinity
    return loss.mean()



def fourwayASCLoss(probs, size=1):
    _, _, h, w = probs.size()
    softmax = F.softmax(probs, dim=1)
    p = size
    softmax_pad = F.pad(softmax, [p]*4, mode='replicate')
    affinity_group = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if abs(st_y-st_x) == size:
                affinity_paired = torch.sum(
                    softmax_pad[:, :, st_y:st_y+h, st_x:st_x+w] * softmax, dim=1)
                affinity_group.append(affinity_paired.unsqueeze(1))
    affinity = torch.cat(affinity_group, dim=1)
    loss = 1.0 - affinity
    return loss.mean()