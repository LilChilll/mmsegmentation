# Copyright (c) OpenMMLab. All rights reserved.
import functools
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from timm.models.layers import trunc_normal_

from scipy.ndimage.morphology import distance_transform_edt
def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = mmcv.load(class_weight)

    return class_weight


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

def mask_to_onehot(mask, gt_cls):
    # mask = mask.cpu()
    _mask = [mask == i for i in gt_cls]
    return np.array(_mask).astype(np.uint8)

def onehot_to_binary_edges(mask, radius, gt_cls):
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    edgemap = np.zeros(mask.shape[1:])
    for i,cls in enumerate(gt_cls):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class InverseNet(nn.Module):
    def __init__(self,pretrained=None):
        super(InverseNet, self).__init__()
        # Regressor for the 3 * 2 affine matrix
        self.fc = nn.Sequential(
            nn.Linear(224*224*2, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )
        # self.apply(self._init_weights)
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self, pretrained=None):
        # pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))
        pretrained = pretrained if pretrained else self.pretrained
        if pretrained:
            pretrained_dict = torch.load(pretrained)
            
            model_dict = self.state_dict()
            updated_model_dict = {}
            for k_model, v_model in model_dict.items():
                if k_model.startswith('model') or k_model.startswith('module'):
                    k_updated = '.'.join(k_model.split('.')[1:])
                    updated_model_dict[k_updated] = k_model
                else:
                    updated_model_dict[k_model] = k_model

            updated_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('model') or k.startswith('modules'):
                    k = '.'.join(k.split('.')[1:])
                if k in updated_model_dict.keys() and model_dict[k].shape == v.shape:
                    updated_pretrained_dict[updated_model_dict[k]] = v

            model_dict.update(updated_pretrained_dict)
            self.load_state_dict(model_dict)
            print("InverseNet weights loaded.")
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
            print("Warning: InverseNet weights not loaded.")

        

    def forward(self, x1, x2):
        # Perform the usual forward pass
        x = torch.cat((x1.view(-1, 224*224),x2.view(-1, 224*224)), dim=1)
        out=self.fc(x)
        return x1, x2, out