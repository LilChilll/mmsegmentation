from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
import matplotlib.pyplot as plt
from mmcv.runner import auto_fp16, force_fp32
from mmseg.models.losses import accuracy
from torch.cuda.amp import autocast
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from timm.models.layers import trunc_normal_
import cv2
from .utils import positional_encoding

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

@HEADS.register_module()
class CLIPClassificationHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            base_class,
            both_class,
            use_stages=1,
            align_corners = False,
            **kwargs,
    ):
        super(CLIPClassificationHead, self).__init__(
            in_channels=in_channels, **kwargs)
        self.align_corners = align_corners
        self.image_size = img_size
        self.use_stages = use_stages
        self.base_class = base_class
        self.both_class = both_class


        self.novel_class = self.both_class.copy()
        for i_idx in self.base_class:
            self.novel_class.remove(i_idx)

        delattr(self, 'conv_seg')

        

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs,labels):
        logits = self.forward(inputs)
        
        losses = F.cross_entropy(logits, labels)
        
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training=self_training)

    def forward(self, logits):
        return logits
        
        
    def compute_similarity(image_features, text_features, t=2):
        # image_features [bs, n_i, c]
        # text_features [bs, n_t, c]
        prob_1 = image_features[:, :1, :] @ text_features.permute(0, 2, 1)
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[1], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(b, 1, n_t, c)
        similarity = feats.sum(-1)
        return (similarity/0.07).softmax(-1), prob_1
    

    @force_fp32(apply_to=('logits',))
    def losses(self, logits, gt_semantic_seg, num_classes=None):
        """Compute segmentation loss."""
        label = self.prepare_targets(gt_semantic_seg)
        if isinstance(logits, dict):
            loss=dict()
            loss['loss_logits']=F.cross_entropy(logits, label)
            # loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss
    def prepare_targets(self,gt_semantic_seg):
        targets = []
        for targets_per_image in gt_semantic_seg:
            # gt_cls
            gt_cls = targets_per_image.unique()
            # bg_cls = gt_cls[gt_cls == self.ignore_index]
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            
            targets.append(gt_cls)
        return targets




if __name__ == "__main__":
    # test
    base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    novel_class = [15, 16, 17, 18, 19]
    both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    model = ATMSingleHeadSeg(
        img_size=512,
        in_channels = 512,
        channels=512,
        num_classes = len(base_class),
        base_class=base_class,
        both_class=both_class,
        embed_dims=512,
        use_stages=1,            
        num_layers=3,
        num_heads=8,
        use_proj=False,
        crop_train=False
    )
    model.init_weights()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # global_embedding = torch.randn(4, 512).to(device)
    # visual_embedding=torch.randn(4, 512, 32, 32).to(device)
    # prompt_text_embeddings = torch.randn(4, 15,512).to(device)
    # inputs = [visual_embedding,global_embedding,prompt_text_embeddings]
    pred_masks = torch.randn(4, 15, 512, 512).to(device)
    contour_map  = model.forward_contour(pred_masks)
    # visualize one contour map
    target = contour_map[0].cpu().numpy()
    # transfer 1 to 255
    target = (1-target) * 255
    plt.imsave(fname='contour_map.png',arr=target, cmap='gray')

    print(contour_map.shape)