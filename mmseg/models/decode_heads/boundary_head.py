from torch import nn
import numpy as np
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import math
from mmcv.cnn import build_norm_layer
import torch
from mmcv.runner import force_fp32
@HEADS.register_module()
class BoundaryDecodeHead(BaseDecodeHead):
    def __init__(self, 
            in_channels,
            num_classes=1,
            #TODO add norm config in config file
            **kwargs):
        super(BoundaryDecodeHead, self).__init__(
            in_channels=in_channels,num_classes=num_classes, **kwargs)
        self.edgeocr_cls_head = nn.Conv2d(
                in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                bias=True)
        _,self.edge_norm = build_norm_layer(self.norm_cfg, self.out_channels)
    
        delattr(self, 'conv_seg')

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, list):
            out_feats = []
            for x in inputs:
                x = x.permute(0,2,1).contiguous() # b,c,n  (b,512,1024)
                bs,c,n = x.shape
                w = h=int(math.sqrt(n))
                x = x.reshape(bs,c,h,w)
                out_feat = self.edgeocr_cls_head(x)
                out_feats.append(out_feat)
            out_feats = torch.stack(out_feats, 1).sum(1)
        else:
            x = inputs.permute(0,2,1).contiguous()
            bs,c,n = x.shape
            w = h=int(math.sqrt(n))
            x = x.reshape(bs,c,h,w)
            out_feats = self.edgeocr_cls_head(x)

        out_feats = self.edge_norm(out_feats)
        out = {"pred_contours":out_feats}
        return out     
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        edge_feat = self.forward(inputs)
        gt_semantic_seg[gt_semantic_seg == -1] = 255  # 把unseen类别像素置为背景像素   gt_semantic_seg：(bs,1,512,512)
        gt_semantic_seg[gt_semantic_seg == 200] = 255
        losses = self.losses(edge_feat, gt_semantic_seg)
        return losses
    
    @force_fp32(apply_to=('edge_logit',))
    def losses(self, edge_logit, seg_label, num_classes=None):
        """Compute Edge loss."""
        # edge_logit["pred_edges"] b,1,32,32
        if isinstance(edge_logit, dict):
            # edge loss
            seg_label = seg_label.squeeze(1)
            loss = self.loss_decode(
                edge_logit,
                seg_label,
                ignore_index = self.ignore_index)
            return loss