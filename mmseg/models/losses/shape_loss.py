from .utils import InverseNet,mask_to_onehot, onehot_to_binary_edges
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

from mmseg.models.builder import LOSSES
import cv2
@LOSSES.register_module()
class JointEdgeSegLoss(nn.Module):
    def __init__(self, 
                 num_classes=1, 
                 edge_weight=5., 
                 inv_weight=5., 
                 loss_weight=1.,
                 inverseForm_pretrained=None,
                 reduce_zero_label=False,
                 align_corners=False):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = num_classes
        self.inverse_distance = InverseTransform2D(inverseForm_pretrained,align_corners=align_corners)
        self.reduce_zero_label = reduce_zero_label
        self.weight_dict = {"loss_edge": edge_weight, "loss_inv": inv_weight}
        self.loss_weight = loss_weight
        self.align_corners=align_corners
    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    # def edge_attention(self, input, target, edge):
    #     n, c, h, w = input.size()
    #     filler = torch.ones_like(target) * 255
    #     return self.seg_loss(input, 
    #                          torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, outputs, label,ignore_index=255):
        #self.inverse_distance.inversenet.zero_grad()
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        src_edges = outputs["pred_contours"] # bs,1,32,32
        target_edges = torch.stack([target["edge_gts"] for target in targets], dim=0) # bs,32,32
        target_edges = target_edges.unsqueeze(1) # bs,1,32,32
        losses = {
            "loss_edge": self.bce2d(src_edges, target_edges),
            "loss_inv":self.inverse_distance(src_edges, target_edges)
        }
        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] = losses[k] * self.weight_dict[k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses
    def prepare_targets(self, targets):
        new_targets = [] # targets (bs,512,512)
        for targets_per_image in targets:
            gt_cls = targets_per_image.unique()  # gt_cls: 这张gt_mask中有哪些类别
            if not self.reduce_zero_label:
                gt_cls = gt_cls[gt_cls!=0]
            gt_cls = gt_cls[gt_cls != self.ignore_index].cpu().numpy()  # filter out ignore classes

            _edgemap = targets_per_image.clone()
            _edgemap = _edgemap.cpu().numpy() # 512,512
            _edgemap = mask_to_onehot(_edgemap, gt_cls)
            if len(_edgemap)>0:
                _edgemap = onehot_to_binary_edges(_edgemap, 2, gt_cls) # 1,512,512
                edge_map = torch.from_numpy(_edgemap).float().squeeze(0)  # 512,512
                temp_e = cv2.resize(edge_map.numpy(), (int(edge_map.size(0) / 16), int(edge_map.size(1) / 16)),
                                    interpolation=cv2.INTER_NEAREST)
                small_edge_gts = torch.tensor(temp_e).cuda() # (32,32)
            else:
                small_edge_gts = torch.zeros((32,32)).cuda()
            new_targets.append(
                {
                    "edge_gts":small_edge_gts #(32,32)
                }
            )

        return new_targets


class InverseTransform2D(nn.Module):
    def __init__(self, inverseForm_pretrained=None,align_corners=False):
        super(InverseTransform2D, self).__init__()
        ## Setting up loss
        self.tile_factor = 3
        self.resized_dim = 672
        self.tiled_dim = self.resized_dim//self.tile_factor
        
        inversenet_backbone = InverseNet(pretrained=inverseForm_pretrained)
        self.inversenet = inversenet_backbone.cuda()
        self.align_corners=align_corners
        for param in self.inversenet.parameters():
            param.requires_grad = False            

    def forward(self, inputs, targets):   
        inputs = F.log_softmax(inputs,dim=1) 
            
        inputs = F.interpolate(inputs, size=(self.resized_dim, 2*self.resized_dim), mode='bilinear',align_corners=self.align_corners)
        targets = F.interpolate(targets, size=(self.resized_dim, 2*self.resized_dim), mode='bilinear',align_corners=self.align_corners)
        
        batch_size = inputs.shape[0]

        tiled_inputs = inputs[:,:,:self.tiled_dim,:self.tiled_dim]
        tiled_targets = targets[:,:,:self.tiled_dim,:self.tiled_dim]
        k=1      
        for i in range(0, self.tile_factor):
            for j in range(0, 2*self.tile_factor):
                if i+j!=0:
                    tiled_targets = \
                    torch.cat((tiled_targets, targets[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
                    k += 1

        k=1      
        for i in range(0, self.tile_factor):
            for j in range(0, 2*self.tile_factor):
                if i+j!=0:
                    tiled_inputs = \
                    torch.cat((tiled_inputs, inputs[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
                k += 1
                        
        _, _, distance_coeffs = self.inversenet(tiled_inputs, tiled_targets)
        
        mean_square_inverse_loss = (((distance_coeffs*distance_coeffs).sum(dim=1))**0.5).mean()
        return mean_square_inverse_loss
