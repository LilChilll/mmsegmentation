import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.builder import LOSSES
from .utils import InverseNet
# from .criterion import SegPlusCriterion
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import torch.distributed as dist
import cv2
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def dice_loss(inputs, targets, num_masks,epsilon=1e-6):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1) 
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def cosine_margin_loss(q, e, labels, tau=1.0, m=0.5):
    assert q.shape[1]+1 == e.shape[0]
    bs, n_cls, n_dim = q.shape
    q = q.reshape(bs*n_cls, n_dim)
    pos = torch.exp(F.cosine_similarity(q, e[labels.long()].reshape(bs*n_cls, n_dim)) / tau)
    neg = torch.exp(F.cosine_similarity(q.unsqueeze(1), e.unsqueeze(0), dim=-1) / tau)
    neg = torch.sum(neg, dim=-1) + m
    return 1 - torch.mean(torch.div(pos, neg))

def contrastive_loss(visual_features,text_embeddings, temperature=0.5):
    """
    计算对比损失
    :param visual_features: (B, C, D) 类别级别视觉特征
    :param text_embeddings: (B, C, D) 类别级别文本嵌入
    :return: 对比损失
    """
    B, C, D = visual_features.shape

    # L2 归一化
    visual_features = F.normalize(visual_features, dim=-1)  # (B, C, D)
    text_embeddings = F.normalize(text_embeddings, dim=-1)  # (B, C, D)

    # 计算相似度矩阵
    similarity = torch.einsum('bcd,bcd->bc', visual_features, text_embeddings)  # (B, C)

    # 正样本对的相似度
    positive_similarity = torch.exp(similarity / temperature)  # (B, C)

    # 所有负样本对的相似度
    all_similarity = torch.exp(torch.einsum('bcd,bkd->bck', visual_features, text_embeddings) / temperature)  # (B, C, C)
    negative_similarity = all_similarity.sum(dim=-1) - positive_similarity  # (B, C)

    # 对比损失
    loss = -torch.log(positive_similarity / (positive_similarity + negative_similarity))  # (B, C)
    return loss.mean()


class SegPlusCriterion(nn.Module):
    # in this version, both all masks and logits will be added to compute loss
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, 
                 num_classes, 
                 weight_dict, 
                 losses, 
                 eos_coef=0.1,
                 align_corners=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.align_corners = align_corners
        self.loss_map={"masks": self.loss_masks} # ,"contrastive":self.loss_contrastive
        

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        outputs: pred_logits: (bs, n_cls, 1)                       targets: len = bs
                 pred_masks:  (bs, n_cls, H, W)                    targets[0]: 'labels': eg: have the [2, 4] th classes = 2
                 pred: (bs, n_cls, H, W) = pred_logits*pred_masks              'masks':  eg: (2, H, W)
                 aux_outputs: mediate outputs
        """
        assert "pred_masks" in outputs
        

        # outputs["pred_masks"] = outputs["pred_masks"][:,:-1]

        # for focal loss
        src_masks = outputs["pred_masks"]
        target_masks = self._get_target_mask_binary_cross_entropy(src_masks, targets)

        bs, n_cls, H, W = target_masks.size()
        _, _, H_, W_ = src_masks.size()
        src_masks = src_masks.reshape(bs*n_cls, H_, W_)
        target_masks = target_masks.reshape(bs*n_cls, H, W) # 624,512,512
        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear",align_corners=self.align_corners
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)

        # for dice loss
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks_dice = outputs["pred_masks"]
        if src_masks_dice.dim() != 4:
            return {"no_loss": 0}
        src_masks_dice = src_masks_dice[src_idx]
        masks_dice = [t["target_masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks_dice, valid = nested_tensor_from_tensor_list(masks_dice).decompose()
        target_masks_dice = target_masks_dice.to(src_masks_dice)
        target_masks_dice = target_masks_dice[tgt_idx]

        # upsample predictions to the target size --> for aug_loss
        src_masks_dice = F.interpolate(
            src_masks_dice[:, None], size=target_masks_dice.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks_dice = src_masks_dice[:, 0].flatten(1)

        target_masks_dice = target_masks_dice.flatten(1)
        target_masks_dice = target_masks_dice.view(src_masks_dice.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks_dice, target_masks_dice, num_masks),
        }
   
        return losses
    def loss_contrastive(self, outputs, targets, indices, num_masks):

        visual_features, text_embeddings = outputs["feature_pairs"]
        total_loss = contrastive_loss(visual_features,text_embeddings,temperature=0.5)
        return {"loss_contrastive": total_loss}
    
    
    def _get_target_label_cross_entropy(self,out_masks,targets):
        B, C = out_masks.size()[:2]
        target_labels =torch.zeros(B, C).to(out_masks.device) 
        for i, target in enumerate(targets):
            labels = target["labels"]
            target_labels[i,labels]=1
        return target_labels
    def _get_target_mask_binary_cross_entropy(self, out_masks, targets):
        B, C = out_masks.size()[:2] # b,c+1
        H, W = targets[0]['masks'].size()
        target_masks_o = torch.zeros(B, C, H*W).to(out_masks.device) 
        for i, target in enumerate(targets):
            mask = target['masks'].long().reshape(-1)
            idx = torch.arange(0, H*W, 1).long().to(out_masks.device) 
            mask_o = mask[mask!=255]
            idx = idx[mask!=255]
            target_masks_o[i, mask_o, idx] = 1
        return target_masks_o.reshape(B, C, H, W)
    
    def _get_target_edge_attention_binary_cross_entropy(self, out_masks, targets):
        B, C = out_masks.size()[:2]
        H, W = targets[0]['masks'].size()
        target_masks_o = torch.zeros(B, C, H * W).to(out_masks.device) # bs,171,h*w
        for i, target in enumerate(targets):
            mask = target['masks'].long().reshape(-1)
            idx = torch.arange(0, H * W, 1).long().to(out_masks.device)
            mask_o = mask[mask != 255]
            idx = idx[mask != 255]
            target_masks_o[i, mask_o, idx] = 1  # 把gt中前景像素置全部为1，背景像素置为0
        return target_masks_o.reshape(B, C, H, W)
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = self.loss_map
        
        # if loss=="prompt_tuning":
        #     loss_map.update({"prompt_tuning":self.loss_logits})
        
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets

        labels = [x['labels'] for x in targets]
        indices_new = []
        for label in labels:
            t = torch.arange(len(label))
            indices_new.append([label, t])
        indices = indices_new
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # if "prompt_pred_masks" in outputs:
        #     # self.losses=["masks"]
        #     self.loss_map.update({"prompt_tuning":self.loss_logits})

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # use the indices as the last stage
                aux_loss = ["masks"]
                for loss in aux_loss:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


@LOSSES.register_module()
class SegLossPlus(nn.Module):
    """ATMLoss.
    """
    def __init__(self,
                 num_classes,
                 dec_layers,
                 mask_weight=20.0,
                 dice_weight=1.0,
                 constractive_weight=5.0,
                 loss_weight=1.0,
                 align_corners = False):
        super(SegLossPlus, self).__init__()
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight} #,"loss_contrastive":constractive_weight
        losses = ["masks"]#,"constractive"]
        
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict
        self.criterion = SegPlusCriterion(
            num_classes,
            weight_dict=weight_dict,
            losses=losses,
            align_corners=align_corners
        )
        
        self.loss_weight = loss_weight

    def forward(self,
                outputs,
                label,
                ignore_index=255,
                ):
        """Forward function."""
        
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        # if "feature_pairs" in outputs:
        #     self.criterion.losses.append("contrastive")

        losses = self.criterion(outputs, targets)
        # if not "logits" in outputs:
        #     self.criterion.weight_dict.pop("loss_logit")
        #     self.criterion.weight_dict.pop("loss_prompt_mask")
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def prepare_targets(self, targets):
        new_targets = []
        H,W = targets.size()[1:]
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            # bg_cls = gt_cls[gt_cls == self.ignore_index]
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            out = {
                "labels": gt_cls,
                "target_masks": masks,
                "masks": targets_per_image
            }
            new_targets.append(out)
        return new_targets

