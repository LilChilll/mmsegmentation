import numpy as np
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch
import torch.nn as nn
from mmseg.models.losses import accuracy

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
class SimpleMaskDecodeHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            seen_idx,
            all_idx,
            embed_dims=768,
            use_stages=1,
            align_corners = False,
            **kwargs,
    ):
        super(SimpleMaskDecodeHead, self).__init__(
             **kwargs)
        self.align_corners = align_corners
        self.image_size = img_size
        self.use_stages = use_stages
        # self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        dim = embed_dims
        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)
        self.q_proj = nn.Linear(dim * 2, dim)


    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def compute_similarity(self,image_features, text_features):
        # image_features [bs, n_i, d]
        # text_features [bs, n_t, d]
        prob_1 = image_features[:, :1, :] @ text_features.permute(0, 2, 1)
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[1], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(b, 1, n_t, c) # (bs,n_i,n_cls,d)
        similarity = feats.sum(-1) # (bs,n_i,n_cls)
        return (similarity/0.07).softmax(-1), prob_1
    
    def get_similarity_map(self,sm):
        shape = self.image_size
        side = int(sm.shape[1] ** 0.5)
        sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear', align_corners=self.align_corners)
        sm = sm.permute(0, 2, 3, 1)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid() # b,c,h,w
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1) # b,h,w
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def forward(self, inputs_both, self_training=None):
        inputs = inputs_both[0].flatten(2).permute(0,2,1) # b,1024,512
        cls_token = inputs_both[1].unsqueeze(1)  # (b,1,512)
        image_embeddings = torch.cat((cls_token,inputs),dim=1)
        text_token = inputs_both[2] # b,c',512
        text_embeddings = text_token
        text_embeddings = self.q_proj(self.get_qs(text_embeddings, cls_token.squeeze(1))) # b,c',512
        out = {}
        similarity, _ = self.compute_similarity(image_embeddings, text_embeddings)
        pred = self.get_similarity_map(similarity[:, 1:, :]).permute(0, 3, 1, 2)
        out.update({"pred_masks": pred})

        if self.training:
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, c', 32, 32)
        else:
            if self_training:
                if "prompt_pred_masks" in out:
                    out["pred"] = self.semantic_inference(out, self.seen_idx) #(bs, 20, 224, 224)
                else:
                    out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx) #(bs, 20, 224, 224)
            else:
                if "prompt_pred_masks" in out:
                    out["pred"] = self.semantic_inference(out,self.seen_idx,weight=0.1)
                else:
                    out["pred"] = self.semantic_inference(out["pred_masks"],self.seen_idx, weight=0.1)

            return out["pred"]                  
        return out

    def semantic_inference(self, out,seen_idx, weight=0.0):
        
        mask_pred = out["pred_masks"]
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]


    def get_qs(self, q, cls):
        # q = [q.cls, q]
        
        bs, _ = cls.shape
        if len(q.shape)==2:
            C, dim = q.shape
            q = q.expand(bs, -1, -1)

        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1)
        return q_

    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index = self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss