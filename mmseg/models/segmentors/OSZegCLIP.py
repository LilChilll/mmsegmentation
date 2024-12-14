# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import print_log
from torch import Tensor
from mmseg.utils import get_root_logger
from mmseg.models.builder import MODELS
# from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
#                          OptSampleList, SampleList, add_prefix)
from mmseg.core import add_prefix
from .encoder_decoder import EncoderDecoder
import numpy as np
import torch
from mmseg.ops import resize

@MODELS.register_module()
class OSZegCLIP(EncoderDecoder):

    def __init__(self,
                 text_encoder,
                 prompt_learner,
                 agnostic_mask_generater,
                #  class_names:List,
                 base_class:List,
                 novel_class:List,
                 both_class:List,
                 load_text_embedding:Optional[str]=None,
                 self_training = False,
                 ft_backbone=False,
                 exclude_key=None,
                 training = True,
                 **args
                 ):
        super(OSZegCLIP, self).__init__(**args)
        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        pretrained = args.get('pretrained', None)
        if pretrained is not None:
            text_encoder.pretrained = pretrained
        self.text_encoder = MODELS.build(text_encoder)
        if agnostic_mask_generater is not None:
            self.agnostic_mask_generater = MODELS.build(agnostic_mask_generater)
        if prompt_learner is not None:
            self.prompt_learner = MODELS.build(prompt_learner)
            self.prompt_learner.init_context(self.text_encoder)


        # assert self.with_decode_head
        self.exclude_key = exclude_key
        self.load_text_embedding = load_text_embedding
        if training:
            if len(self.base_class) != len(self.both_class): # zero-shot setting
                self._visiable_mask(self.base_class, self.novel_class, self.both_class)
                if not self.self_training:
                    self._visiable_mask(self.base_class, self.novel_class, self.both_class)
                else:
                    self._visiable_mask_st(self.base_class, self.novel_class, self.both_class)
                    self._st_mask(self.base_class, self.novel_class, self.both_class)
            self._freeze_stages(self.text_encoder, exclude_key=exclude_key)
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)
            # if prompt_learner:
            #     # self._freeze_stages(self.prompt_learner)
            #     enabled = set()
            #     for name, param in self.prompt_learner.named_parameters():
            #         if param.requires_grad:
            #             enabled.add(name)
            #     print(f"Text Prompt Parameters to be updated: {enabled}")

        else:
            self.text_encoder.eval()
            self.backbone.eval()
            if prompt_learner!=None:
                self.prompt_learner.eval()
        
    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in encoder:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes, novel_classes, both_classes):
        """
        Inductive setting
        :param seen_classes:
        :param novel_classes:
        :param both_classes:
        :return:
        """
        seen_map = np.array([-1]*256)
        seen_map[255] = 255
        for i,n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting:', self.visibility_seen_mask)

    def _visiable_mask_st(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([-1]*256)
        seen_map[255] = 255
        for i,n in enumerate(list(seen_classes)):
            seen_map[n] = n
        seen_map[200] = 200 # pixels of padding will be excluded
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting in self_traning stage:', self.visibility_seen_mask) 
    
    def _st_mask(self, seen_classes, novel_classes, both_classes):
        st_mask  = np.array([255]*256)
        st_mask[255] = 255
        for i,n in enumerate(list(novel_classes)):
            st_mask[n] = n
        self.st_mask = st_mask.copy()
        print('Making st mask for zero-shot setting in self_traning stage:', self.st_mask) 
    def _init_decode_head(self, decode_head) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.base_class = self.decode_head.base_class
        self.novel_class = self.decode_head.novel_class
        self.both_class = self.decode_head.both_class

    def _init_auxiliary_head(self, auxiliary_head) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    

    def _decode_head_forward_test(self, x, img_metas, self_training):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg, self_training)
        return seg_logits

    def _decode_head_forward_train(self, feat, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        if self.training:
            if len(self.base_class) != len(self.both_class): # zero setting
                gt_semantic_seg = torch.Tensor(self.visibility_seen_mask).type_as(gt_semantic_seg)[gt_semantic_seg]
                # bs,1,512,512  忽略像素为-1,(200)，背景像素为255. 在进入loss前，所有-1会转变为255
            
                
        
        losses = dict()
        losses = dict()
        if self.self_training: #and self._iter_counter >= self.start_self_train[0]:
            loss_decode = self.decode_head.forward_train(feat, 
                                                        img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        self.self_training,
                                                        self.st_mask)
        else:
            loss_decode = self.decode_head.forward_train(feat, 
                                                        img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        self.self_training)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, feat, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for edge head in
        training."""
        # if self.training:
        #     if len(self.base_class) != len(self.both_class): # zero setting
        #         gt_semantic_seg = torch.Tensor(self.visibility_seen_mask).type_as(gt_semantic_seg)[gt_semantic_seg]

        losses = dict()
        loss_edge= self.auxiliary_head.forward_train(feat,img_metas,gt_semantic_seg,self.train_cfg)

        losses.update(add_prefix(loss_edge, 'edge'))
        return losses
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
        # pdb.set_trace()
        bs = img.shape[0]
        global_embedding, visual_embedding, patch_tokens = self.extract_feat(img)
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding) # (c,512)
            text_feat = torch.from_numpy(text_feat).to(img.device)
            
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                num_cls, num_prompts, _ = self.texts.size()
                text_feat = self.text_embedding(self.texts.reshape(num_cls*num_prompts, -1), img)
                text_feat = text_feat.reshape(num_cls, num_prompts, -1).mean(dim=1)
                text_feat /= text_feat.norm(dim=-1).unsqueeze(1)
        
        feat= []
        prompt_text_feat = self.prompt_learner(patch_tokens,self.text_encoder)
        feat.append(visual_embedding)
        feat.append(global_embedding)
        feat.append(prompt_text_feat)
        if not self.self_training:
            text_feat = text_feat[self.base_class, :]
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(feat, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        # if self.with_edge_head:
        #     loss_edge= self._auxiliary_head_forward_train(
        #         edge_feat, img_metas, gt_semantic_seg)
        #     losses.update(loss_edge)

        return losses
    def _decode_head_forward_test(self, x, img_metas, self_training):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg, self_training)
        return seg_logits


    def encode_decode(self, img, img_metas):
        visual_feat = self.extract_feat(img)

        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = torch.from_numpy(text_feat).to(img.device) # c,512
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                num_cls, num_prompts, _ = self.texts.size()
                text_feat = self.text_embedding(self.texts.reshape(num_cls*num_prompts, -1), img)
                text_feat = text_feat.reshape(num_cls, num_prompts, -1).mean(dim=1)
                text_feat /= text_feat.norm(dim=-1).unsqueeze(1)
                
        local_feat = visual_feat[0] # b,d,h,w
        global_feat = visual_feat[1] # b,d
        edge_feat = visual_feat[2] # b,1,h,w
        # multi_scale_feat = visual_feat[3] # L ,h*w, b , 512
        prompt_text_feat = self.text_prompt_tuning(global_feat) # b,c,512
        feat = []
        feat.append(local_feat)
        feat.append(global_feat)
        feat.append(prompt_text_feat)
        out = self._decode_head_forward_test(feat, img_metas, self.self_training)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = len(self.both_class) #+1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        # preds[preds==num_classes-1] = 255
        return preds

    