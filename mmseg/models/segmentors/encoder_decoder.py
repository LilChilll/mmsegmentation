# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.core import add_prefix
from mmseg.core.utils import (ConfigType, OptConfigType, OptMultiConfig,
                              OptSampleList, SampleList)
from mmseg.registry import MODELS
from .base import BaseSegmentor


@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super(EncoderDecoder, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, batch_inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, batch_inputs: Tensor,
                      batch_img_metas: List[dict]) -> List[Tensor]:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(batch_inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return list(seg_logits)

    def _decode_head_forward_train(self, batch_inputs: List[Tensor],
                                   batch_data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(batch_inputs, batch_data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(
        self,
        batch_inputs: List[Tensor],
        batch_data_samples: SampleList,
    ) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(batch_inputs, batch_data_samples,
                                         self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(batch_inputs,
                                                batch_data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            img (Tensor): Input images.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(batch_inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, batch_data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

        seg_logit_list = self.inference(batch_inputs, batch_img_metas)

        return self.postprocess_result(seg_logit_list, batch_img_metas)

    def _forward(self,
                 batch_inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(batch_inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, batch_inputs: Tensor,
                        batch_img_metas: List[dict]) -> List[Tensor]:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            batch_inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            List[:obj:`Tensor`]: List of segmentation results, seg_logits from
                model of each input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = batch_inputs.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = batch_inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = batch_inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = batch_inputs[:, :, y1:y2, x1:x2]
                # change the img shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is list of seg logits map
                # with shape [C, H, W]
                crop_seg_logit = torch.stack(
                    self.encode_decode(crop_img, batch_img_metas), dim=0)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits_list = list(preds / count_mat)

        return seg_logits_list

    def whole_inference(self, batch_inputs: Tensor,
                        batch_img_metas: List[dict]) -> List[Tensor]:
        """Inference with full image.

        Args:
            batch_inputs (Tensor): The tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            List[:obj:`Tensor`]: List of segmentation results, seg_logits from
                model of each input image.
        """

        seg_logits_list = self.encode_decode(batch_inputs, batch_img_metas)

        return seg_logits_list

    def inference(self, batch_inputs: Tensor,
                  batch_img_metas: List[dict]) -> List[Tensor]:
        """Inference with slide/whole style.

        Args:
            batch_inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            List[:obj:`Tensor`]: List of segmentation results, seg_logits from
                model of each input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit_list = self.slide_inference(batch_inputs,
                                                  batch_img_metas)
        else:
            seg_logit_list = self.whole_inference(batch_inputs,
                                                  batch_img_metas)

        return seg_logit_list

    def aug_test(self, batch_inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(batch_inputs[0], batch_img_metas[0],
                                   rescale)
        for i in range(1, len(batch_inputs)):
            cur_seg_logit = self.inference(batch_inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(batch_inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
