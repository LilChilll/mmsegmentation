import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
from torch.nn import functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn import Conv2d, ConvModule

class BasePixelDecoder(nn.Module):
    
    def __init__(
        self,
        input_shape: Dict[str, Dict[str, int]],
        *,
        conv_dim: int,
        mask_dim: int,
        norm_cfg=None,
        act_cfg=None
        # in_channels: Union[List[int], Tuple[int]],
        # feat_channels: int,
        # out_channels: int,
        # norm_cfg=None,
        # act_cfg=dict(type='ReLU'),
        # init_cfg= None
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]
        lateral_convs = []
        output_convs = []

        use_bias = norm_cfg == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                _,output_norm = build_norm_layer(norm_cfg, conv_dim)
                output_conv = ConvModule(
                    in_channels = in_channels,
                    out_channels= conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                )

                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)

    

    def forward(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y) # b,256,160,160
        return self.mask_features(y), None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)