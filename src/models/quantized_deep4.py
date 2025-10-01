import numpy as np
import torch
from torch import nn
from torch.nn import init

import brevitas.nn as qnn
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var

# ---------- Quantizer configs ----------

class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
    """
    Input quantizer: keep it <= 8b so FINN can convert to MultiThreshold.
    Set min/max to realistic EEG ranges for your dataset if you know them.
    """
    bit_width = 8
    min_val = -4096.0
    max_val = 4095.0
    scaling_impl_type = ScalingImplType.CONST  # fixed range

def act(bit_width=8):
    # Activation quantizer: always return quant tensor so QONNX carries metadata.
    return qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

def aid(bit_width=8):
    # Identity that still carries quant metadata.
    return qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

# ---------- Model ----------

class QuantDeep4Net(BaseModel):
    """
    FINN-friendly Deep4Net from [1]:
      - QuantConv2d everywhere (optionally split first layer)
      - Activations <= 8-bit + return_quant_tensor=True
      - Bias enabled to simplify BN fold
      - Native pooling (MaxPool2d / AvgPool2d), final pool non-overlapping
    
    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_time_length,
        final_conv_length,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=9,
        pool_time_length=4,
        pool_time_stride=4,
        n_filters_2=50,
        filter_length_2=9,
        n_filters_3=100,
        filter_length_3=9,
        n_filters_4=200,
        filter_length_4=9,
        first_pool_mode="max",   # "max" or "avg"
        later_pool_mode="max",   # "max" or "avg"
        drop_prob=0.0,           # keep 0.0 for export; dropout is a no-op in eval
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
        act_bit_width=8,         # <= 8 so FINN converts Quant->MultiThreshold
        weight_bit_width=8,      # pick 4/6/8; keep <= 8 for simplicity
    ):
        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def _pool_cls(self, mode):
        if mode == "max":
            return nn.MaxPool2d
        elif mode == "avg":
            return nn.AvgPool2d
        raise ValueError(f"Unsupported pool mode: {mode}")

    def create_network(self):
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride

        FirstPool = self._pool_cls(self.first_pool_mode)
        LaterPool  = self._pool_cls(self.later_pool_mode)

        model = nn.Sequential()

        # Quantize input
        model.add_module(
            "input_quant",
            qnn.QuantHardTanh(act_quant=InputQuantizer, return_quant_tensor=True)
        )

        # First block (optionally split into temporal + spatial convs)
        if self.split_first_layer:
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            model.add_module(
                "conv_time",
                qnn.QuantConv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                    bias=True,
                    weight_bit_width=self.weight_bit_width,
                ),
            )
            model.add_module(
                "conv_spat",
                qnn.QuantConv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=True,
                    weight_bit_width=self.weight_bit_width,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module(
                "conv_time",
                qnn.QuantConv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=True,
                    weight_bit_width=self.weight_bit_width,
                ),
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            model.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )

        model.add_module("conv_nonlin", act(self.act_bit_width))
        model.add_module(
            "pool",
            FirstPool(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)),
        )
        model.add_module("pool_nonlin", aid(self.act_bit_width))

        # Helper to append repeated conv/pool blocks
        def add_conv_pool_block(model, n_in, n_out, filt_len, idx, last):
            sfx = f"_{idx}"
            if self.drop_prob and self.drop_prob > 0:
                model.add_module("drop" + sfx, nn.Dropout(p=self.drop_prob))
            model.add_module(
                "conv" + sfx,
                qnn.QuantConv2d(
                    n_in,
                    n_out,
                    (filt_len, 1),
                    stride=(1, 1) if not self.stride_before_pool else (self.pool_time_stride, 1),
                    bias=True,
                    weight_bit_width=self.weight_bit_width,
                ),
            )
            if self.batch_norm:
                model.add_module(
                    "bnorm" + sfx,
                    nn.BatchNorm2d(
                        n_out,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            model.add_module("nonlin" + sfx, act(self.act_bit_width))

            if not last:
                model.add_module(
                    "pool" + sfx,
                    LaterPool(kernel_size=(self.pool_time_length, 1),
                              stride=(1, 1) if self.stride_before_pool else (self.pool_time_stride, 1)),
                )
                model.add_module("pool_nonlin" + sfx, aid(self.act_bit_width))
            else:
                # FINAL POOL: make non-overlapping so FINN converts it to StreamingMaxPool
                model.add_module(
                    "pool" + sfx,
                    LaterPool(kernel_size=(5, 1), stride=(5, 1))
                )
                model.add_module("pool_nonlin" + sfx, aid(self.act_bit_width))

        add_conv_pool_block(model, n_filters_conv, self.n_filters_2, self.filter_length_2, 2, False)
        add_conv_pool_block(model, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3, False)
        add_conv_pool_block(model, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4, True)

        # Determine final conv length if needed
        model.eval()
        if self.final_conv_length == "auto":
            with torch.no_grad():
                dummy = np_to_var(
                    np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32)
                )
                out = model(dummy)
                n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        # Classifier conv (quantized) — produces (N, C, 1, 1)
        model.add_module(
            "conv_classifier",
            qnn.QuantConv2d(
                self.n_filters_4,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
                weight_bit_width=self.weight_bit_width,
            ),
        )

        # Softmax and Squeeze operations will be removed in dataflow partition step.
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        # ---------- Initialization (like original) ----------
        init.xavier_uniform_(model.conv_time.weight, gain=1)
        if self.split_first_layer:
            init.xavier_uniform_(model.conv_spat.weight, gain=1)
        if self.batch_norm:
            init.constant_(model.bnorm.weight, 1)
            init.constant_(model.bnorm.bias, 0)

        # conv blocks
        param_dict = dict(list(model.named_parameters()))
        for block_nr in range(2, 5):
            init.xavier_uniform_(param_dict[f"conv_{block_nr}.weight"], gain=1)
            if self.batch_norm:
                init.constant_(param_dict[f"bnorm_{block_nr}.weight"], 1)
                init.constant_(param_dict[f"bnorm_{block_nr}.bias"], 0)

        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)

        # Start in eval mode
        model.eval()
        return model


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)