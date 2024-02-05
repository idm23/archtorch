"""File containing basic decoder architectures
(Generally trades feature-dimensionality for data-dimensionality)
"""
# ======== standard imports ========
from typing import Iterable, Optional, Callable
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.exceptions as archexcp
import archtorch.basemodels.basicblocks as archblocks
import archtorch.basemodels.encoders as archenc
# ==================================

class TConvDecoder(archenc.ConvEncoder):
    def __init__(
            self,
            ndims:int, initial_input_features:int,
            n_levels:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            **kwargs
        ):

        super().__init__(
            ndims, initial_input_features,
            n_levels, features_multiplier,
            kernel_size,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop,
            conv_block_type=archblocks.BasicTConvBlock,
            **kwargs
        )

class HalvingTConvDecoder(archenc.HalvingConvEncoder):
    def __init__(
            self,
            ndims:int, initial_input_features:int,
            n_levels:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            **kwargs
        ):

        super().__init__(
            ndims, initial_input_features,
            n_levels, features_multiplier,
            kernel_size,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop, 
            conv_block_type = archblocks.BasicTConvBlock,
            **kwargs
        )

def test_tconv_decoder():
    B = 64
    input_features = 128
    H = W = D = 32
    input_shape = (B, input_features, H, W)
    decoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    tconv_decoder_no_extras = TConvDecoder(
        2, input_features, len(kernel_sizes), 0.5, kernel_sizes, final_feature_drop=3
    )
    print(tconv_decoder_no_extras, tconv_decoder_no_extras(decoder_input).shape)
    tconv_decoder_no_block_conv_args = TConvDecoder(
        2, input_features, len(kernel_sizes), 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False
    )
    print(tconv_decoder_no_block_conv_args, tconv_decoder_no_block_conv_args(decoder_input).shape)
    tconv_decoder_block_conv_args = TConvDecoder(
        2, input_features, len(kernel_sizes), 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1
    )
    print(tconv_decoder_block_conv_args, tconv_decoder_block_conv_args(decoder_input).shape)

def test_halving_tconv_decoder():
    B = 64
    input_features = 128
    H = W = D = 32
    input_shape = (B, input_features, H, W)
    decoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    tconv_decoder_no_extras = HalvingTConvDecoder(
        2, input_features, len(kernel_sizes), 0.5, kernel_sizes, final_feature_drop=3
    )
    print(tconv_decoder_no_extras, tconv_decoder_no_extras(decoder_input).shape)
    tconv_decoder_no_block_conv_args = HalvingTConvDecoder(
        2, input_features, len(kernel_sizes), 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False
    )
    print(tconv_decoder_no_block_conv_args, tconv_decoder_no_block_conv_args(decoder_input).shape)
    tconv_decoder_block_conv_args = HalvingTConvDecoder(
        2, input_features, len(kernel_sizes), 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1
    )
    print(tconv_decoder_block_conv_args, tconv_decoder_block_conv_args(decoder_input).shape)
    
def quicktests():
    test_tconv_decoder()
    test_halving_tconv_decoder()

if __name__ == "__main__":
    quicktests()

