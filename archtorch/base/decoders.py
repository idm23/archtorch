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
import archtorch.base.blocks as archblocks
from archtorch.base.structures.stacks import ConvStack
# ==================================

class Decoder(torch.nn.Sequential):
    def __init__(
            self,
            decoder: torch.nn.Module,
            activation: Optional[torch.nn.Module] = None,
        ):
        super().__init__()
        self.append(decoder)
        if activation is not None:
            self.append(activation)

class TConvDecoder(Decoder):
    def __init__(
            self,
            ndims:int, n_levels:int,
            initial_input_features:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            raw_output:bool = True,
            final_activation:Optional[torch.nn.Module] = None,
            **kwargs
        ):
        conv_stack = ConvStack(
            archblocks.TConvBlock, ndims, n_levels,
            initial_input_features, features_multiplier,
            kernel_size,
            initial_feature_jump=initial_feature_jump,
            final_feature_drop = final_feature_drop,
            raw_output=raw_output,
            **kwargs
        )
        self.output_features = conv_stack.output_features
        super().__init__(conv_stack, final_activation)

class HalvingTConvDecoder(Decoder):
    def __init__(
            self,
            ndims:int, n_levels:int,
            initial_input_features:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            raw_output:bool = True,
            final_activation:Optional[torch.nn.Module] = None,
            **kwargs
        ):
        conv_stack = ConvStack(
            archblocks.TConvBlock, ndims, n_levels,
            initial_input_features, features_multiplier,
            kernel_size,
            operation_kwargs_override = ConvStack.halving_kwargs_override,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop,
            raw_output = raw_output,
            **kwargs
        )
        self.output_features = conv_stack.output_features
        super().__init__(conv_stack, final_activation)

def test_tconv_decoder():
    B = 64
    input_features = 128
    H = W = D = 8
    input_shape = (B, input_features, H, W)
    decoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    tconv_decoder_no_extras = TConvDecoder(
        2, len(kernel_sizes), input_features, 0.5, kernel_sizes, final_feature_drop=3
    )
    print(tconv_decoder_no_extras, tconv_decoder_no_extras(decoder_input).shape)
    tconv_decoder_no_block_conv_args = TConvDecoder(
        2, len(kernel_sizes), input_features, 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False
    )
    print(tconv_decoder_no_block_conv_args, tconv_decoder_no_block_conv_args(decoder_input).shape)
    tconv_decoder_block_conv_args = TConvDecoder(
        2, len(kernel_sizes), input_features, 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1
    )
    print(tconv_decoder_block_conv_args, tconv_decoder_block_conv_args(decoder_input).shape)

def test_halving_tconv_decoder():
    B = 64
    input_features = 128
    H = W = D = 8
    input_shape = (B, input_features, H, W)
    decoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    tconv_decoder_no_extras = HalvingTConvDecoder(
        2, len(kernel_sizes), input_features, 0.5, kernel_sizes, final_feature_drop=3
    )
    print(tconv_decoder_no_extras, tconv_decoder_no_extras(decoder_input).shape)
    tconv_decoder_no_block_conv_args = HalvingTConvDecoder(
        2, len(kernel_sizes), input_features, 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False
    )
    print(tconv_decoder_no_block_conv_args, tconv_decoder_no_block_conv_args(decoder_input).shape)
    tconv_decoder_block_conv_args = HalvingTConvDecoder(
        2, len(kernel_sizes), input_features, 0.5, kernel_sizes, final_feature_drop=3,
        padding = 1, bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1
    )
    print(tconv_decoder_block_conv_args, tconv_decoder_block_conv_args(decoder_input).shape)
    
def quicktests():
    test_tconv_decoder()
    test_halving_tconv_decoder()

if __name__ == "__main__":
    quicktests()

