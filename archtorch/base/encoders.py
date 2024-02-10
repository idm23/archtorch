"""File containing basic encoder architectures
(Generally trades data-dimensionality for feature-dimensionality)
"""
# ======== standard imports ========
from typing import Iterable, Optional, Callable
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.exceptions as archexcp
import archtorch.base.blocks as archblocks
from archtorch.base.structures.stacks import ConvStack
# ==================================

class Encoder(torch.nn.Sequential):
    def __init__(
            self,
            encoder: torch.nn.Module,
            activation: Optional[torch.nn.Module] = None,
        ):
        super().__init__()
        self.append(encoder)
        if activation is not None:
            self.append(activation)

class BasicConvStackEncoder(Encoder):
    def __init__(
            self,
            ndims:int, n_levels:int,
            initial_input_features:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            raw_output:bool = True,
            **kwargs
        ):
        conv_stack = ConvStack(
            archblocks.ConvBlock, ndims, n_levels,
            initial_input_features, features_multiplier,
            kernel_size,
            initial_feature_jump=initial_feature_jump,
            final_feature_drop = final_feature_drop,
            raw_output=raw_output,
            **kwargs
        )
        self.output_features = conv_stack.output_features
        super().__init__(conv_stack)

class HalvingConvStackEncoder(Encoder):
    def __init__(
            self,
            ndims:int, n_levels:int,
            initial_input_features:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            raw_output:bool = True,
            **kwargs
        ):
        conv_stack = ConvStack(
            archblocks.ConvBlock, ndims, n_levels,
            initial_input_features, features_multiplier,
            kernel_size,
            operation_kwargs_override=ConvStack.halving_kwargs_override,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop,
            raw_output=raw_output,
            **kwargs
        )
        self.output_features = conv_stack.output_features
        super().__init__(conv_stack)
    
def test_basic_conv_encoder():
    B = 64
    input_features = 3
    H = W = D = 32
    input_shape = (B, input_features, H, W)
    encoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    conv_encoder_no_extras = BasicConvStackEncoder(
        2, len(kernel_sizes), input_features, 2, kernel_sizes, initial_feature_jump=32,
    )
    print(conv_encoder_no_extras, conv_encoder_no_extras(encoder_input).shape)
    conv_encoder_no_block_conv_args = BasicConvStackEncoder(
        2, len(kernel_sizes), input_features, 2, kernel_sizes, initial_feature_jump=32,
        padding = 'same', bias = False
    )
    print(conv_encoder_no_block_conv_args, conv_encoder_no_block_conv_args(encoder_input).shape)
    conv_encoder_block_conv_args = BasicConvStackEncoder(
        2, len(kernel_sizes), input_features, 2, kernel_sizes, initial_feature_jump=32,
        padding = 'same', bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1
    )
    print(conv_encoder_block_conv_args, conv_encoder_block_conv_args(encoder_input).shape)

def test_halving_conv_encoder():
    B = 64
    input_features = 3
    H = W = D = 32
    input_shape = (B, input_features, H, W)
    encoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    conv_encoder_no_extras = HalvingConvStackEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, 32
    )
    print(conv_encoder_no_extras, conv_encoder_no_extras(encoder_input).shape)
    conv_encoder_no_block_conv_args = HalvingConvStackEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, 32,
        padding = 'same', bias = False
    )
    print(conv_encoder_no_block_conv_args, conv_encoder_no_block_conv_args(encoder_input).shape)
    conv_encoder_block_conv_args = HalvingConvStackEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, 32,
        padding = 'same', bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1
    )
    print(conv_encoder_block_conv_args, conv_encoder_block_conv_args(encoder_input).shape)
    
def quicktests():
    test_basic_conv_encoder()
    test_halving_conv_encoder()

if __name__ == "__main__":
    quicktests()

