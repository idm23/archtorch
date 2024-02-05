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
import archtorch.basemodels.structures as archstructs
import archtorch.basemodels.basicblocks as archblocks
# ==================================

class ConvEncoder(archstructs.StackedBlocks):
    def __init__(
            self,
            ndims:int, initial_input_features:int,
            n_levels:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            operation_kwargs_override:Optional[Callable[[Iterable, dict], dict]] = None,
            conv_block_type:archblocks.ABSTRACTConvBlock = archblocks.BasicConvBlock,
            **kwargs
        ):
        self.ndims = ndims
        self.n_levels = n_levels
        block_kwargs, conv_kwargs = archblocks.split_block_and_operation_kwargs(
            conv_block_type.DIM2CONV[ndims], **kwargs
        )
        self.kernel_sizes = self.get_kernel_sizes(kernel_size)  

        super().__init__(
            conv_block_type, ndims, initial_input_features,
            n_levels, features_multiplier, block_kwargs, conv_kwargs,
            [(ksize,) for ksize in self.kernel_sizes], # Wrap since this is our only additonal argument,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop,
            operation_kwargs_override=operation_kwargs_override
        )

    def get_kernel_input_type(
            self, kernel_size, level = 1
        ) -> int|Iterable[int]|Iterable[int|Iterable[int]]:
        if isinstance(kernel_size, int):
            return int
        elif isinstance(kernel_size, Iterable) and level == 0:
            for ksize in kernel_size:
                if not isinstance(ksize, int):
                    raise archexcp.ModelConstructionError(f'Invalid argument supplied for kernel_size on kernel:{ksize}')
            return Iterable[int]
        elif isinstance(kernel_size, Iterable) and level > 0:
            ksize_types = [self.get_kernel_input_type(ksize, level=level-1) for ksize in kernel_size]
            if int in ksize_types and Iterable[int] in ksize_types:
                return Iterable[int|Iterable[int]]
            elif int in ksize_types and Iterable[int] not in ksize_types:
                return int
            elif int not in ksize_types and Iterable[int] in ksize_types:
                return Iterable[int]
            else:
                raise archexcp.ModelConstructionError(f'Invalid argument supplied for kernel_size on kernel:{kernel_size}')
        else:
            raise archexcp.ModelConstructionError(f'Invalid argument supplied for kernel_size on kernel:{kernel_size}')

    def get_kernel_sizes(self, kernel_size) -> list[tuple[int]]:
        kernel_typing = self.get_kernel_input_type(kernel_size)
        if kernel_typing == int:
            # Cast out int to be kernel size for all conv_blocks
            kernel_sizes = [(kernel_size,)*self.ndims for level in self.n_levels]
        elif kernel_typing == Iterable[int] and len(kernel_size) == self.ndims:
            # If this is a tuple with the same dimensionality as the input,
            # cast out the tuple for all conv_blocks
            kernel_sizes = [kernel_size for level in self.n_levels]
        elif kernel_typing == Iterable[int] and len(kernel_size) == self.n_levels:
            # If this is a tuple with the same number of entries as levels in the stack,
            # enumerate the tuple as ints for each conv_block
            kernel_sizes = [(ksize,)*self.ndims for ksize in kernel_size]
        elif kernel_typing == Iterable[int] and len(kernel_size) != self.ndims and len(kernel_size) != self.n_levels:
            # That means we got an iterable that doesn't match either
            # the data dimensionality or the stack size
            raise archexcp.ModelConstructionError(f'Invalid argument supplied for kernel_size: {kernel_size}')
        elif kernel_typing == Iterable[int|Iterable[int]] and len(kernel_size) != self.n_levels:
            # Means we got an Iterable containing kernels for each conv_block
            # and the outermost iterable doesn't contain enough or too many kernels
            raise archexcp.ModelConstructionError(f'Not enough/Too Many additional operation args supplied for n_levels = {self.n_levels}')
        elif kernel_typing == Iterable[int|Iterable[int]] and len(kernel_size) == self.n_levels:
            # Means we got an Iterable containing kernels for each conv_block
            kernel_sizes = []
            for ksize in kernel_size:
                ksize_typing = self.get_kernel_input_type(ksize, level=0)
                if ksize_typing == Iterable[int] and len(ksize) != self.ndims:
                    # We got a dimension mismatch in one of the given kernels
                    raise archexcp.ModelConstructionError(f'Invalid argument supplied for kernel_size on kernel:{ksize}')
                elif ksize_typing == Iterable[int] and len(ksize) == self.ndims:
                    # Dimensions of kernel match conv_block dimensions
                    kernel_sizes.append(ksize)
                else:
                    # Kernel is uniform int
                    kernel_sizes.append((ksize,)*self.ndims)
        else:
            # We didn't get an int, Iterable of ints, or Iterable of (ints or Iterables of ints)
            raise archexcp.ModelConstructionError(f'Invalid argument supplied for kernel_size: {kernel_size}')
        
        return kernel_sizes


class HalvingConvEncoder(ConvEncoder):
    def __init__(
            self,
            ndims:int, initial_input_features:int,
            n_levels:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            conv_block_type:archblocks.ABSTRACTConvBlock = archblocks.BasicConvBlock,
            **kwargs
        ):

        super().__init__(
            ndims, initial_input_features,
            n_levels, features_multiplier,
            kernel_size,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop, 
            operation_kwargs_override=self.get_halving_kwargs,
            conv_block_type= conv_block_type,
            **kwargs
        )

    def get_halving_kwargs(self, kernel_size:int|Iterable[int], **conv_kwargs:dict):
        if isinstance(kernel_size, int):
            stride = 2
            padding = ((kernel_size + 1) // 2) - 1
        if isinstance(kernel_size, Iterable):
            stride = (2,)*len(kernel_size)
            padding = [((ksize + 1) // 2) - 1 for ksize in kernel_size]
        conv_kwargs = conv_kwargs.copy()
        conv_kwargs['stride'] = stride
        conv_kwargs['padding'] = padding
        return conv_kwargs
    
def test_basic_conv_encoder():
    B = 64
    input_features = 3
    H = W = D = 32
    input_shape = (B, input_features, H, W)
    encoder_input = torch.randn(*input_shape)
    kernel_sizes = [5,(3,2),3]
    conv_encoder_no_extras = ConvEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, initial_feature_jump=32
    )
    print(conv_encoder_no_extras, conv_encoder_no_extras(encoder_input).shape)
    conv_encoder_no_block_conv_args = ConvEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, initial_feature_jump=32,
        padding = 'same', bias = False
    )
    print(conv_encoder_no_block_conv_args, conv_encoder_no_block_conv_args(encoder_input).shape)
    conv_encoder_block_conv_args = ConvEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, initial_feature_jump=32,
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
    conv_encoder_no_extras = HalvingConvEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, 32
    )
    print(conv_encoder_no_extras, conv_encoder_no_extras(encoder_input).shape)
    conv_encoder_no_block_conv_args = HalvingConvEncoder(
        2, input_features, len(kernel_sizes), 2, kernel_sizes, 32,
        padding = 'same', bias = False
    )
    print(conv_encoder_no_block_conv_args, conv_encoder_no_block_conv_args(encoder_input).shape)
    conv_encoder_block_conv_args = HalvingConvEncoder(
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

