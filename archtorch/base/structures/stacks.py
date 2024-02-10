"""File containing stacked block structures and
functions useful for constructing them
"""
# ======== standard imports ========
from typing import Iterable, Callable, Optional
import inspect
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.base.blocks as archblocks
import archtorch.exceptions as archexcp
# ==================================

class StackedBlocks(torch.nn.Sequential):
    def __init__(
        self,
        block_type:archblocks.Block,
        ndims:int, n_levels:int,
        initial_input_features:int, features_multiplier:float,
        additional_operation_args:Iterable[Iterable],
        block_kwargs:dict, operation_kwargs:dict,
        operation_kwargs_override:Optional[Callable[[Iterable, dict], dict]] = None,
        initial_feature_jump:Optional[int] = None,
        final_feature_drop:Optional[int] = None,
        raw_output:bool = True
    ):
        super().__init__()
        if n_levels < 1:
            raise archexcp.ModelConstructionError('Need to have at least 1 level for a stack')
        self.ndims = ndims
        self.n_levels = n_levels
        self.initial_input_features = initial_input_features
        self.features_multiplier = features_multiplier
        self.block_kwargs = block_kwargs
        self.operation_kwargs = operation_kwargs
        self.initial_feature_jump = initial_feature_jump
        self.final_feature_drop = final_feature_drop
        self.raw_output = raw_output

        self.block_type = block_type
        self.unfilled_block_args_types = {
            name:param.annotation
            for name, param in inspect.signature(self.block_type.__init__).parameters.items()
            if (
                (name not in ['self', 'ndims', 'input_features', 'output_features', 'kwargs'])
                and (param.default is inspect.Parameter.empty)
            )
        }
        #print(self.unfilled_block_args_types)
        # TODO: Check typing on additional_operation_args as well as casting out.
        # If reference is necessary later, check Conv encoders
        self.additional_operation_args = additional_operation_args

        # TODO: Add typing support on operation_kwargs_override
        if operation_kwargs_override is None:
            self.operation_kwargs_override = lambda *given_operation_args, **operation_kwargs: operation_kwargs
        else:
            self.operation_kwargs_override = operation_kwargs_override
        
        self.initialize_blocks()
    
    def initialize_blocks(self) -> None:
        input_features = self.initial_input_features
        for level in range(self.n_levels):

            # Handle feature growth
            if level == 0 and self.initial_feature_jump is not None:
                output_features = self.initial_feature_jump
            elif level == (self.n_levels - 1) and self.final_feature_drop is not None:
                output_features = self.final_feature_drop
            else:
                output_features = int(input_features * self.features_multiplier)            

            # Base operational_kwargs off of the other parameters provided
            ckwargs = self.operation_kwargs_override(
                *self.additional_operation_args[level],
                **self.operation_kwargs
            )

            # Ensure last layer doesn't have block arguments
            if level != (self.n_levels - 1):
                ckwargs.update(self.block_kwargs)
            elif level == (self.n_levels - 1) and not self.raw_output:
                ckwargs.update(self.block_kwargs)
                
            # Build and stack block
            self.append(
                self.block_type(
                    self.ndims, input_features, output_features,
                    *self.additional_operation_args[level],
                    **ckwargs
                )
            )
            input_features = output_features
        
        self.output_features = output_features
    
class ConvStack(StackedBlocks):
    def __init__(
            self,
            conv_block_type:archblocks.ABSTRACTConvBlock,
            ndims:int, n_levels:int,
            initial_input_features:int, features_multiplier:float,
            kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            operation_kwargs_override:Optional[Callable[[Iterable, dict], dict]] = None,
            initial_feature_jump:Optional[int] = None,
            final_feature_drop:Optional[int] = None,
            raw_output:bool = True,
            **kwargs
        ):
        self.ndims = ndims
        self.n_levels = n_levels
        block_kwargs, conv_kwargs = archblocks.split_block_and_operation_kwargs(
            conv_block_type.DIM2CONV[ndims], **kwargs
        )

        self.kernel_sizes = self.get_kernel_sizes(kernel_size)
        super().__init__(
            conv_block_type, ndims, n_levels,
            initial_input_features, features_multiplier,
            [(ksize,) for ksize in self.kernel_sizes], # Wrap since this is our only additonal argument,
            block_kwargs, conv_kwargs,
            operation_kwargs_override = operation_kwargs_override,
            initial_feature_jump = initial_feature_jump,
            final_feature_drop = final_feature_drop,
            raw_output = raw_output
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
            kernel_sizes = [(kernel_size,)*self.ndims for level in range(self.n_levels)]
        elif kernel_typing == Iterable[int] and len(kernel_size) == self.ndims:
            # If this is a tuple with the same dimensionality as the input,
            # cast out the tuple for all conv_blocks
            kernel_sizes = [kernel_size for level in range(self.n_levels)]
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
    
    @classmethod
    def halving_kwargs_override(cls, kernel_size:int|Iterable[int], **conv_kwargs:dict):
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