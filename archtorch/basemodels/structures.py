"""File containing common structures in neural networks
"""
# ======== standard imports ========
from typing import Iterable, Callable, Optional
import inspect
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.basemodels.basicblocks as archblocks
import archtorch.exceptions as archexcp
# ==================================

class StackedBlocks(torch.nn.Module):
    def __init__(
        self,
        block_type:archblocks.BasicBlock,
        ndims:int, initial_input_features:int,
        n_levels:int, features_multiplier:float,
        block_kwargs:dict, operation_kwargs:dict,
        additional_operation_args:Iterable[Iterable],
        operation_kwargs_override:Optional[Callable[[Iterable, dict], dict]] = None,
        initial_feature_jump:Optional[int] = None,
        final_feature_drop:Optional[int] = None,
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
        self.stacked_blocks = torch.nn.ModuleList()
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
                
            # Build and stack block
            self.stacked_blocks.append(
                self.block_type(
                    self.ndims, input_features, output_features,
                    *self.additional_operation_args[level],
                    **ckwargs
                )
            )
            input_features = output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.stacked_blocks:
            x = block(x)
        return x
    
"""
class ResidualLayer(torch.nn.Module):
    def __init__(self, input_features, inner_block:archblocks.BasicBlock):
        super().__init__()
        self.in_norm = inner_block.block['norm']
        self.in_activation = inner_block_module.activation
        self.inner_block_module = inner_block_module
        self.outer_operation = inner_block_module.operation
"""