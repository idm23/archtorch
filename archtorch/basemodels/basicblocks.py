"""File containing basic units of neural network construction
"""
# ======== standard imports ========
import inspect
from typing import Optional, Iterable
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.exceptions as archexcp
# ==================================

class BasicBlock(torch.nn.Module):
    DIM2BATCHNORM:dict[int, torch.nn.Module] = {
    0:torch.nn.BatchNorm1d,
    1:torch.nn.BatchNorm1d,
    2:torch.nn.BatchNorm2d,
    3:torch.nn.BatchNorm3d
    }
    DIM2DROPOUT:dict[int, torch.nn.Module] = {
        0:torch.nn.Dropout1d,
        1:torch.nn.Dropout1d,
        2:torch.nn.Dropout2d,
        3:torch.nn.Dropout3d
    }
    LETTER2ORDER:dict[str,str] = {
        'O':'operation',
        'A':'activation',
        'D':'dropout',
        'N':'norm'
    }
    def __init__(
        self,
        features:int, batchless_ndims:int,
        activation:Optional[torch.nn.Module] = None,
        norm:Optional[torch.nn.Module] = None,
        dropout:Optional[torch.nn.Module] = None,
        ordering:str = 'OADN'
    ):
        super().__init__()
        self.temp_block = {}
    
        if norm == None:
            self.temp_block['norm'] = torch.nn.Identity()
        elif norm == 'bn':
            self.temp_block['norm'] = self.DIM2BATCHNORM[batchless_ndims](features)
        else:
            raise archexcp.ModelConstructionError(f'Norm {norm} not supported yet')
        
        if dropout == None:
            self.temp_block['dropout'] = torch.nn.Identity()
        elif type(dropout)==float and (0 < dropout < 1):
            self.temp_block['dropout'] = self.DIM2DROPOUT[batchless_ndims](dropout)
        else:
            raise archexcp.ModelConstructionError(f'Dropout {dropout} not supported yet')
        
        if activation == None:
            self.temp_block['activation'] = torch.nn.Identity()
        elif activation == 'relu':
            self.temp_block['activation'] = torch.nn.ReLU()
        else:
            raise archexcp.ModelConstructionError(f'Activation {activation} not supported yet')
        
        self.temp_block['operation'] = torch.nn.Identity()
        self.block = torch.nn.ModuleDict()
        self.ordering = [self.LETTER2ORDER[str.upper(letter)] for letter in ordering]
        for layer_name in self.ordering:
            self.block[layer_name] = self.temp_block[layer_name]

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer_name in self.ordering:
            x = self.block[layer_name](x)
        return x

def split_block_and_operation_kwargs(
        operation_module:torch.nn.Module, **kwargs:dict
    ) -> tuple[dict, dict]:
    block_kwargs = {
        name:(kwargs[name] if name in kwargs.keys() else param.default)
        for name, param in inspect.signature(BasicBlock.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    operation_kwargs = {
        name:(kwargs[name] if name in kwargs.keys() else param.default)
        for name, param in inspect.signature(operation_module.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    try:
        assert set(block_kwargs.keys()).intersection(operation_kwargs.keys()) == set()
    except:
        raise archexcp.ModelConstructionError(
            f'Error constructing BasicBlock with'
            +f'\n\tBlock Kwargs {block_kwargs}'
            +f'\n\tOperation Kwargs {operation_kwargs}'
            +'\nOverlap in keys detected.'
        )
    return block_kwargs, operation_kwargs

class BasicFCBlock(BasicBlock):
    def __init__(
            self,
            ndims:int, input_features:int, output_features:int,
            **kwargs
        ):
        block_kwargs, fc_kwargs = split_block_and_operation_kwargs(torch.nn.Linear, **kwargs)
        super().__init__(output_features, 1, **block_kwargs)
        # TODO: Add reshaping
        self.block['operation'] = torch.nn.Sequential(
            torch.nn.Linear(
                input_features,
                output_features,
                **fc_kwargs
            ),
        )
    
class ABSTRACTConvBlock(BasicBlock):
    DIM2CONV = {
        1:None,
        2:None,
        3:None
    }

    def __init__(
            self,
            ndims:int, input_features:int, output_features:int, kernel_size:int|Iterable[int],
            **kwargs
        ):
        if isinstance(kernel_size, int):
            assert kernel_size > 0
        elif isinstance(kernel_size, Iterable):
            for ksize in kernel_size:
                assert ksize > 0
        else:
            raise archexcp.ModelConstructionError(
                f'A kernel size was less than 0: {kernel_size}'
            )
        operation_module = self.DIM2CONV[ndims]
        block_kwargs, conv_kwargs = split_block_and_operation_kwargs(operation_module, **kwargs)
        super().__init__(output_features, ndims, **block_kwargs)
        self.block['operation'] = operation_module(
            input_features, output_features, kernel_size,
            **conv_kwargs
        )
    
class BasicConvBlock(ABSTRACTConvBlock):
    DIM2CONV = {
        1:torch.nn.Conv1d,
        2:torch.nn.Conv2d,
        3:torch.nn.Conv3d
    }
    
class BasicTConvBlock(ABSTRACTConvBlock):
    DIM2CONV = {
        1:torch.nn.ConvTranspose1d,
        2:torch.nn.ConvTranspose2d,
        3:torch.nn.ConvTranspose3d
    }
    
def test_fc_blocks():
    B = 64
    input_features = 100
    output_features = 200
    fc_input = torch.randn(B, input_features)
    fc_no_extras = BasicFCBlock(1, input_features, output_features)
    print(fc_no_extras, fc_no_extras(fc_input).shape)
    fc_no_block_no_bias = BasicFCBlock(1, input_features, output_features, bias = False)
    print(fc_no_block_no_bias, fc_no_block_no_bias(fc_input).shape)
    fc_block_no_bias = BasicFCBlock(1, input_features, output_features, bias = False, activation = 'relu', norm = 'bn', dropout = 0.1)
    print(fc_block_no_bias, fc_block_no_bias(fc_input).shape)

def test_convn_blocks(n_dim):
    B = 64
    input_features = 100
    output_features = 200
    volume = (B, input_features) + ((32,)*n_dim)
    conv_input = torch.randn(*volume)
    conv_no_extras = BasicConvBlock(n_dim, input_features, output_features, 3)
    print(conv_no_extras, conv_no_extras(conv_input).shape)
    conv_no_block_no_bias = BasicConvBlock(n_dim, input_features, output_features, 3, padding = 1, bias = False)
    print(conv_no_block_no_bias, conv_no_block_no_bias(conv_input).shape)
    conv_block_no_bias = BasicConvBlock(n_dim, input_features, output_features, 3, padding = 1, bias = False, activation = 'relu', norm = 'bn', dropout = 0.1)
    print(conv_block_no_bias, conv_block_no_bias(conv_input).shape)

def test_conv_blocks():
    for i in range(1,4):
        test_convn_blocks(i)

def test_tconv_blocks():
    pass
    
def quicktests():
    test_fc_blocks()
    test_conv_blocks()
    test_tconv_blocks()

if __name__ == "__main__":
    quicktests()