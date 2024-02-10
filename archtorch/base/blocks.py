"""File containing basic units of neural network construction
"""
# ======== standard imports ========
import inspect
from typing import Optional, Iterable
from collections import OrderedDict
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.exceptions as archexcp
# ==================================

class Block(torch.nn.Sequential):
    DIM2BATCHNORM:dict[int, torch.nn.Module] = {
        0:torch.nn.BatchNorm1d,
        1:torch.nn.BatchNorm1d,
        2:torch.nn.BatchNorm2d,
        3:torch.nn.BatchNorm3d
    }
    NAME2NORM:dict[str, dict[int, torch.nn.Module]] = {
        'bn': DIM2BATCHNORM
    }
    NAME2ACTIVATION:dict[str, torch.nn.Module] = {
        'relu': torch.nn.ReLU
    }
    DIM2DROPOUT:dict[int, torch.nn.Module] = {
        0:torch.nn.Dropout1d,
        1:torch.nn.Dropout1d,
        2:torch.nn.Dropout2d,
        3:torch.nn.Dropout3d
    }
    LETTER2OP:dict[str,str] = {
        'O':'operation',
        'A':'activation',
        'D':'dropout',
        'N':'norm'
    }
    def __init__(
        self,
        in_features:int, batchless_ndims:int,
        activation:Optional[torch.nn.Module] = None,
        norm:Optional[torch.nn.Module] = None,
        dropout:Optional[torch.nn.Module] = None,
        operation_and_outfeatures:Optional[tuple[torch.nn.Module, int]] = None,
        ordering:str = 'NOAD'
    ):
        super().__init__()

        for letter in ordering:
            match (letter.upper(), activation, norm, dropout, operation_and_outfeatures):
                case ('N', _, None, _, _):
                    pass
                case ('N', _, norm_name, _, _) if norm_name in self.NAME2NORM.keys():
                    self.append(self.NAME2NORM[norm_name][batchless_ndims](in_features))
                case ('N', _, _, _, _):
                    raise archexcp.ModelConstructionError(f'Norm {norm} not supported yet')
                case ('D', _, _, None, _):
                    pass
                case ('D', _, _, p, _) if isinstance(p, float):
                    self.append(self.DIM2DROPOUT[batchless_ndims](p))
                case ('D', _, _, _, _):
                    raise archexcp.ModelConstructionError(f'Dropout {dropout} not supported yet')
                case ('A', None, _, _, _):
                    pass
                case ('A', activation_str, _, _, _) if activation_str in self.NAME2ACTIVATION.keys():
                    self.append(self.NAME2ACTIVATION[activation_str]())
                case ('A', _, _, _, _):
                    raise archexcp.ModelConstructionError(f'Activation {activation} not supported yet')
                case ('O', _, _, _, None):
                    pass
                case ('O', _, _, _, (operation, out_features)) if isinstance(operation, torch.nn.Module) and isinstance(out_features, int): 
                    self.append(operation)
                    in_features = out_features
                case ('O', _, _, _, _):
                    raise archexcp.ModelConstructionError(f'Operation must be a torch.nn.Module and output features must be an int')
                case (_ as _letter, _, _, _, _):
                    raise archexcp.ModelConstructionError(f'Invalid letter: {_letter} for ordering')

def split_block_and_operation_kwargs(
        operation_module:torch.nn.Module, **kwargs:dict
    ) -> tuple[dict, dict]:
    block_kwargs = {
        name:(kwargs[name] if name in kwargs.keys() else param.default)
        for name, param in inspect.signature(Block.__init__).parameters.items()
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

class FCBlock(Block):
    # TODO: Add reshaping
    def __init__(
            self,
            ndims:int, input_features:int, output_features:int,
            **kwargs
        ):
        block_kwargs, fc_kwargs = split_block_and_operation_kwargs(torch.nn.Linear, **kwargs)
        fc_layer = torch.nn.Sequential(
            torch.nn.Linear(
                input_features,
                output_features,
                **fc_kwargs
            ),
        )
        block_kwargs['operation_and_outfeatures'] = (fc_layer, output_features)
        super().__init__(input_features, 1, **block_kwargs)
    
class ABSTRACTConvBlock(Block):
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
                f'Kernel size not understood as int or Iterable: {kernel_size}'
            )
        operation_module = self.DIM2CONV[ndims]
        block_kwargs, conv_kwargs = split_block_and_operation_kwargs(operation_module, **kwargs)
        conv_layer = operation_module(
            input_features, output_features,
            kernel_size,
            **conv_kwargs
        )
        block_kwargs['operation_and_outfeatures'] = (conv_layer, output_features)
        super().__init__(input_features, ndims, **block_kwargs)
    
class ConvBlock(ABSTRACTConvBlock):
    DIM2CONV = {
        1:torch.nn.Conv1d,
        2:torch.nn.Conv2d,
        3:torch.nn.Conv3d
    }
    
class TConvBlock(ABSTRACTConvBlock):
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
    fc_no_extras = FCBlock(1, input_features, output_features)
    print(fc_no_extras, fc_no_extras(fc_input).shape)
    fc_no_block_no_bias = FCBlock(1, input_features, output_features, bias = False)
    print(fc_no_block_no_bias, fc_no_block_no_bias(fc_input).shape)
    fc_block_no_bias = FCBlock(1, input_features, output_features, bias = False, activation = 'relu', norm = 'bn', dropout = 0.1)
    print(fc_block_no_bias, fc_block_no_bias(fc_input).shape)

def test_convn_blocks(n_dim):
    B = 64
    input_features = 100
    output_features = 200
    volume = (B, input_features) + ((32,)*n_dim)
    conv_input = torch.randn(*volume)
    conv_no_extras = ConvBlock(n_dim, input_features, output_features, 3)
    print(conv_no_extras, conv_no_extras(conv_input).shape)
    conv_no_block_no_bias = ConvBlock(n_dim, input_features, output_features, 3, padding = 1, bias = False)
    print(conv_no_block_no_bias, conv_no_block_no_bias(conv_input).shape)
    conv_block_no_bias = ConvBlock(n_dim, input_features, output_features, 3, padding = 1, bias = False, activation = 'relu', norm = 'bn', dropout = 0.1)
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