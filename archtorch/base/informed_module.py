"""File containing definitions for informed modules
"""
# ======== standard imports ========
from collections import OrderedDict
from typing import Iterable, Optional, Callable
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.exceptions as archexcp
import archtorch.base.structures as archstructs
import archtorch.base.blocks as archblocks
# ==================================

class ShapeConstraint:
    def __init__(self, str:str):
        pass


class TensorShape:
    def __init__(
            self,
            shape_vals:Iterable[int|str|ShapeConstraint|None],
            shape_labels:Iterable[str],
            
        ):
        assert len(shape_vals) == len(shape_labels)
        self.shape = {}
        for shape_label, shape_val in zip(shape_labels, shape_vals):
            assert shape_label not in self.shape.keys()
            if isinstance(shape_val, (int, ShapeConstraint, None)):
                self.shape[shape_label] = shape_val
            elif isinstance(shape_val, str):
                self.shape[shape_label] = ShapeConstraint(shape_val)
            else:
                raise Exception(f'Invalid shape value: {shape_val}')

class InformedModule(torch.nn.Module):
    def __init__(self, module, ):
        super().__init__()
        self.module = module
        self.batchless_input_shape
        self.batchless_output_shape

    def forward(self, x):
        return self.module(x)
