"""File defining types for TorchArchitecture.
"""
# ======== standard imports ========
from typing import Callable, Iterable, TypeAlias
# ==================================

# ======= third party imports ======
from torch import Tensor
# ==================================

# ========= program imports ========
# ==================================

# TODO: When pytorch supports python 3.12, change all this to just be types
NAMED_INPUTS: TypeAlias = dict[str, Tensor]
NAMED_TARGETS: TypeAlias = dict[str, Tensor]
NAMED_OUTPUTS: TypeAlias = dict[str, Tensor]
NAMED_LOSSES: TypeAlias = dict[str, Tensor]

NAMED_LOSS_INPUTS: TypeAlias = dict[str, Iterable[Tensor]]
LOSSFN: TypeAlias = Callable[[Iterable[Tensor]], Tensor]

DATAPROVIDER: TypeAlias = Iterable[tuple[NAMED_INPUTS, NAMED_TARGETS]]
CALLBACK: TypeAlias = Callable[[int, NAMED_INPUTS, NAMED_TARGETS, NAMED_OUTPUTS, NAMED_LOSSES], None]

