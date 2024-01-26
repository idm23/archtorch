"""File defining writer classes for TorchArchitecture Deep
Learning Models
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
import torch
from torch.utils.tensorboard import SummaryWriter
# ==================================

# ========= program imports ========
import archtorch.architecture.types as archtypes
# ==================================

class BasicWriter:
    def __init__(self):
        self.writer = SummaryWriter()

    def _write_losses(self, tag_prefix:str, epoch:int, losses:archtypes.NAMED_LOSSES):
        for loss_name in losses.keys():
            self.writer.add_scalar(
                '/'.join([tag_prefix, loss_name]),
                torch.mean(losses[loss_name]), epoch
            )

    def _callback(
            self, tag_prefix:str,
            epoch:int,
            inputs:archtypes.NAMED_INPUTS, targets:archtypes.NAMED_TARGETS,
            outputs:archtypes.NAMED_OUTPUTS, losses:archtypes.NAMED_LOSSES
        ):
        self._write_losses(tag_prefix, epoch, losses)

    def training_callback(
            self, *args
        ):
        self._callback('Train', *args)

    def validation_callback(
            self, *args
        ):
        self._callback('Validation', *args)

#TODO:
"""
class BasicKeywordDrivenWriter(BasicWriter):
    def _callback(
            self, tag_prefix: str,
            epoch: int,
            inputs: archtypes.NAMED_INPUTS, targets: archtypes.NAMED_TARGETS,
            outputs: archtypes.NAMED_OUTPUTS, losses: archtypes.NAMED_LOSSES
        ):
        super()._callback(tag_prefix, epoch, inputs, targets, outputs, losses)
        for output_name in outputs.keys():
            if ':binary' in output_name:
"""
                

        

