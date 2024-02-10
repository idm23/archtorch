"""File containing common autoencoder models
(Unsupervised model compressing input and self regulating reconstruction)
"""
# ======== standard imports ========
from typing import Iterable, Optional, Callable
# ==================================

# ======= third party imports ======
import torch
from archtorch.architecture.types import NAMED_INPUTS, NAMED_LOSS_INPUTS, NAMED_OUTPUTS
# ==================================

# ========= program imports ========
import archtorch.exceptions as archexcp
import archtorch.base.structures as archstructs
import archtorch.base.blocks as archblocks
import archtorch.base.encoders as archenc
import archtorch.base.decoders as archdec

from archtorch.architecture.dag_objs import Component
from archtorch.architecture.types import LOSSFN
# ==================================

class AutoEncoder(Component):
    def __init__(
            self,
            input_name:str, batchless_input_shape:Iterable[int], output_name:str, output_features:int,
            encoder:torch.nn.Module,
            decoder:torch.nn.Module,
            reconstruction_loss_fn:LOSSFN
        ):
        self.input_name = input_name
        self.output_name = output_name
        super().__init__({input_name:batchless_input_shape}, {output_name:(output_features,)})
        self.encoder = encoder
        self.decoder = decoder
        self.add_internal_loss_fn(reconstruction_loss_fn, 'reconLoss')

    def _model_pass(self, x: NAMED_INPUTS) -> tuple[NAMED_OUTPUTS, NAMED_LOSS_INPUTS]:
        x = x[self.input_name]
        output = self.encoder(x)
        recon_input = self.decoder(output)
        return {self.output_name: output.view(x.shape[0], -1)}, {'reconLoss': [recon_input, x]}

class ConvAutoEncoder(AutoEncoder):
    def __init__(
            self,
            input_name:str, batchless_input_shape:Iterable[int], output_name:str,
            n_levels:int, features_multiplier:float,
            encoder_kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            decoder_kernel_size:int|Iterable[int]|Iterable[int|Iterable[int]],
            encoder_initial_feature_jump:Optional[int] = None,
            encoder_final_feature_drop:Optional[int]=None,
            **kwargs
        ):
        self.initial_input_features, *self.dims = batchless_input_shape
        self.ndims = len(self.dims)

        self.n_levels = n_levels
        self.features_multiplier = features_multiplier

        self.input_name = input_name
        self.output_name = output_name

        encoder = archenc.HalvingConvStackEncoder(
            self.ndims, self.n_levels,
            self.initial_input_features, self.features_multiplier,
            encoder_kernel_size,
            initial_feature_jump=encoder_initial_feature_jump,
            final_feature_drop=encoder_final_feature_drop,
            raw_output=False,
            **kwargs
        )
        decoder = archdec.HalvingTConvDecoder(
            self.ndims, self.n_levels,
            encoder.output_features, (1/self.features_multiplier),
            decoder_kernel_size,
            final_feature_drop=self.initial_input_features,
            final_activation=torch.nn.Sigmoid(),
            **kwargs
        )
        super().__init__(
            input_name, batchless_input_shape, output_name, decoder.output_features,
            encoder, decoder, torch.nn.L1Loss()
        )
    
def test_basic_autoencoder():
    B = 64
    input_features = 3
    H = W = D = 32
    input_shape = (B, input_features, H, W)
    bae_input = {'randomImg': torch.randn(*input_shape)}

    bae_no_block_no_conv_args = ConvAutoEncoder(
        'randomImg', input_shape[1:], 'embeddedImg',
        4, 2.0, 4, 4, encoder_initial_feature_jump=32
    )
    print(bae_no_block_no_conv_args)
    outputs, losses = bae_no_block_no_conv_args(bae_input)
    for output_name, output in outputs.items():
        print(output_name, output.shape)
    for loss_name, loss in losses.items():
        print(loss_name, loss.shape)

    bae_no_block_conv_args = ConvAutoEncoder(
        'randomImg', input_shape[1:], 'embeddedImg',
        4, 2.0, 4, 4, encoder_initial_feature_jump=32,
        bias = False
    )
    print(bae_no_block_conv_args)
    outputs, losses = bae_no_block_conv_args(bae_input)
    for output_name, output in outputs.items():
        print(output_name, output.shape)
    for loss_name, loss in losses.items():
        print(loss_name, loss.shape)

    bae_block_conv_args = ConvAutoEncoder(
        'randomImg', input_shape[1:], 'embeddedImg',
        4, 2.0, 4, 4, encoder_initial_feature_jump=32,
        bias = False,
        activation = 'relu', norm = 'bn', dropout = 0.1, ordering = 'NOA'
    )
    print(bae_block_conv_args)
    outputs, losses = bae_no_block_conv_args(bae_input)
    for output_name, output in outputs.items():
        print(output_name, output.shape)
    for loss_name, loss in losses.items():
        print(loss_name, loss.shape)
    
def quicktests():
    test_basic_autoencoder()

if __name__ == "__main__":
    quicktests()