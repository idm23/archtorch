"""File for runnning a sample model.
This file is intended to change over time, add features
and subsequently push those features off into other files
for integration into a broader system.
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
# ==================================

def directional_objfn(model_output, target_data):
    return torch.nn.functional.binary_cross_entropy_with_logits(model_output, (target_data[:, :, -1, 0] > 0).float())

def collate_fn(data):
    input_data, target_data = zip(*data)
    input_data = torch.stack(input_data).to(sbc.DEVICE)
    target_data = torch.stack(target_data).to(sbc.DEVICE)
    return {'rawOHLCV' : input_data}, {'fourteenDaysOut': target_data}

def visualize_data_pass(self, tag_prefix, epoch):
    sua.add_scalars(self.writer, tag_prefix, epoch, self.all_losses, self.all_outputs, self.all_targets)
    sua.add_roc(self.writer, tag_prefix, epoch, self.all_outputs, self.all_targets)
    sua.add_histograms(self.writer, tag_prefix, epoch, self.all_outputs, self.all_targets)
    #sua.add_images(self.writer, tag_prefix, epoch, self.all_inputs, self.all_outputs, self.all_targets)

def main():
    jan1_2023 = dt.date.fromisocalendar(2023, 1, 1)
    training_dset = DailyDifferencedPyTorchDataset(
        'AAPL', end_date = jan1_2023, fixed_target_ref=True, white_noise=(0, 1)
    )
    validation_dset = DailyDifferencedPyTorchDataset(
        'AAPL', start_date = jan1_2023, fixed_target_ref=True
    )

    autoencoder = sbconv.SingleConvAutoEncoder(
        'rawOHLCV', 'embeddedOHLCV',
        (sbc.STANDARD_DAILY_LENGTH, sbc.OHLCV_CHANNELS+1), n_levels=3,
        name='ConvAutoEncoder'
    )
    encoder = sbfc.SingleFCEncoder(
       'embeddedOHLCV', 'direction',
        autoencoder.batchless_output_shapes['embeddedOHLCV'],
        num_outputs=1, n_levels=1,
        name='FCEncoderHead'
    )
    encoder.add_external_loss_fn('direction', directional_objfn, 'directionalBCE', 'fourteenDaysOut')
    flow = archdag.ComponentFlow()
    flow.add_component(autoencoder)
    flow.add_component(encoder)
    flow.add_connection(autoencoder, encoder)
    print(flow)
    tb_writer = BasicWriter()
    model = DLModel(
        flow,
        {
            'lr':3e-4,
            'weight_decay': 1e-2
        },
        verbose=True
    ).to(sbc.DEVICE)
    print(f'Number of parameters in the model: {get_n_params(model)}')
    model.fit_to_data(
        torch.utils.data.DataLoader(training_dset, batch_size = 64, shuffle = True, collate_fn = collate_fn),
        torch.utils.data.DataLoader(validation_dset, batch_size = 64, shuffle = True, collate_fn = collate_fn),
        post_training_callback=tb_writer.training_callback,
        post_validation_callback=tb_writer.validation_callback
    )

if __name__ == '__main__':
    main()