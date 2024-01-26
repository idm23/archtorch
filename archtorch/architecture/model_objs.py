"""File defining classes building end to end deep learning models
with various loss functions.
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import archtorch.architecture.types as archtypes
import archtorch.architecture.exceptions as archexcp
import archtorch.architecture.dag_objs as archdag
# ==================================

def dict_append(
        all_dict:dict[str, list[torch.Tensor]],
        batched_dict:dict[str, torch.Tensor]
    ):
    all_dict.update({
        name:all_dict[name]+[batched_dict[name].detach().cpu()]
        for name in all_dict.keys()
    })

def dict_flatten(all_dict:dict[str, list[torch.Tensor]]):
    all_dict.update({
        name:torch.concat(all_dict[name])
        for name in all_dict.keys()
    })

class DLModel(torch.nn.Module):
    def __init__(
            self,
            componentFlow: archdag.ComponentFlow,
            optimizer_kwargs,
            verbose = False
        ):
        super().__init__()
        self.componentFlow = componentFlow
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_kwargs)
        self.verbose = verbose
    
    def update_model(self, batched_losses:archtypes.NAMED_LOSSES):
        self.optimizer.zero_grad()
        num_losses = len(batched_losses.keys())
        for i, loss_name in enumerate(batched_losses.keys()):
            not_last_item = i != (num_losses-1)
            batched_losses[loss_name].backward(retain_graph = not_last_item)
        self.optimizer.step()

    # TODO: Make a flow function to return all loss names/model_outputs to avoid if in variable_dict_append
    def epoch_reset(self):
        self.all_inputs = dict.fromkeys(self.componentFlow.required_input_strings, [])
        self.all_targets = dict.fromkeys(self.componentFlow.required_target_strings, [])
        self.all_outputs = dict.fromkeys(self.componentFlow.all_external_outputs, [])
        self.all_losses = dict.fromkeys(self.componentFlow.all_losses, [])

    def append_batched_data(
            self,
            inputs:archtypes.NAMED_INPUTS, targets:archtypes.NAMED_TARGETS,
            outputs:archtypes.NAMED_OUTPUTS, losses:archtypes.NAMED_LOSSES
        ):
        dict_append(self.all_inputs, inputs)
        dict_append(self.all_targets, targets)
        dict_append(self.all_outputs, outputs)
        # Losses are made to be zero-dimensional in nearly every supplied loss function
        dict_append(
            self.all_losses,
            {loss_name:losses[loss_name].unsqueeze(0) for loss_name in losses}
        )

    def flatten_stored_data(self):
        dict_flatten(self.all_inputs)
        dict_flatten(self.all_targets)
        dict_flatten(self.all_outputs)
        dict_flatten(self.all_losses)

    def training_loss_handler(self, batched_losses):
        self.update_model(batched_losses)

    def validation_loss_handler(self, batched_losses):
        pass   
    
    def forward(self, given_data:archtypes.DATAPROVIDER):
        self.epoch_reset()
        for input_data, target_data in given_data:

            # Perform model pass
            model_outputs, batched_losses = self.componentFlow(
                input_data
            )

            # Calculate External Loss Functions
            for combined_name in model_outputs.keys():
                comp_name, output_name = combined_name.split('_')
                relevant_component:archdag.Component = self.componentFlow.components[comp_name]
                fn_name, fn, target_name = relevant_component.external_loss_fns[output_name]
                batched_losses.update({
                    comp_name+'_'+fn_name : fn(
                        model_outputs[combined_name], target_data[target_name]
                    )
                })
            
            # Update model depending on model mode
            self.loss_handler(batched_losses)

            # Add all data to cpu storage   
            self.append_batched_data(input_data, target_data, model_outputs, batched_losses)
        self.flatten_stored_data()

    def default_callback(
            self,
            epoch:int,
            inputs:archtypes.NAMED_INPUTS, targets:archtypes.NAMED_TARGETS,
            outputs:archtypes.NAMED_OUTPUTS, losses:archtypes.NAMED_LOSSES,
        ):
        print(f'Epoch {epoch}')
        for loss_name in losses.keys():
            print(f'\t{loss_name}: {torch.mean(losses[loss_name])}')

    def default_training_callback(self, *args):
        print('Training Losses')
        self.default_callback(*args)

    def default_validation_callback(self, *args):
        print('Validation Losses')
        self.default_callback(*args)

    def set_callbacks(self, post_training_callback, post_validation_callback):
        if post_training_callback is None and self.verbose:
            post_training_callback = self.default_training_callback
        elif post_training_callback is None:
            post_training_callback = lambda *args: None
        else:
            pass
            #TODO: Assert the passed in function's signature
        if post_validation_callback is None and self.verbose:
            post_validation_callback = self.default_validation_callback
        elif post_validation_callback is None:
            post_validation_callback = lambda *args: None
        else:
            pass
            #TODO: Assert the passed in function's signature
        return post_training_callback, post_validation_callback

    def fit_to_data(
        self,
        training_data:archtypes.DATAPROVIDER, #TODO
        validation_data:archtypes.DATAPROVIDER, #TODO
        tol:float = 1e-6,
        consec_iters:int = 10,
        max_iters:int = 1000,
        #num_saved_samples = 3
        post_training_callback:archtypes.CALLBACK|None = None,
        post_validation_callback:archtypes.CALLBACK|None = None
    ):
        post_training_callback, post_validation_callback = self.set_callbacks(
            post_training_callback, post_validation_callback
        )
        no_good_progress_counter = 0
        niters = 0
        while (no_good_progress_counter < consec_iters) and niters < max_iters:
            self.train()
            self.loss_handler = self.training_loss_handler
            self(training_data)
            post_training_callback(
                niters, self.all_inputs, self.all_targets,
                self.all_outputs, self.all_losses
            )

            self.eval()
            self.loss_handler = self.validation_loss_handler
            with torch.no_grad():
                self(validation_data)
            post_validation_callback(
                niters, self.all_inputs, self.all_targets,
                self.all_outputs, self.all_losses
            )

            # TODO: Check losses to see if making good progress

            niters += 1
            #self.writer.flush()
    
    
        