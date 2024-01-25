"""File defining classes for Directed Acyclic Graphs (DAG)
composed of PyTorch Modules
"""
# ======== standard imports ========
from collections import deque
# ==================================

# ======= third party imports ======
import torch
# ==================================

# ========= program imports ========
import stockbot.architecture.exceptions as archexcp
import stockbot.architecture.types as archtypes
# ==================================

class Component(torch.nn.Module):
    COMPONENT_NUMBER = 0

    def __init__(
            self,
            batchless_input_shapes: dict[str, tuple],
            batchless_output_shapes: dict[str, tuple],
            name:str|None
        ):
        super().__init__()
        
        self.batchless_input_shapes:dict[str, tuple] = batchless_input_shapes
        self.batchless_output_shapes:dict[str, tuple] = batchless_output_shapes
        if name == None:
            name = f'Component{self.COMPONENT_NUMBER}'
            # TODO: Like 99% certain this feature doesn't work.
            # the idea was that you could supply no name and this would
            # still count upwards for all models upon each initialization
            self.COMPONENT_NUMBER += 1
        self.name:str = name

        self._assert_valid_inputs()
        self._construct_modules()
        self.internal_loss_fns:dict[str, archtypes.LOSSFN] = {}
        self.external_loss_fns:dict[str, tuple[str, archtypes.LOSSFN, str]] = {}
        self.parent_components:list[Component] = []
        self.child_components:list[Component] = []

    def _assert_valid_inputs(self):
        for output_name in self.batchless_output_shapes.keys():
            assert '_' not in output_name
        for input_name in self.batchless_input_shapes.keys():
            assert '_' not in input_name
        assert '_' not in self.name

    def add_internal_loss_fn(self, fn:archtypes.LOSSFN, fn_name:str):
        assert fn_name not in self.internal_loss_fns.keys()
        self.internal_loss_fns[fn_name] = fn

    def add_external_loss_fn(self, output_name:str, fn:archtypes.LOSSFN, fn_name:str, target_name:str):
        assert fn_name not in [fn_name for (fn_name, _, _) in self.external_loss_fns.values()]
        self.external_loss_fns[output_name] = (fn_name, fn, target_name)

    def add_parent(self, parent):
        self.parent_components.append(parent)

    def add_child(self, child):
        self.child_components.append(child)

    def _construct_modules(self):
        raise NotImplementedError('Implement in subclass')
    
    def _model_pass(
            self,
            x: archtypes.NAMED_INPUTS
        ) -> tuple[archtypes.NAMED_OUTPUTS, archtypes.NAMED_LOSS_INPUTS]:
        raise NotImplementedError('Implement in subclass')
    
    def eval_losses(
            self, lossfn_args: archtypes.NAMED_LOSS_INPUTS,
        ) -> archtypes.NAMED_LOSSES:
        return {
            loss_name:self.internal_loss_fns[loss_name](*lossfn_args[loss_name])
            for loss_name in self.internal_loss_fns.keys()
        }
    
    def forward(
            self, x: archtypes.NAMED_INPUTS
        ) -> tuple[archtypes.NAMED_OUTPUTS, archtypes.NAMED_LOSSES]:
        outputs, lossfn_args = self._model_pass(x)
        losses = self.eval_losses(lossfn_args)
        return outputs, losses

    def confirm_sizing(self):
        batchless_sample_inputs = {
            input_name:torch.randn(self.batchless_input_shapes[input_name])
            for input_name in self.batchless_input_shapes.keys()
        }
        with torch.no_grad():
            self.eval()
            self._test_forward({
                input_name:batchless_sample_inputs[input_name].unsqueeze(0)
                for input_name in batchless_sample_inputs.keys()
            })
            self.train()
            self._test_forward({
                input_name:batchless_sample_inputs[input_name].repeat(2)
                for input_name in batchless_sample_inputs.keys()
            })

    def _test_forward(self, inputs: archtypes.NAMED_INPUTS):
        sample_outputs, sample_losses = self.forward(inputs)
        try:
            assert sample_outputs.keys() == self.batchless_output_shapes.keys()
            assert sample_losses.keys() == self.internal_loss_fns.keys()
            for output_name in range(sample_outputs.keys()):
                assert sample_outputs[output_name].shape == (inputs[0].shape[0],) + self.batchless_output_shapes[output_name]
            for loss_name in range(sample_losses.keys()):
                assert sample_losses[loss_name].shape == (inputs[0].shape[0],)
        except AssertionError as e:
            sample_output_shapes = {
                output_name:sample_outputs[output_name].shape
                for output_name in sample_outputs.keys()
            }
            sample_loss_shapes = {
                loss_name:sample_losses[loss_name].shape
                for loss_name in sample_losses.keys()
            }
            raise archexcp.ModelSizingError(
                f"Specified batchless output shapes as {self.batchless_output_shapes}\
                but got\n\
                \tSample Output Shapes: {sample_output_shapes}\n\
                \tSample Loss Shapes: {sample_loss_shapes}"
            )
    
class ComponentFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.components = torch.nn.ModuleDict()
        self._assign_attributes()

    def __str__(self):
        origninal_str = super().__str__()
        dag_info = (
            "DAG Info:\n"
            + f"\tIn Degrees: {self.in_degrees}\n"
            + f"\tAll Losses: {self.all_losses}\n"
            + f"\t\tAll Internal Losses: {self.all_internal_losses}\n"
            + f"\t\tAll External Losses: {self.all_external_losses}\n"
            + f"\tComponent Leaf Outputs {self.component_leaf_outputs}\n"
            + f"\tAll External Outputs: {self.all_external_outputs}\n"
            + f"\tComponent Execution Order: {self.component_execution_order}\n"
            + f"\tParent Child Data Relationships: {self.parent2child_datapass}\n"
            + f"\tRequired Inputs: {self.required_input_strings}\n"
            + f"\tRequired Targets: {self.required_target_strings}\n"
        )
        return origninal_str + '\n' + dag_info

    def add_component(self, component:Component):
        self.components.add_module(component.name, component)
        self._assign_attributes()

    def add_connection(self, parent_component:Component, child_component:Component):
        assert (
            len(
                parent_component.batchless_output_shapes.keys()
                & child_component.batchless_input_shapes.keys()
            ) >= 1
        )
        parent_component.add_child(child_component)
        child_component.add_parent(parent_component)
        self._assign_attributes()

    def _assign_attributes(self):
        self._assign_in_degrees()
        self._assign_losses()
        self._assign_data_pathways()

    def _assign_in_degrees(self):
        self.in_degrees = dict[str, int]()
        self.in_degrees_queue = deque[Component]()
        # Calculate in-degrees
        self.in_degrees = {comp_name: 0 for comp_name in self.components.keys()}
        for comp in self.components.values():
            for child in comp.child_components:
                self.in_degrees[child.name] += 1
        self.in_degrees_queue = deque(
            self.components[comp_name] for comp_name, in_degree in self.in_degrees.items()
            if in_degree == 0
        )
    
    def _assign_losses(self):
        self.all_losses = set[str]()
        self.all_internal_losses = set[str]()
        self.all_external_losses = set[str]()
        for component in self.components.values():
            self.all_internal_losses.update({
                component.name+'_'+loss_name
                for loss_name in component.internal_loss_fns.keys()
            })
            self.all_external_losses.update({
                component.name+'_'+loss_name
                for loss_name in [
                    fn_name for fn_name, _, _ in component.external_loss_fns.values()
                ]
            })
        self.all_losses = self.all_internal_losses.copy()
        self.all_losses.update(self.all_external_losses)

    def _assign_data_pathways(self):
        self.component_execution_order = list[str]()
        self.all_external_outputs = set[str]()
        self.component_leaf_outputs = dict[str, set[str]]()
        self.parent2child_datapass = dict[str, dict[str, set[str]]]()
        self.required_input_strings = set[str]()
        self.required_target_strings = set[str]()
        in_degrees = self.in_degrees.copy()
        in_degrees_queue = self.in_degrees_queue.copy()
        while in_degrees_queue:
            ccomp = in_degrees_queue.popleft()
            self.required_input_strings.update(ccomp.batchless_input_shapes.keys())
            self.required_target_strings.update([
                target_name for _, _, target_name in ccomp.external_loss_fns.values()
            ])
            self.component_execution_order.append(ccomp.name)
            
            all_child_inputs = set()
            self.parent2child_datapass[ccomp.name] = {child.name:set() for child in ccomp.child_components}
            for child in ccomp.child_components:
                matched_child_inputs = (child.batchless_input_shapes.keys() & ccomp.batchless_output_shapes.keys())
                self.parent2child_datapass[ccomp.name][child.name].update(matched_child_inputs)
                all_child_inputs.update(matched_child_inputs)
                in_degrees[child.name] -= 1
                if in_degrees[child.name] == 0:
                    in_degrees_queue.append((child))

            self.component_leaf_outputs[ccomp.name] = [
                output_name for output_name in ccomp.batchless_output_shapes.keys()
                if output_name not in all_child_inputs
            ]
            self.all_external_outputs.update({
                (ccomp.name + '_' + output_name)
                for output_name in ccomp.batchless_output_shapes.keys()
                if (
                    (output_name not in all_child_inputs)
                    or (output_name in ccomp.external_loss_fns.keys())
                )
            })
        for parent_name in self.parent2child_datapass.keys():
            for provided_inputs in self.parent2child_datapass[parent_name].values():
                self.required_input_strings -= provided_inputs
        

    def forward(
            self, all_external_inputs: archtypes.NAMED_INPUTS
        ) -> tuple[archtypes.NAMED_OUTPUTS, archtypes.NAMED_LOSSES]:
        all_internal_losses = dict.fromkeys(self.all_losses)
        all_external_outputs = dict.fromkeys(self.all_external_outputs)

        # Assign inputs
        comp_inputs:dict[Component, archtypes.NAMED_INPUTS] = {
            comp: {
                input_name:all_external_inputs[input_name]
                for input_name in (
                    all_external_inputs.keys() &
                    comp.batchless_input_shapes.keys()
                )
            }
            for comp in self.components.values()
        }

        for comp_name in self.component_execution_order:
            component:Component = self.components[comp_name]
            # Feed-forward inputs
            outputs, internal_losses = component(comp_inputs[component])
    
            # Store losses
            all_internal_losses.update({
                (comp_name+'_'+lossname):internal_losses[lossname]
                for lossname in internal_losses.keys()
            })

            # Store external outputs
            all_external_outputs.update({
                (comp_name+'_'+external_output_name):outputs[external_output_name]
                for external_output_name in self.component_leaf_outputs[comp_name]
            })

            # Pass data through to children
            for child in component.child_components:
                comp_inputs[child].update({
                    output_name:outputs[output_name]
                    for output_name in self.parent2child_datapass[comp_name][child.name]
                })
            
        return all_external_outputs, all_internal_losses