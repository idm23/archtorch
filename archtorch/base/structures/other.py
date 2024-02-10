class ResidualLayer(torch.nn.Module):
    def __init__(self, input_features, inner_block:archblocks.BasicBlock):
        super().__init__()
        self.in_norm = inner_block.block['norm']
        self.in_activation = inner_block_module.activation
        self.inner_block_module = inner_block_module
        self.outer_operation = inner_block_module.operation