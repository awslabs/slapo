import torch
import torch.nn as nn
import torch.fx as fx

# Modified from Colossal-AI
# https://colossalai.org/docs/features/1D_tensor_parallel
class MLP(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = nn.Linear(dim, intermediate_dim)
        print(
            f"Weight of the first linear layer: {self.dense_1.weight.transpose(0, 1).shape}"
        )
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_dim, dim)
        print(
            f"Weight of the second linear layer: {self.dense_2.weight.transpose(0, 1).shape}"
        )

    def forward(self, x):
        x = self.dense_1(x)
        print(f"Output of the first linear layer: {x.shape}")
        x = self.activation(x)
        x = self.dense_2(x)
        print(f"Output of the second linear layer: {x.shape}")
        return x


X = torch.ones((256, 256))
m = MLP()
Y = m(X)
print(Y)


def transform(m: nn.Module) -> nn.Module:
    # Symbolic tracing frontend - captures the semantics of the module
    gm: fx.GraphModule = fx.symbolic_trace(m)

    # High-level intermediate representation (IR) - Graph representation
    print(gm.graph)
    modules = dict(gm.named_modules())
    print(modules["dense_1"])
    modules["dense_1"].out_features = 512
    print(modules["dense_1"])

    # Modify gm.graph
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == "call_module":
            # The target attribute is the function
            # that call_function calls.
            if node.target == "dense_1":
                print(type(node.target))
                print("here", node.target)

    gm.graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()
    print("Done recompilation")
    print(gm)

    # Code generation - valid Python code
    print(gm.code)

    return gm


t_m = transform(m)
# t_m.print_readable()
print(t_m(X))
