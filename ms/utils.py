import gc
import torch
import torch.fx as fx
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple

# https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/optimization.py
def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def _get_unique_module_name(gm_or_modules, name):
    if isinstance(gm_or_modules, fx.GraphModule):
        named_module = dict(gm_or_modules.named_modules())
    else:
        named_module = gm_or_modules
    num = 1
    new_name = name + "_0"
    while new_name in named_module.keys():
        new_name = name + "_" + str(num)
        num += 1
    return new_name


def report_memory(rank, report_gc=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(
        "rank {}: {:.2f} MiB".format(
            rank, torch.cuda.max_memory_allocated() / 1024 / 1024
        )
    )
    if report_gc:
        gc.collect()
        tc = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    if dist.get_rank() == 0:
                        print("GC Tensor", type(obj), obj.size())
                    tc += obj.numel()
            except:
                pass


def profile(model, inputs):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        with record_function("model_inference_fw"):
            output = model(*inputs)
        # backward
        with record_function("model_inference_bw"):
            output["logits"].mean().backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_time_total", row_limit=10
        )
    )
