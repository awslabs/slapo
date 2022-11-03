import gc
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

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
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
        with record_function("model_inference_fw"):
            output = model(*inputs)
        # backward
        with record_function("model_inference_bw"):
            output["logits"].mean().backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))
