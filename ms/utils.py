import gc
import torch
import torch.distributed as dist


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
