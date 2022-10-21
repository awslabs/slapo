import time
import inspect
import argparse
import transformers.utils.fx as fx
from transformers import BertLMHeadModel, BertConfig
import numpy as np
import torch
import torch.nn as nn
import ms
import copy
from ms.utils import report_memory

def train(rank, args):
    # https://huggingface.co/bert-large-uncased/blob/main/config.json
    bert = BertLMHeadModel(BertConfig(num_attention_heads=16, hidden_size=1024, num_hidden_layers=24, is_decoder=True))
    bert.half()

    input_names = list(bert.dummy_inputs.keys())
    input_names += ["attention_mask", "labels"]
    sig = inspect.signature(bert.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    class NewTracer(fx.HFTracer):

        def __init__(self) -> None:
            super(NewTracer, self).__init__()

        # def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        #     # if "dense" in module_qualified_name:
        #     #     return False
        #     # else:
        #     #     return (
        #     #         (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
        #     #         and not isinstance(m, nn.Sequential)
        #     #     )
        #     if "self" in module_qualified_name:
        #         return True
        #     else:
        #         return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and super().is_leaf_module(
        #             m, module_qualified_name
        #         )

        def trace(self, *args, **kwargs):
            graph = super().trace(*args, **kwargs)
            return graph


    device = "cuda:{}".format(rank)
    # gm = fx.symbolic_trace(bert)
    traced_graph = NewTracer().trace(bert, concrete_args=concrete_args)
    gm = fx.GraphModule(bert, traced_graph).to(device)

    optimizer = torch.optim.SGD(bert.parameters(), lr=0.001)

    sch = ms.create_schedule(copy.deepcopy(gm), optimizer, args.world_size, rank)
    # print(sch.forward_ops)
    # print(sch.modules)
    # print(bert.config.vocab_size)

    sch.trace_module()
    for i in range(24):
        # MLP
        sch["bert.encoder.layer.{}.intermediate.dense".format(i)].shard(axis=1, param="weight")
        sch["bert.encoder.layer.{}.output.dense".format(i)].shard(axis=0, param="weight")
        sch["bert.encoder.layer.{}.output.dense".format(i)].gather()

        # # Attention
        # sch["bert.encoder.layer.{}.attention.self.query".format(i)].shard(axis=1, param="weight")
        # sch["bert.encoder.layer.{}.attention.self.key".format(i)].shard(axis=1, param="weight")
        # sch["bert.encoder.layer.{}.attention.self.value".format(i)].shard(axis=1, param="weight")
        # sch["bert.encoder.layer.{}.attention.output.dense".format(i)].shard(axis=0, param="weight")
        # sch["bert.encoder.layer.{}.attention.output.dense".format(i)].gather()

    report_memory(rank)
    model, optimizer = ms.build(sch)
    bert.cpu()
    report_memory(rank)

    bs = 6
    seq_length = 512
    bert_input_dict = {
        'input_ids': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
        'labels': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
        'attention_mask': torch.ones(bs, seq_length, device=device)}

    fw_time = []
    bw_time = []
    total_time = []
    for i in range(10):
        start_time = time.time()
        output = model(bert_input_dict["input_ids"].cuda(rank), bert_input_dict["attention_mask"].cuda(rank), bert_input_dict["labels"].cuda(rank))
        mid_time = time.time()
        # output["logits"].mean().backward()
        final_time = time.time()
        # optimizer.step()
        fw_time.append(mid_time - start_time)
        bw_time.append(final_time - mid_time)
        total_time.append(final_time - start_time)
        print(f"Finish step {i}, fw: {fw_time[-1]:.10f}s, bw: {bw_time[-1]:.10f}s, total: {total_time[-1]:.10f}s")
    fw_avg =np.array(fw_time[1:-1]).mean()
    bw_avg =np.array(bw_time[1:-1]).mean()
    total_avg =np.array(total_time[1:-1]).mean()
    print(f"Average fw: {fw_avg*1000:.10f}ms, bw: {bw_avg*1000:.10f}ms, total: {total_avg*1000:.10f}ms")

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[
    #         ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         output = model(bert_input_dict["input_ids"], bert_input_dict["attention_mask"], bert_input_dict["labels"])
        # # backward
        # output = model(bert_input_dict["input_ids"], bert_input_dict["attention_mask"], bert_input_dict["labels"])
        # with record_function("model_inference"):
        #     output["logits"].mean().backward()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    ms.execute(train, args)
