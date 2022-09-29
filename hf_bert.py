import time
import transformers.utils.fx as fx
from transformers import BertLMHeadModel, BertConfig
import torch
import ms

device = "cuda:0"

bert = BertLMHeadModel(BertConfig(is_decoder=True)).to(device)
bert.eval()
gm = fx.symbolic_trace(bert)

optimizer = torch.optim.SGD(bert.parameters(), lr=0.001)

sch = ms.create_schedule(gm, optimizer)
# print(sch.forward_ops)
# print(sch.modules)
# print(bert.config.vocab_size)

def replace_layernorm():
    print("Replace LayerNorm with FusedLayerNorm")
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    for op in sch.forward_ops:
        if "LayerNorm" in op:
            sch[op].replace(FusedLayerNorm, arg_names=["normalized_shape"])

def replace_gelu():
    print("Replace GeLU with FusedBiasGeLU")
    print(sch.func_ops)
    sch["gelu"].replace(ms.op.bias_gelu_impl)
    print(sch.gm.graph)

# replace_layernorm()
replace_gelu()

model, optimizer = ms.build(sch)
print(model.graph)

bs = 16
seq_length = 512
bert_input_dict = {
    'input_ids': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
    'labels': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
    'attention_mask': torch.ones(bs, seq_length, device=device)}

for i in range(5):
    start_time = time.time()
    output = model(bert_input_dict["input_ids"])
    mid_time = time.time()
    output["logits"].mean().backward()
    optimizer.step()
    elapsed_time = time.time() - start_time
    print(f"Finish step {i}, fw time: {mid_time - start_time:.10f}s, bw time: {elapsed_time:.10f}s")

from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(bert_input_dict["input_ids"])

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))