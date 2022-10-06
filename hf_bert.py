import time
import inspect
import transformers.utils.fx as fx
from transformers import BertLMHeadModel, BertConfig
import torch
import ms

device = "cuda:0"

bert = BertLMHeadModel(BertConfig(is_decoder=True)).to(device)
# bert.eval()

input_names = bert.dummy_inputs.keys()
sig = inspect.signature(bert.forward)
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

# from megatron.model import BertModel, ModelType
# from megatron.initialize import initialize_megatron
# from megatron.training import get_model
# initialize_megatron()
# def model_provider_func(pre_process=True, post_process=True):
#     model = BertModel(
#             num_tokentypes=0,
#             add_binary_head=False,
#             parallel_output=True,
#             pre_process=False,
#             post_process=False)
#     return model
# bert = get_model(model_provider_func, ModelType.encoder_or_decoder)
# concrete_args = {"tokentype_ids": None, "lm_labels": None}

class NewTracer(fx.HFTracer):

    def __init__(self) -> None:
        super(NewTracer, self).__init__()

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if "self" in module_qualified_name:
            return True
        else:
            return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and super().is_leaf_module(
                m, module_qualified_name
            )

    def trace(self, *args, **kwargs):
        graph = super().trace(*args, **kwargs)
        return graph

# gm = fx.symbolic_trace(bert)
traced_graph = NewTracer().trace(bert, concrete_args=concrete_args)
gm = fx.GraphModule(bert, traced_graph)
print(gm.graph)
sys.exit()

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
    # https://github.com/NVIDIA/Megatron-LM/blob/master/megatron/model/fused_bias_gelu.py
    print("Replace GeLU with FusedBiasGeLU")
    print(sch.func_ops)
    # sch["gelu"].replace(ms.op.gelu)
    sch["gelu"].replace_module(ms.op.BiasGeLU)

def replace_attention():
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L384
    # https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/model/transformer.py#L306
    # MASTER_ADDR=localhost MASTER_PORT=6000 python3 hf_bert.py --micro-batch-size 4 --num-layers 12 --hidden-size 512 --num-attention-heads 2 --max-position-embeddings 512 --encoder-seq-length 512
    print("Replace HF BertAttention with Megatron CoreAttention")
    from megatron.model.transformer import ParallelAttention
    from megatron.model.utils import init_method_normal, scaled_init_method_normal
    from megatron.initialize import initialize_megatron
    initialize_megatron()

    class SelfAttention(torch.nn.Module):

        def __init__(self, layer_number):
            init_method = init_method_normal(0.006)
            output_layer_init_method = scaled_init_method_normal(0.006, layer_number)
            self.parallel_attention = ParallelAttention(init_method, output_layer_init_method, layer_number)

        def forward(self, hidden_states, attention_mask):
            output, bias = self.parallel_attention(hidden_states, attention_mask)
            return output + bias

    for op in sch.forward_ops:
        if "self" in op:
            print(op)
            sch[op].replace(SelfAttention, 12)

# replace_layernorm()
# replace_gelu()
# replace_attention()

model, optimizer = ms.build(sch)
# print(sch.gm)
# print(sch.gm.graph)

bs = 8
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
    final_time = time.time()
    optimizer.step()
    print(f"Finish step {i}, fw: {mid_time - start_time:.10f}s, bw: {final_time - mid_time:.10f}s, total: {final_time - start_time:.10f}s")

# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(bert_input_dict["input_ids"])

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))