import time
import inspect
import transformers.utils.fx as fx
from transformers import BertLMHeadModel, BertConfig
import torch
import torch.nn as nn
import ms

device = "cuda:0"

# https://huggingface.co/bert-large-uncased/blob/main/config.json
bert = BertLMHeadModel(BertConfig(num_attention_heads=16, hidden_size=1024, num_hidden_layers=24, is_decoder=True)).to(device)
bert.half()

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

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
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
# print(gm.graph)
# sys.exit()

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
    sch["gelu"].replace_module(ms.op.BiasGeLU, half=True)

def replace_attention():
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L384
    # https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/model/transformer.py#L306
    #  MASTER_ADDR=localhost MASTER_PORT=6000 python3 hf_bert.py --micro-batch-size 8 --num-layers 12 --hidden-size 1024 --num-attention-heads 16 --max-position-embeddings 512 --encoder-seq-length 512 --fp16
    print("Replace HF BertAttention with Megatron CoreAttention")
    from megatron.model.transformer import ParallelAttention
    from megatron.model.utils import init_method_normal, scaled_init_method_normal
    from megatron.initialize import initialize_megatron
    initialize_megatron()

    class SelfAttention(nn.Module):

        def __init__(self, layer_number = 12):
            super(SelfAttention, self).__init__()
            init_method = init_method_normal(0.006)
            output_layer_init_method = scaled_init_method_normal(0.006, layer_number)
            self.parallel_attention = ParallelAttention(init_method, output_layer_init_method, layer_number)

        def forward(self, hidden_states, attention_mask):
            hidden_states = hidden_states.permute(1, 0, 2)
            output, bias = self.parallel_attention(hidden_states, attention_mask)
            output = output.permute(1, 0, 2)
            return [(output + bias,)]

    class SelfAttention_Pattern(ms.Pattern):
        """
        %bert_encoder_layer_0_attention_self : [#users=2] = call_module[target=bert.encoder.layer.0.attention.self](args = (%bert_embeddings_dropout, %mul_1, None, None, None, None, False), kwargs = {})
        %getitem_401 : [#users=1] = call_function[target=operator.getitem](args = (%bert_encoder_layer_0_attention_self, 0), kwargs = {})
        %bert_encoder_layer_0_attention_output_dense : [#users=1] = call_module[target=bert.encoder.layer.0.attention.output.dense](args = (%getitem_401,), kwargs = {})
        %getitem_402 : [#users=1] = call_function[target=operator.getitem](args = (%bert_encoder_layer_0_attention_self, slice(1, None, None)), kwargs = {})
        """

        def __init__(self, layer_num):
            super(SelfAttention_Pattern, self).__init__()
            self.layer_num = layer_num

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            return x[0]

        def starting_point(self, op):
            if "layer.{}.attention.self".format(self.layer_num) in op:
                return True
            else:
                return False

    for i in range(12):
        op_lst = sch.find(SelfAttention_Pattern(i))
        for op in sch._ops:
            node = sch._ops[op].node
            if "layer.{}.attention.output.dense".format(i) in node.target:
                op_lst[0].insert(2, node)
                break
        sch[op_lst].replace(SelfAttention, seq=False)

def replace_qkv():
    print("Replace HF QKV Dense with FusedQKV")

    class FusedQKV(nn.Module):
        
        def __init__(self, hidden_size = 768, num_heads = 12) -> None:
            super(FusedQKV, self).__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.fused_linear = nn.Linear(hidden_size * 3, num_heads * self.head_size * 3)

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size, 3)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states): # [8, 512, 768]
            expanded_states = torch.concat((hidden_states, hidden_states, hidden_states), axis=2)
            qkv = self.fused_linear(expanded_states)
            transposed_qkv = self.transpose_for_scores(qkv)
            return [torch.squeeze(t) for t in torch.split(transposed_qkv, 1, dim=-1)]

    class QKV_Pattern(ms.Pattern):
        
        def __init__(self, layer_num):
            super(QKV_Pattern, self).__init__()
            self.layer_num = layer_num

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (12, 768)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def starting_point(self, op):
            if "layer.{}.".format(self.layer_num) in op and "self" in op:
                if "query" in op or "key" in op or "value" in op:
                    return True
            return False

    for i in range(12):
        op_lst = sch.find(QKV_Pattern(i))
        sch[op_lst].replace(FusedQKV, seq=False)

# replace_layernorm()
replace_gelu()
replace_attention()
# replace_qkv()
# print(gm.graph)

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