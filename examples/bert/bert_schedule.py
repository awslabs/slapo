import torch
import torch.nn as nn
import ms


def replace_layernorm(sch):
    print("Replace LayerNorm with FusedLayerNorm")
    from apex.normalization.fused_layer_norm import FusedLayerNorm

    ops = sch.find_module(lambda node: "LayerNorm" in node.target)
    for op in ops:
        sch[op].replace(FusedLayerNorm)


def replace_gelu():
    # https://github.com/NVIDIA/Megatron-LM/blob/master/megatron/model/fused_bias_gelu.py
    print("Replace GeLU with FusedBiasGeLU")
    print(sch.func_ops)
    # sch["gelu"].replace(ms.op.gelu)
    sch["gelu"].replace_module(ms.op.BiasGeLU, half=True)


def replace_xformer_attention():
    # https://github.com/huggingface/transformers/blob/344e2664d450eaa9167ce31f7d1fc0f0fe3b10af/src/transformers/models/bert/modeling_bert.py#L243
    # https://github.com/comaniac/epoi/blob/main/epoi/ops/xformers_attn.py#L45
    print("Replace HF BertSelfAttention with xformer Attention")
    from ms.op.xformers_attn import BertSelfAttentionXFormer

    class SelfAttention_Pattern(ms.Pattern):
        def __init__(self, layer_num):
            super(SelfAttention_Pattern, self).__init__()
            self.layer_num = layer_num

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            return x

        def starting_point(self, node):
            if node.op != "call_module":
                return False
            name = node.target
            if "layer.{}.attention.self".format(self.layer_num) in name:
                return True
            else:
                return False

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("bert-large-uncased")
    config.hidden_size = 1024
    config.num_attention_heads = 16
    config.intermediate_size = 4096
    config.vocab_size = 30522
    sch.trace_module()
    for i in range(12):
        op_lst = sch.find(SelfAttention_Pattern(i))
        sch[op_lst[0][0].name.replace("_", ".")].replace(
            BertSelfAttentionXFormer, config
        )


def replace_attention():
    # https://github.com/huggingface/transformers/blob/344e2664d450eaa9167ce31f7d1fc0f0fe3b10af/src/transformers/models/bert/modeling_bert.py#L243
    # https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/model/transformer.py#L306
    #  MASTER_ADDR=localhost MASTER_PORT=6000 python3 hf_bert.py --micro-batch-size 8 --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --max-position-embeddings 512 --encoder-seq-length 512 --fp16
    print("Replace HF BertSelfAttention with Megatron CoreAttention")
    from megatron.model.transformer import ParallelAttention
    from megatron.model.utils import init_method_normal, scaled_init_method_normal
    from megatron.initialize import initialize_megatron

    initialize_megatron()

    class SelfAttention(nn.Module):
        def __init__(self, layer_number=12):
            super(SelfAttention, self).__init__()
            init_method = init_method_normal(0.006)
            output_layer_init_method = scaled_init_method_normal(0.006, layer_number)
            self.parallel_attention = ParallelAttention(
                init_method, output_layer_init_method, layer_number
            )

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

        def starting_point(self, node):
            if node.op != "call_module":
                return False
            name = node.target
            if "layer.{}.attention.self".format(self.layer_num) in name:
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


def replace_softmax():
    print("Replace HF Softmax with Megatron FusedScaleMaskSoftmax")
    from megatron.model.fused_softmax import FusedScaleMaskSoftmax
    from megatron.model.enums import AttnMaskType
    from megatron.model.utils import attention_mask_func
    import operator

    class Softmax_Pattern(ms.Pattern):
        """
        %truediv: attention_scores
        %mul_1: attention_masks

        %add_7 : [#users=1] = call_function[target=operator.add](args = (%truediv, %mul_1), kwargs = {})
        %softmax : [#users=1] = call_function[target=torch.nn.functional.softmax](args = (%add_7,), kwargs = {dim: -1, _stacklevel: 3, dtype: None})
        """

        def __init__(self):
            super(Softmax_Pattern, self).__init__()

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            return nn.functional.softmax(x)

        def starting_point(self, node):
            if node.op != "call_function":
                return False
            if node.target == operator.add:
                return True
            else:
                return False

    config = {
        "input_in_fp16": True,
        "input_in_bf16": False,
        "attn_mask_type": AttnMaskType.padding,
        "scaled_masked_softmax_fusion": True,
        "mask_func": attention_mask_func,
        "softmax_in_fp32": True,
        "scale": None,
    }
    op_lst = sch.find(Softmax_Pattern())
    for ops in op_lst:
        sch[ops].replace(FusedScaleMaskSoftmax, kwargs=config, seq=True)


def replace_qkv():
    print("Replace HF QKV Dense with FusedQKV")

    class FusedQKV(nn.Module):
        def __init__(self, hidden_size=768, num_heads=12) -> None:
            super(FusedQKV, self).__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.fused_linear = nn.Linear(
                hidden_size * 3, num_heads * self.head_size * 3
            )

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size, 3)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states):  # [8, 512, 768]
            expanded_states = torch.concat(
                (hidden_states, hidden_states, hidden_states), axis=2
            )
            qkv = self.fused_linear(expanded_states)
            transposed_qkv = self.transpose_for_scores(qkv)
            return [torch.squeeze(t) for t in torch.split(transposed_qkv, 1, dim=-1)]

    class QKV_Pattern(ms.Pattern):
        def __init__(self, layer_num):
            super(QKV_Pattern, self).__init__()
            self.layer_num = layer_num

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (16, 1024)  # (12, 768)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def starting_point(self, node):
            if node.op != "call_module":
                return False
            name = node.target
            if "layer.{}.".format(self.layer_num) in name and "self" in name:
                if "query" in name or "key" in name or "value" in name:
                    return True
            return False

    for i in range(24):
        op_lst = sch.find(QKV_Pattern(i))
        sch[op_lst].replace(
            FusedQKV, kwargs={"hidden_size": 1024, "num_heads": 16}, seq=False
        )
