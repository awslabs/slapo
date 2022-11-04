import torch
import torch.nn as nn
import ms


def replace_layernorm(sch):
    print("Replace LayerNorm with FusedLayerNorm")
    from apex.normalization.fused_layer_norm import FusedLayerNorm

    ops = sch.find_module(lambda node: "LayerNorm" in node.target)
    for op in ops:
        sch[op].replace(FusedLayerNorm)


def replace_gelu(sch):
    # https://github.com/NVIDIA/Megatron-LM/blob/master/megatron/model/fused_bias_gelu.py
    print("Replace GeLU with FusedBiasGeLU")
    from ms.op import BiasGeLU

    raise RuntimeError("Not correct! Should fuse with previous linear bias!")
    ops = sch.find_function(lambda node: "gelu" in str(node.target))
    for op in ops:
        sch[op].replace(BiasGeLU)


def replace_xformer_attention(sch):
    # https://github.com/huggingface/transformers/blob/344e2664d450eaa9167ce31f7d1fc0f0fe3b10af/src/transformers/models/bert/modeling_bert.py#L243
    # https://github.com/comaniac/epoi/blob/main/epoi/ops/xformers_attn.py#L45
    print("Replace HF BertSelfAttention with xformer Attention")
    from ms.op import BertSelfAttentionXFormer
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("bert-large-uncased")
    config.hidden_size = 1024
    config.num_attention_heads = 16
    config.intermediate_size = 4096
    config.vocab_size = 30522
    ops = sch.find_module(lambda node: ".attention.self" in node.target)
    for op in ops:
        sch[op].replace(BertSelfAttentionXFormer, config=config)


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
