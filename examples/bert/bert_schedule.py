import torch
import torch.nn as nn
import torch.fx as fx
import torch.distributed as dist
import ms


def replace_layernorm(sch):
    from apex.normalization.fused_layer_norm import FusedLayerNorm

    ops = sch.find_module(lambda name: name == "LayerNorm")
    for op in ops:
        sch[op].replace(FusedLayerNorm)


def replace_xformer_attention(sch, config):
    # https://github.com/huggingface/transformers/blob/344e2664d450eaa9167ce31f7d1fc0f0fe3b10af/src/transformers/models/bert/modeling_bert.py#L243
    # https://github.com/comaniac/epoi/blob/main/epoi/ops/xformers_attn.py#L45
    from epoi.inject.policy.bert import InjectHFBertSelfAttentionPolicy
    from epoi.ops.xformers_attn import GenericSelfAttention

    kwargs = InjectHFBertSelfAttentionPolicy.gen_init_config_from_config(config)
    ops = sch.find_module(
        lambda name: ".attention.self_m" in name and "self_m." not in name
    )
    for _, op in enumerate(ops):
        sch[op].replace(GenericSelfAttention, **kwargs)

        parent_name = op[0]
        sch_xformer = sch[f"{parent_name}.GenericSelfAttention_0"].subschedule(
            leaf_modules=["MemoryEfficientAttentionOp"],
            concrete_args={
                "layer_past": None,
                "use_cache": False,
            },
        )

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        class FusedQKV(nn.Module):
            def __init__(self, hidden_size, num_heads) -> None:
                super(FusedQKV, self).__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                self.fused_linear = nn.Linear(
                    hidden_size, num_heads * self.head_size * 3
                )

            def reshape_for_scores(self, x):
                new_x_shape = x.size()[:-1] + (
                    self.num_heads // sch.world_size,
                    self.head_size,
                    3,
                )
                x = x.view(new_x_shape)
                return x.contiguous()

            def forward(self, hidden_states):
                qkv = self.fused_linear(hidden_states)
                reshaped_qkv = self.reshape_for_scores(qkv)
                q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
                q = torch.squeeze(q)
                k = torch.squeeze(k)
                v = torch.squeeze(v)
                return [q, k, v]

        class QKV_Pattern(ms.Pattern):
            def __init__(self):
                super(QKV_Pattern, self).__init__()

            @staticmethod
            def func(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
                x = x.view(new_x_shape)
                return x

            def starting_point(self, _, node):
                return node.op == "call_module" and any(
                    t in node.target for t in ["query", "key", "value"]
                )

        op_lst = sch_xformer.find(QKV_Pattern())
        assert len(op_lst) != 0
        sch_xformer[op_lst].replace(
            FusedQKV, hidden_size=hidden_size, num_heads=num_heads
        )
        if sch.world_size > 1:
            sch_fusedqkv = sch_xformer["FusedQKV_0"].subschedule()
            sch_fusedqkv["fused_linear"].shard("weight", axis=0)
            sch_fusedqkv["fused_linear"].shard("bias", axis=0)
            sch_fusedqkv["fused_linear"].sync(mode="backward")
            fix_number_of_heads(sch_xformer)


def replace_qkv(sch, bert_config):
    hidden_size = bert_config.hidden_size
    num_heads = bert_config.num_attention_heads
    num_layers = bert_config.num_hidden_layers

    class FusedQKV(nn.Module):
        def __init__(self, hidden_size, num_heads) -> None:
            super(FusedQKV, self).__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.fused_linear = nn.Linear(hidden_size, num_heads * self.head_size * 3)

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (
                self.num_heads // sch.world_size,
                self.head_size,
                3,
            )
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states):  # [8, 512, 768]
            qkv = self.fused_linear(hidden_states)
            transposed_qkv = self.transpose_for_scores(qkv)
            q, k, v = torch.split(transposed_qkv, 1, dim=-1)
            q = torch.squeeze(q)
            k = torch.squeeze(k)
            v = torch.squeeze(v)
            return [q, k, v]

    class QKV_Pattern(ms.Pattern):
        def __init__(self, layer_num):
            super(QKV_Pattern, self).__init__()
            self.layer_num = layer_num

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def starting_point(self, parent_name, node):
            return (f"layer.{self.layer_num}.attention" in parent_name) and (
                node.op == "call_module"
                and any(t in node.target for t in ["query", "key", "value"])
            )

    for i in range(num_layers):
        op_lst = sch.find(QKV_Pattern(i))
        assert len(op_lst) != 0
        sch[op_lst].replace(FusedQKV, hidden_size=hidden_size, num_heads=num_heads)


def fix_number_of_heads(sch):
    import operator

    ops = sch.find_method(
        lambda node: node.target == "view"
        and isinstance(node.args[1], fx.Node)  # args[0] is self
        and node.args[1].op == "call_function"
        and node.args[1].target == operator.add
    )

    def new_view(tensor, old_shape):
        new_shape = old_shape[:-1] + (-1,)
        return tensor.view(new_shape)

    for op in ops:
        sch[op].replace(new_view)

    # EPOI attention module uses repeat to process attention mask to
    # align xformer attention mask shape:
    # (B, 1, 1, S) -repeat->  (B, H, S, S) -reshape-> (B x H, S, S),
    # so we need to replace "repeat" wit the sharded H.
    ops = sch.find_method(
        lambda node: node.target == "repeat"
        and len(node.args) == 5  # args[0] is self
        and node.args[1] == 1
        and node.args[-1] == 1
    )

    def new_repeat(tensor, *old_args):
        assert len(old_args) == 4
        new_args = (old_args[0],) + (old_args[1] // sch.world_size,) + old_args[2:]
        return tensor.repeat(*new_args)

    for op in ops:
        sch[op].replace(new_repeat)


def shard_params(sch, config, fused_qkv=False, prefix=""):
    prefix = "" if prefix == "" else prefix + "."

    # Embedding
    sch[prefix + "embeddings.word_embeddings"].shard("weight", axis=0)
    # Build the mask
    vocab_start_index = sch.rank * config.vocab_size // sch.world_size
    vocab_end_index = (sch.rank + 1) * config.vocab_size // sch.world_size

    def fw_pre_hook(_input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    sch[prefix + "embeddings.word_embeddings"].hook("fw_pre", fw_pre_hook)

    def fw_post_hook(_input, output):
        # Mask the output embedding
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[prefix + "embeddings.word_embeddings"].hook("fw_post", fw_post_hook)

    # fmt:off
    for i in range(config.num_hidden_layers):
        # MLP
        sch[prefix+f"encoder.layer.{i}.intermediate.dense"].shard("weight", axis=0)
        sch[prefix+f"encoder.layer.{i}.intermediate.dense"].shard("bias", axis=0)
        sch[prefix+f"encoder.layer.{i}.output.dense"].shard("weight", axis=1)
        sch[prefix+f"encoder.layer.{i}.output.dense"].sync(mode="forward")
        sch[prefix+f"encoder.layer.{i}.intermediate.dense"].sync(mode="backward")

        # Attention
        if fused_qkv is None: # Done sharding in previous opt
            pass
        elif fused_qkv == False:
            sch[prefix+f"encoder.layer.{i}.attention.self_m.query"].shard("weight", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.key"].shard("weight", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.value"].shard("weight", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.query"].shard("bias", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.key"].shard("bias", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.value"].shard("bias", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.query"].sync(mode="backward")
            sch[prefix+f"encoder.layer.{i}.attention.self_m.key"].sync(mode="backward")
            sch[prefix+f"encoder.layer.{i}.attention.self_m.value"].sync(mode="backward")
        else:
            # Method 1
            # sch_fusedqkv = sch[prefix+f"encoder.layer.{i}.attention.self_m.FusedQKV_0"].subschedule()
            # sch_fusedqkv["fused_linear"].shard("weight", axis=0)
            # sch_fusedqkv["fused_linear"].shard("bias", axis=0)
            # sch_fusedqkv["fused_linear"].sync(mode="backward")
            # Method 2
            sch[prefix+f"encoder.layer.{i}.attention.self_m.FusedQKV_0.fused_linear"].shard("weight", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.FusedQKV_0.fused_linear"].shard("bias", axis=0)
            sch[prefix+f"encoder.layer.{i}.attention.self_m.FusedQKV_0.fused_linear"].sync(mode="backward")

        sch[prefix+f"encoder.layer.{i}.attention.output.dense"].shard("weight", axis=1)
        sch[prefix+f"encoder.layer.{i}.attention.output.dense"].sync(mode="forward")
    # fmt:on

    fix_number_of_heads(sch)


def checkpoint(sch, config, prefix="", ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return
    prefix = "" if prefix == "" else prefix + "."

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    for i in range(n_ckpt):
        sch[prefix + f"encoder.layer.{i}"].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast_input(inputs):
        for t in inputs:
            dist.broadcast(t, src=0, group=sch.group)
        return inputs

    sch[""].hook("fw_pre", broadcast_input)


def shard_loss(sch, config):
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # outputs: (N, C)
    # labels: (C)
    from ms.op.cross_entropy import ParallelCrossEntropy
    sch["crossentropyloss_0"].replace(ParallelCrossEntropy)
