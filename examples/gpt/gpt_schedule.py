import re

import torch
import torch.nn as nn
import torch.distributed as dist
import ms


def replace_qkv(sch, num_layers, num_heads, hidden_size):
    class FusedQKV(nn.Module):
        def __init__(self, hidden_size, num_heads) -> None:
            super(FusedQKV, self).__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.fused_linear = nn.Linear(hidden_size, num_heads * self.head_size * 3)

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (
                self.num_heads // dist.get_world_size(),
                self.head_size,
                3,
            )
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states):
            qkv = self.fused_linear(hidden_states)
            transposed_qkv = self.transpose_for_scores(qkv)
            q, k, v = torch.split(transposed_qkv, 1, dim=-1)
            q = torch.squeeze(q)
            k = torch.squeeze(k)
            v = torch.squeeze(v)
            return [q, k, v]

    class QKVPattern(ms.Pattern):
        def __init__(self, layer_num):
            super(QKVPattern, self).__init__()
            self.layer_num = layer_num

        @staticmethod
        def func(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def starting_point(self, node):
            if node.op != "call_module":
                return False
            name = node.target
            if f"h.{self.layer_num}." in name and "attention" in name:
                if "k_proj" in name or "q_proj" in name or "v_proj" in name:
                    return True
            return False

    cnt = 0
    for i in range(num_layers):
        op_lst = sch.find(QKVPattern(i))
        assert op_lst, "Cannot find QKV pattern"
        sch[op_lst].replace(FusedQKV, hidden_size=hidden_size, num_heads=num_heads)
        cnt += 1
    print(f"Replace {cnt} QKV patterns")


def replace_attention(sch, config, attn_path="h.N.attn.attention"):
    # https://github.com/huggingface/transformers/blob/344e2664d450eaa9167ce31f7d1fc0f0fe3b10af/src/transformers/models/bert/modeling_bert.py#L243
    # https://github.com/comaniac/epoi/blob/main/epoi/ops/xformers_attn.py#L45
    from ms.op import GenericSelfAttention

    def find_dropout_prob(config_or_mod):
        """A helper function to find the dropout probability
        of GPT models. config_or_mod could either be a model config, or an attention module.
        This function supports GPT-2, GPT-Neo and GPT-J implementations.
        """
        if hasattr(config_or_mod, "attention_dropout"):
            attn_pdrop = config_or_mod.attention_dropout
        elif hasattr(config_or_mod, "attn_pdrop"):
            attn_pdrop = config_or_mod.attn_pdrop
        elif hasattr(config_or_mod, "attn_dropout"):
            attn_pdrop = config_or_mod.attn_dropout
        else:
            raise ValueError("Cannot find attention dropout probability")

        if hasattr(config_or_mod, "resid_pdrop"):
            resid_pdrop = config_or_mod.resid_pdrop
        elif hasattr(config_or_mod, "resid_dropout"):
            resid_pdrop = config_or_mod.resid_dropout
        elif hasattr(config_or_mod, "resid_dropout"):
            resid_pdrop = config_or_mod.resid_dropout
        else:
            raise ValueError("Cannot find resid_pdrop or resid_dropout in config.")

        attn_pdrop = attn_pdrop.p if hasattr(attn_pdrop, "p") else attn_pdrop
        resid_pdrop = resid_pdrop.p if hasattr(resid_pdrop, "p") else resid_pdrop

        return attn_pdrop, resid_pdrop

    # Need to remove useless code first. TODO: Apply this pass implicitly.
    sch.gm.graph.eliminate_dead_code()

    # Generate arguments for xformer attention.
    attn_pdrop, resid_pdrop = find_dropout_prob(config)
    new_args = {
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "is_decoder": True,
        "attn_pdrop": attn_pdrop,
        "resid_pdrop": resid_pdrop,
        "attn_op_name": "cutlass",
        "fused_qkv": False,  # Fuse later.
    }

    attn_pat = attn_path.replace("N", "\d+")
    ops = sch.find_module(lambda node: bool(re.search(attn_pat, node.target)))
    for op in ops:
        new_op = sch[op].replace(GenericSelfAttention, **new_args)
        sch_xformer = sch[new_op].subschedule(
            leaf_modules=["MemoryEfficientAttention"],
            concrete_args={
                "layer_past": None,
                "use_cache": False,
            },
        )
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        class FusedQKV(nn.Module):
            def __init__(self, hidden_size, num_heads) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                self.fused_linear = nn.Linear(hidden_size, self.num_heads * self.head_size * 3)

            def reshape_for_scores(self, x):
                new_x_shape = x.size()[:-1] + (self.num_heads // sch.world_size, self.head_size, 3)
                x = x.view(new_x_shape)
                return x.contiguous()

            def forward(self, hidden_states):
                qkv = self.fused_linear(hidden_states)
                reshaped_qkv = self.reshape_for_scores(qkv)
                q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
                q = torch.squeeze(q).contiguous()
                k = torch.squeeze(k).contiguous()
                v = torch.squeeze(v).contiguous()
                return [q, k, v]

        class QKVPattern(ms.Pattern):
            def __init__(self):
                super().__init__()

            @staticmethod
            def func(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
                x = x.view(new_x_shape)
                return x

            def starting_point(self, node):
                if node.op != "call_module":
                    return False
                name = node.target
                return "query" in name or "key" in name or "value" in name

        op_lst = sch_xformer.find(QKVPattern())
        assert len(op_lst) != 0
        sch_xformer[op_lst].replace(FusedQKV, hidden_size=hidden_size, num_heads=num_heads)
        if sch.world_size > 1:
            sch_fusedqkv = sch_xformer["FusedQKV_0"].subschedule()
            sch_fusedqkv["fused_linear"].shard("weight", axis=0)
            sch_fusedqkv["fused_linear"].shard("bias", axis=0)
            sch_fusedqkv["fused_linear"].sync(backward=True)
            sch_xformer["FusedQKV_0"].compose(sch_fusedqkv)
            sch_xformer["out_proj"].shard("weight", axis=1)
            sch_xformer["out_proj"].sync()

        sch[new_op].compose(sch_xformer)
    print(f"Replace {len(ops)} attention patterns")


def replace_softmax(sch):
    print("Replace HF Softmax with Megatron FusedScaleMaskSoftmax")
    from megatron.model.fused_softmax import FusedScaleMaskSoftmax
    from megatron.model.enums import AttnMaskType
    from megatron.model.utils import attention_mask_func
    import operator

    class SoftmaxPattern(ms.Pattern):
        """
        %truediv: attention_scores
        %mul_1: attention_masks
        %add_7 = call_function[target=operator.add](args = (%truediv, %mul_1), ...)
        %softmax = call_function[target=torch.nn.functional.softmax](args = (%add_7,), ...)
        FIXME: GPT-Neo processes attention mask in the beginning of the model,
        so this pattern should also cover it:

        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        ...
        attn_weight = attn_weight + attention_mask
        attn_weight = softmax(attn_weight)
        """

        def __init__(self):
            super().__init__()

        @staticmethod
        def func(attn_weights: torch.Tensor, causal_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
            attn_weights = attn_weights + attention_mask
            return nn.functional.softmax(attn_weights)

        def starting_point(self, node):
            return node.op == "call_function" and node.target == operator.getitem and node.args[0].op == "get_attr" and "h." in node.args[0].target and ".attn.attention.bias" in node.args[0].target

    config = {
        "input_in_fp16": True,
        "input_in_bf16": False,
        "attn_mask_type": AttnMaskType.padding,
        "scaled_masked_softmax_fusion": True,
        "mask_func": attention_mask_func,
        "softmax_in_fp32": True,
        "scale": None,
    }
    ops = sch.find(SoftmaxPattern())
    print(ops)
    sys.exit()
    for op in ops:
        sch[op].replace(FusedScaleMaskSoftmax, **config)
    print(f"Replace {len(ops)} softmax ops")


def remove_cast(sch):
    """Remove .to(torch.float32) in GPT-Neo attention to align
    HF and Megatron GPT-2 behavior.
    """
    ops = sch.find_method(
        lambda node: node.op == "call_method"
        and node.target == "to"
        and len(node.args) == 2
        and node.args[1] == torch.float32
    )

    for op in ops:
        sch[op].replace(lambda x, *args: x)
    print(f"Remove {len(ops)} .to(torch.float32) ops")


def shard_word_embedding(sch, vocab_size, word_embed_name="wte"):
    if sch.world_size == 1:
        return

    # Embedding
    sch[word_embed_name].shard("weight", axis=0)
    # Build the mask
    vocab_start_index = sch.rank * vocab_size // sch.world_size
    vocab_end_index = (sch.rank + 1) * vocab_size // sch.world_size

    def fw_pre_hook(_input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    sch[word_embed_name].hook("fw_pre", fw_pre_hook)

    def fw_post_hook(_input, output):
        # Mask the output embedding
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    sch[word_embed_name].hook("fw_post", fw_post_hook)


def shard_qkv(
    sch,
    num_layers,
    path="h.N.attn.attention",
    qkv_name="FusedQKV_0",
    out_proj_name="out_proj",
    transpose_weights=False,
):
    def fix_shape_after_shard():
        # Fix shape of view ops after sharding.
        import operator

        ops = sch.find_method(
            lambda node: node.op == "call_method"
            and node.target == "view"
            and len(node.args) == 2
            and node.args[0].target == "contiguous"
            and isinstance(node.args[1], torch.fx.Node)
            and node.args[1].target == operator.add
        )

        def new_view(tensor, old_shape):
            new_shape = old_shape[:-1] + (-1,)
            return tensor.view(new_shape)

        for op in ops:
            sch[op].replace(new_view)
        print(f"Fix {len(ops)} view ops after sharding")

    axes = [1, 0] if transpose_weights else [0, 1]
    for i in range(num_layers):
        prefix = path.replace("N", str(i))
        # TODO: Implicitly call create_schedule and replace.
        sch_fusedqkv = ms.create_schedule(
            sch.get_module(f"{prefix}.{qkv_name}"),
            world_size=sch.world_size,
            rank=sch.rank,
            tracer="pytorch",
        )
        sch_fusedqkv["fused_linear"].shard("weight", axis=axes[0])
        sch_fusedqkv["fused_linear"].shard("bias", axis=0)
        sch_fusedqkv["fused_linear"].sync(backward=True)
        opt_fusedqkv, _ = ms.build(sch_fusedqkv)
        sch[f"{prefix}.{qkv_name}"].replace(opt_fusedqkv)

        sch[f"{prefix}.{out_proj_name}"].shard("weight", axis=axes[1])
        sch[f"{prefix}.{out_proj_name}"].sync()
    fix_shape_after_shard()


def replace_and_shard_mlp(
    sch, config, path="h.N.mlp", fc_names=["c_fc", "c_proj"], transpose_weights=False
):
    from ms.op import FusedBiasAct

    # Need to remove useless code first. TODO: Apply this pass implicitly.
    sch.gm.graph.eliminate_dead_code()

    act = config.activation_function
    if FusedBiasAct.check_act_type(act):

        class FusedMLP(nn.Module):
            """A wrapper MLP to make use of fused bias+new_gelu."""

            def __init__(self, hidden_size, intermediate_size, resid_pdrop):
                super().__init__()
                self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.act = FusedBiasAct(intermediate_size, act=act, prev_weight=self.fc_in.weight)
                self.fc_out = nn.Linear(intermediate_size, hidden_size)
                self.dropout = nn.Dropout(resid_pdrop)

            def forward(self, hidden_states):
                hidden_states = self.fc_in(hidden_states)
                hidden_states = self.act(hidden_states)
                hidden_states = self.fc_out(hidden_states)
                hidden_states = self.dropout(hidden_states)
                return hidden_states

        inner_dim = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * config.hidden_size
        )
        new_args = {
            "hidden_size": config.hidden_size,
            "intermediate_size": inner_dim,
            "resid_pdrop": config.resid_dropout,
        }

        path = path.replace("N", "\d+")
        ops = sch.find_module(lambda node: bool(re.search(path, node.target)))
        for op in ops:
            new_op = sch[op].replace(FusedMLP, **new_args)
            sch_mlp = sch[new_op].subschedule(leaf_modules=["FusedBiasAct"])
            if sch.world_size > 1:
                sch_mlp["fc_in"].shard("weight", axis=0)
                sch_mlp["act"].shard("bias", axis=0)
                sch_mlp["fc_in"].sync(backward=True)
                sch_mlp["fc_out"].shard("weight", axis=1)
                sch_mlp["fc_out"].sync()
            sch[new_op].compose(sch_mlp)
    elif sch.world_size > 1:
        axes = [1, 0] if transpose_weights else [0, 1]
        for i in range(config.num_layers):
            prefix = path.replace("N", str(i))
            sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=axes[0])
            sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
            sch[f"{prefix}.{fc_names[0]}"].sync(backward=True)
            sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=axes[1])
            sch[f"{prefix}.{fc_names[1]}"].sync()
