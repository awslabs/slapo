# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test custom ops. Note that attn test has to be invoked by torchrun since
most custom ops are for tensor parallelism.
"""
# pylint: disable=unused-argument
import os

import pytest
import torch
from torch import distributed as dist
from torch import nn

from slapo import op
from slapo.random import get_cuda_rng_tracker, set_random_seed


def test_dropout(init_dist):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    data = torch.rand(10, 10).cuda(local_rank)
    dist.broadcast(data, src=0)

    get_cuda_rng_tracker().reset()

    # The custom dropout should throw error if set_random_seed is not called.
    with pytest.raises(Exception):
        op.DropoutWithTensorParallel(p=0.5)(data)

    set_random_seed(123, tp_rank=local_rank)

    # Assuming all devices are in the same TP group, the native dropout
    # should produce the same output on all devices.
    out = nn.Dropout(p=0.5)(data)
    out_reduced = out.clone()
    dist.all_reduce(out_reduced)
    torch.testing.assert_close(
        out * world_size,
        out_reduced,
        msg=lambda msg: f"output mismatch\n{msg}",
    )

    # The custom dropout should produce different outputs on different devices
    # even they are in the same TP group.
    out = op.DropoutWithTensorParallel(p=0.5)(data)
    out_reduced = out.clone()
    dist.all_reduce(out_reduced)
    with pytest.raises(Exception):
        torch.testing.assert_close(out * world_size, out_reduced)


@pytest.mark.parametrize("op_name", ["cutlass", "triton"])
@pytest.mark.parametrize("shape", [(4, 1024, 2048, 16, 50264)])
def test_attention(op_name, shape):
    try:
        from transformers import AutoConfig
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    except ImportError:
        pytest.skip(reason="transformers not installed")

    batch_size, seq_len, hidden_size, num_heads, vocab_size = shape
    config = AutoConfig.from_pretrained("gpt2-medium")
    config.max_position_embeddings = seq_len
    config.n_embed = config.hidden_size = hidden_size
    config.n_head = config.num_attention_heads = num_heads
    config.vocab_size = vocab_size

    # Disable dropout for correctness checking.
    config.attn_pdrop = 0.0
    config.resid_pdrop = 0.0

    def _init(attn_op_name, config):
        if attn_op_name is None:
            attn = GPT2Attention(config)
        else:
            try:
                attn = op.FlashSelfAttention(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    is_decoder=True,
                    attn_pdrop=config.attn_pdrop,
                    resid_pdrop=config.resid_pdrop,
                    attn_op_name=attn_op_name,
                    fused_qkv=True,
                )
            except Exception as err:
                pytest.skip(
                    reason=f"{attn_op_name} is not available in mlp environment "
                    f"for testing: {err}"
                )
        return attn.half().cuda()

    def _run_forward_backward(func, inputs):
        outs = func(*inputs)
        target_out = []
        target_grad = []
        for out in outs:
            if out is not None:
                target_out.append(out)
                target_grad.append(torch.ones_like(out))
        torch.autograd.backward(target_out, target_grad)
        torch.cuda.synchronize()
        return outs

    # Initialize the attention module.
    attn_ref = _init(None, config)
    attn = _init(op_name, config)

    # Sync parameters. Note that GPT-2 uses fused QKV and Conv1D.
    requires_grad = attn_ref.c_attn.weight.requires_grad
    attn.qkv.weight = torch.nn.Parameter(
        attn_ref.c_attn.weight.transpose(-1, 0).contiguous(),
        requires_grad=requires_grad,
    )
    attn.qkv.bias = attn_ref.c_attn.bias
    attn.out_proj.weight = torch.nn.Parameter(
        attn_ref.c_proj.weight.transpose(-1, 0).contiguous(),
        requires_grad=requires_grad,
    )
    attn.out_proj.bias = attn_ref.c_proj.bias

    # Generate inputs.
    hidden_states = torch.randn(
        [batch_size, seq_len, hidden_size],
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )
    attn_mask = torch.randn(
        [batch_size, 1, 1, seq_len],
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )
    inputs = [hidden_states, None, attn_mask]

    # Run reference.
    out_ref = _run_forward_backward(attn_ref, inputs)
    grads_ref = [inp.grad for inp in inputs if inp is not None]

    # Zero out gradients.
    for inp in inputs:
        if inp is not None:
            inp.grad = None

    # Run custom op.
    out = _run_forward_backward(attn, inputs)
    grads = [inp.grad for inp in inputs if inp is not None]

    torch.testing.assert_close(out[0], out_ref[0], atol=5e-2, rtol=5e-2)
    for grad, grad_ref in zip(grads, grads_ref):
        if grad is None:
            # Bias gradient is not supported yet.
            continue
        torch.testing.assert_close(grad, grad_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("act_name", ["gelu", "gelu_new"])
@pytest.mark.parametrize("shape", [(8, 512, 1024)])
def test_mlp(act_name, shape):
    try:
        from transformers import AutoConfig
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
    except ImportError:
        pytest.skip(reason="transformers not installed")

    batch_size, seq_len, hidden_size = shape
    config = AutoConfig.from_pretrained("gpt2-medium")
    config.max_position_embeddings = seq_len
    config.n_embed = config.hidden_size = hidden_size
    config.resid_pdrop = 0.0
    config.activation_function = act_name
    intermediate_size = 4 * hidden_size

    def _run(func, hidden_states):
        out = func(hidden_states)
        torch.autograd.backward(out, torch.ones_like(out))
        torch.cuda.synchronize()
        grad = hidden_states.grad
        hidden_states.grad = None
        return out, grad

    # Initialize the MLP module.
    mlp_ref = GPT2MLP(intermediate_size, config).half().cuda()
    mlp = op.FusedMLP(
        config.hidden_size,
        intermediate_size,
        config.activation_function,
        config.resid_pdrop,
    )
    mlp = mlp.half().cuda()

    # Sync parameters. Note that GPT-2 uses Conv1D with transposed weights.
    requires_grad = mlp_ref.c_fc.weight.requires_grad
    mlp.fc_in.weight = torch.nn.Parameter(
        mlp_ref.c_fc.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
    )
    mlp.act.bias = mlp_ref.c_fc.bias
    mlp.fc_out.weight = torch.nn.Parameter(
        mlp_ref.c_proj.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
    )
    mlp.fc_out.bias = mlp_ref.c_proj.bias

    # Generate inputs.
    hidden_states = torch.randn(
        [batch_size, seq_len, hidden_size],
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )

    # Run reference.
    out_ref, grad_ref = _run(mlp_ref, hidden_states)
    out, grad = _run(mlp, hidden_states)

    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(grad, grad_ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    pytest.main([__file__])
