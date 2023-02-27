# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test custom ops. Note that attn test has to be invoked by torchrun since
most custom ops are for tensor parallelism.
"""
import random

import numpy as np
import pytest
import torch

from slapo import op

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

@pytest.mark.parametrize("op_name", ["cutlass", "cuda", "triton"])
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
        # pylint: disable=redefined-variable-type
        if attn_op_name is None:
            attn = GPT2Attention(config)
        else:
            try:
                attn = op.FlashAttention(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    output_proj=True,
                    attn_pdrop=config.attn_pdrop,
                    resid_pdrop=config.resid_pdrop,
                    attn_op_name=attn_op_name,
                    fused_qkv=True,
                )
            except Exception as err:
                pytest.skip(
                    reason=f"{attn_op_name} is not available in this environment "
                    f"for testing: {err}"
                )
        return attn.half().cuda()

    def _relayout_qkv(gpts_attn, weight):
        # GPT-2 uses contiguous weight layout; while we use interleaved
        # which is more efficient for sharding, so we need to transpose the
        # weights. Bias is all 0's so it is fine.
        weight = weight.view(
            3, gpts_attn.num_heads, gpts_attn.head_dim, gpts_attn.embed_dim
        )
        split_weights = [w.squeeze(0) for w in weight.split(1, dim=0)]
        weight = torch.concatenate(split_weights, dim=1)
        return weight.view(-1, weight.shape[-1]).contiguous()

    # Initialize the attention module.
    attn_ref = _init(None, config)
    attn = _init(op_name, config)

    # Sync parameters. Note that GPT-2 uses fused QKV and Conv1D.
    requires_grad = attn_ref.c_attn.weight.requires_grad
    attn.qkv.weight = torch.nn.Parameter(
        _relayout_qkv(
            attn_ref, attn_ref.c_attn.weight.detach().clone().transpose(1, 0)
        ),
        requires_grad=requires_grad,
    )
    attn.qkv.bias = attn_ref.c_attn.bias
    attn.out_proj.weight = torch.nn.Parameter(
        attn_ref.c_proj.weight.detach().clone().transpose(1, 0).contiguous(),
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
    inputs = [hidden_states, attn_mask, None]
    out = _run_forward_backward(attn, inputs)
    grads = [inp.grad for inp in inputs if inp is not None]

    torch.testing.assert_close(out[0], out_ref[0], atol=5e-2, rtol=5e-2)
    for grad, grad_ref in zip(grads, grads_ref):
        if grad is None:
            # Bias gradient is not supported yet.
            continue
        torch.testing.assert_close(grad, grad_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("shape", [(4, 1024, 2048)])
def test_fused_mlp(shape):
    try:
        from transformers import AutoConfig
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
    except ImportError:
        pytest.skip(reason="transformers not installed")

    batch_size, seq_len, hidden_size = shape
    intermediate_size = hidden_size * 4
    config = AutoConfig.from_pretrained("gpt2-medium")
    config.n_embed = config.hidden_size = hidden_size
    config.resid_pdrop = 0.2
    config.activation_function = "gelu_new"

    def _init(config, ref):
        # pylint: disable=redefined-variable-type
        if ref:
            mlp = GPT2MLP(intermediate_size, config)
        else:
            mlp = op.FusedMLP(
                hidden_size,
                intermediate_size,
                config.activation_function,
                config.resid_pdrop,
            )
        return mlp.half().cuda()

    def _reset_random_seeds():
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)

    # Initialize the mlp module.
    mlp_ref = _init(config, True)
    mlp = _init(config, False)

    # Sync parameters.
    requires_grad = mlp_ref.c_fc.weight.requires_grad
    mlp.fc_in.weight = torch.nn.Parameter(
        mlp_ref.c_fc.weight.detach().clone().transpose(1, 0),
        requires_grad=requires_grad,
    )
    mlp.act.bias = mlp_ref.c_fc.bias
    mlp.fc_out.weight = torch.nn.Parameter(
        mlp_ref.c_proj.weight.detach().clone().transpose(1, 0),
        requires_grad=requires_grad,
    )
    mlp.bias_dropout.bias = mlp_ref.c_proj.bias

    # Generate inputs.
    hidden_states = torch.randn(
        [batch_size, seq_len, hidden_size],
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )

    # Run reference.
    _reset_random_seeds()
    out_ref = _run_forward_backward(mlp_ref, [hidden_states])
    grad_ref = hidden_states.grad

    # Zero out gradients.
    hidden_states.grad = None

    # Run custom op.
    _reset_random_seeds()
    out = _run_forward_backward(mlp, [hidden_states])
    grad = hidden_states.grad

    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(grad, grad_ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    pytest.main([__file__])
