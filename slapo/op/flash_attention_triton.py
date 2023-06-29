"""
Copy from:
https://github.com/jfc4050/fast-ops/blob/main/fast_ops/flash_attention/flash_attention_triton.py

*Experimental* implementation of FlashAttention in Triton.
There's still some unresolved race conditions for certain input types.
In particular there's issues if:
- seq_len % 128 != 0
- head_dim != 128 
before using this please double check the unit tests to make sure your use case
is covered.

We use the FlashAttention implementation from Tri Dao
(https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
as a starting point, who in turn used the FlashAttention implementation from Phil Tillet
(https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py)
as a starting point.

This version adds dropout support and does some additional optimization, at the
cost of breaking some input cases that worked in Tri Dao's implementation.

TODOs:
* upgrade triton version
* block-sparse attention
* NestedTensor support
* bias gradient support
"""

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function

import triton
import triton.language as tl


@triton.jit
def make_dropout_mask(dropout_p, dropout_seed, indices):
    """integer hashing function rather than using philox PRNG. ends up being a lot faster
    and still passes my statistical tests.
    see http://burtleburtle.net/bob/hash/integer.html
    """
    a = indices.to(tl.uint32, bitcast=True)
    a ^= dropout_seed.to(tl.uint32)  # to introduce dependency on seed
    a ^= a >> 4
    a = (a ^ 0xDEADBEEF) + (a << 5)
    a ^= a >> 11

    return a > (dropout_p * 4294967295).to(tl.uint32)


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    dropout_p,
    rng_seed,
    rng_offset,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    USE_DROPOUT: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    Q = Q + off_b * stride_qb + off_h * stride_qh
    K = K + off_b * stride_kb + off_h * stride_kh
    V = V + off_b * stride_vb + off_h * stride_vh
    Out = Out + off_b * stride_ob + off_h * stride_oh
    TMP = TMP + off_hb * seqlen_q_rounded
    Lse = Lse + off_hb * seqlen_q_rounded
    if BIAS_TYPE != "none":
        Bias = Bias + off_b * stride_bb + off_h * stride_bh
    if USE_DROPOUT:
        dropout_rng_offset_hb = rng_offset.to(off_hb.dtype) + (off_hb * seqlen_q * seqlen_k)

    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_iter = start_n + tl.arange(0, BLOCK_N)
        # -- compute qk ----
        k_ptrs = K + (offs_n_iter[:, None] * stride_kn + offs_d[None, :])
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs)
            else:
                k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs, mask=offs_n_iter[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(
                    k_ptrs,
                    mask=(offs_n_iter[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where(offs_n_iter[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= offs_n_iter[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                b_ptrs = Bias + offs_n_iter
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n_iter < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                b_ptrs = Bias + (offs_m[:, None] * stride_bm + offs_n_iter[None, :])
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m[:, None] < seqlen_q) & (offs_n_iter[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        t_ptrs = TMP + offs_m
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        # apply dropout
        if USE_DROPOUT:
            indices = dropout_rng_offset_hb + (offs_m[:, None] * seqlen_k + offs_n_iter[None, :])
            dropout_mask = make_dropout_mask(dropout_p, rng_seed, indices)
            p *= tl.where(dropout_mask, 1.0 / (1.0 - dropout_p), 0.0)

        # update acc_o
        v_ptrs = V + (offs_n_iter[:, None] * stride_vn + offs_d[None, :])
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs)
            else:
                v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs, mask=offs_n_iter[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(
                    v_ptrs,
                    mask=(offs_n_iter[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    t_ptrs = TMP + offs_m
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    tl.store(Lse + offs_m, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "USE_DROPOUT",
        "BIAS_TYPE",
        "IS_CAUSAL",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    dropout_p,
    rng_seed,
    rng_offset,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    USE_DROPOUT: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    dropout_rng_offset_hb = rng_offset + (off_hb * seqlen_q * seqlen_k)
    start_n = tl.program_id(0)

    # initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # load bias if loop-invariant
    if BIAS_TYPE == "vector":
        # vector bias is same over all values of m, so
        # can be loaded once and kept in SRAM over all iterations
        b_ptrs = Bias + offs_n
        if EVEN_N:
            bias = tl.load(b_ptrs).to(tl.float32)
        else:
            bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
        bias = bias[None, :]
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    k_ptrs = K + ((offs_n * stride_kn)[:, None] + offs_d[None, :])
    v_ptrs = V + ((offs_n * stride_vn)[:, None] + offs_d[None, :])
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )

    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # loop over rows
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    end_m = tl.cdiv(seqlen_q, BLOCK_M) * BLOCK_M
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + tl.arange(0, BLOCK_M)
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)

        q_ptrs = Q + (offs_m_curr * stride_qm)[:, None] + offs_d[None, :]
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        # recompute p = softmax(qk, dim=-1).T
        s = softmax_scale * tl.dot(q, k, trans_b=True)
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            s = tl.where(offs_n[None, :] < seqlen_k, s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), s, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                pass  # already loaded before entering loop
            elif BIAS_TYPE == "matrix":
                b_ptrs = Bias + (offs_m_curr * stride_bm)[:, None] + offs_n[None, :]
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            s += bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(s - lse_i[:, None])

        if USE_DROPOUT:
            # compute Zij (sort of, see below). has values:
            #   1 / (1 - p) with probability 1 - p
            #   0 with probability p

            indices = (
                dropout_rng_offset_hb.to(offs_m_curr.dtype)
                + (offs_m_curr * seqlen_k)[:, None]
                + offs_n[None, :]
            )

            # don't need to materialize Zij, just store drop bit into sign bit of Pij
            # (trick stolen from CUDA implementation) to save SRAM/registers
            dropout_mask = make_dropout_mask(dropout_p, rng_seed, indices)
            p *= tl.where(dropout_mask, 1.0, -1.0)

        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        do_ptrs = DO + (offs_m_curr * stride_dom)[:, None] + offs_d[None, :]
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

        if USE_DROPOUT:
            p_dropped = tl.where(dropout_mask, p * (1.0 / (1.0 - dropout_p)), 0.0)
            dv += tl.dot(p_dropped.to(do.dtype), do, trans_a=True).to(dv.dtype)
        else:
            dv += tl.dot(p.to(do.dtype), do, trans_a=True).to(dv.dtype)

        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256)
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, v, trans_b=True)
        # we don't need to explicitly compute dPij from dPij_dropped, see below where
        # we compute dSij for more details.

        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)

        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        if USE_DROPOUT:
            # for computing dSij = tau * Pij * (dPij - Di)
            # we have:
            #   - Pij' which has negative values where Zij = 0, otherwise same as Pij
            #   - dPij' = dPij_dropped
            #
            # IF KEEPING:
            #   Pij = Pij'
            #   dPij = dPij'
            #
            #   Pij * (dPij - Di) = Pij' * (dPij' - Di)
            #                     = Pij' * ((dPij' * (1 / (1 - p))) - Di)
            #
            # IF DROPPING:
            #   Pij = -Pij'
            #   dPij = 0
            #
            #   Pij * (dPij - Di) = Pij * (0 - Di)
            #                     = Pij * -Di
            #                     = -Pij * Di
            #                     = Pij' * Di
            #   note earlier we flipped sign of p if element was to be dropped
            ds = (
                softmax_scale
                * p
                * tl.where(p > 0.0, (dp * (1.0 / (1.0 - dropout_p))) - Di[:, None], Di[:, None])
            )
            ds = ds.to(q.dtype)
        else:
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(ds, q, trans_a=True).to(dk.dtype)
        # compute dq
        dq_ptrs = DQ + (offs_m_curr * stride_dqm)[:, None] + offs_d[None, :]
        dq = tl.dot(ds, k)
        if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
            tl.atomic_add(dq_ptrs, dq)
        else:
            if EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
            else:
                tl.atomic_add(
                    dq_ptrs,
                    dq,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                )
    # write-back
    dv_ptrs = DV + ((offs_n * stride_dvn)[:, None] + offs_d[None, :])
    dk_ptrs = DK + ((offs_n * stride_dkn)[:, None] + offs_d[None, :])
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


def increment_philox_state(increment: int) -> tuple:
    """
    1. extract the current rng seed and offset
    2. increment the offset
    3. then set the new state

    layout of state tensor is as follows:
    [states (200 * 4bytes), seed (uint64_t - 8 bytes), offset (uint64_t - 8 bytes)]
    see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L150-L176
    """
    rng_state = torch.cuda.get_rng_state()
    # reinterpret bytes as uint64_t
    # (not all of the bytes are supposed to be uint64_t but we only care about reading
    # the last 16 bytes)
    rng_state_array_as_uint64 = rng_state.numpy().view(dtype=np.uint64)

    seed = rng_state_array_as_uint64[-2]
    offset = rng_state_array_as_uint64[-1]

    # increment offset (needs to be multiple of 4)
    rng_state_array_as_uint64[-1] += int(math.ceil(increment / 4)) * 4

    torch.cuda.set_rng_state(rng_state)

    return int(seed), int(offset)


@triton.jit
def _dropout_mask_kernel(
    tensor,
    dropout_p,
    seqlen_q,
    seqlen_k,
    seed,
    rng_seq_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, seqlen_k, BLOCK_N):
        indices = (
            (off_hb * seqlen_q * seqlen_k)
            + (offs_m * seqlen_k)[:, None]
            + (start_n + offs_n)[None, :]
        )
        dropout_mask = make_dropout_mask(dropout_p, seed, (rng_seq_offset + indices).to(tl.int32))
        tl.store(
            tensor + indices,
            dropout_mask,
            mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
        )


def triton_dropout_mask(
    dropout_p: float,
    batch_sz: int,
    n_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    seed, rng_offset = increment_philox_state(batch_sz * n_heads * seqlen_q * seqlen_k)

    tensor = torch.empty((batch_sz, n_heads, seqlen_q, seqlen_k), dtype=torch.int32, device=device)

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch_sz * n_heads)
    BLOCK = 128
    _dropout_mask_kernel[grid](
        tensor, dropout_p, seqlen_q, seqlen_k, seed, rng_offset, BLOCK, BLOCK
    )

    return tensor.to(torch.bool)


class FlashAttnTritonFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor = None,
        causal: bool = False,
        dropout_p: float = 0.0,
        softmax_scale: float = None,
    ) -> Tensor:
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        assert d <= 128, "FlashAttention only support head dimensions up to 128"
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

        has_bias = bias is not None
        bias_type = "none"
        if has_bias:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            if bias.stride(-1) != 1:
                bias = bias.contiguous()
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = "vector"
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = "matrix"
            else:
                raise RuntimeError(
                    "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
                )
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

        lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        BLOCK = 128
        num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

        rng_seed, rng_offset = increment_philox_state(batch * nheads * seqlen_q * seqlen_k)

        _fwd_kernel[grid](
            q,
            k,
            v,
            bias,
            o,
            lse,
            tmp,
            softmax_scale,
            dropout_p,
            rng_seed,
            rng_offset,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            *bias_strides,
            o.stride(0),
            o.stride(2),
            o.stride(1),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            d,
            seqlen_q // 32,
            seqlen_k // 32,  # key for triton cache (limit number of compilations)
            dropout_p > 0.0,  # USE_DROPOUT
            bias_type,
            causal,
            BLOCK_HEADDIM,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        ctx.softmax_scale = softmax_scale  # softmax scale could have been updated
        ctx.dropout_seed = rng_seed
        ctx.dropout_offset = rng_offset
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        ctx.dropout_p = dropout_p

        return o

    @staticmethod
    def backward(ctx, do: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        q, k, v, o, lse, bias = ctx.saved_tensors
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        assert seqlen_q % 128 == 0
        assert seqlen_k % 128 == 0
        assert d == 128
        assert do.shape == q.shape
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == do.stride(-1) == 1
        assert lse.shape == (batch, nheads, seqlen_q_rounded)
        assert q.dtype in {torch.float16, torch.bfloat16}
        assert q.dtype == k.dtype == v.dtype == do.dtype

        bias: Tensor
        has_bias = bias is not None
        bias_type = "none"
        if has_bias:
            assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            assert bias.stride(-1) == 1
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = "vector"
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = "matrix"
            else:
                raise ValueError(
                    "Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)"
                )
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)

        softmax_scale = ctx.softmax_scale or 1.0 / math.sqrt(d)

        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq_accum = torch.empty_like(q, dtype=torch.float32)
            delta = torch.empty_like(lse)

            BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
            grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
            _bwd_preprocess_do_o_dot[grid](
                o,
                do,
                delta,
                o.stride(0),
                o.stride(2),
                o.stride(1),
                do.stride(0),
                do.stride(2),
                do.stride(1),
                nheads,
                seqlen_q,
                seqlen_q_rounded,
                d,
                BLOCK_M=128,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
            )

            bias_strides = (
                (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
            )
            grid = lambda META: (
                triton.cdiv(seqlen_k, META["BLOCK_N"]),
                batch * nheads,
            )
            _bwd_kernel[grid](
                q,
                k,
                v,
                bias,
                do,
                dq_accum,
                dk,
                dv,
                lse,
                delta,
                softmax_scale,
                ctx.dropout_p,
                ctx.dropout_seed,
                ctx.dropout_offset,
                q.stride(0),
                q.stride(2),
                q.stride(1),
                k.stride(0),
                k.stride(2),
                k.stride(1),
                v.stride(0),
                v.stride(2),
                v.stride(1),
                *bias_strides,
                do.stride(0),
                do.stride(2),
                do.stride(1),
                dq_accum.stride(0),
                dq_accum.stride(2),
                dq_accum.stride(1),
                dk.stride(0),
                dk.stride(2),
                dk.stride(1),
                dv.stride(0),
                dv.stride(2),
                dv.stride(1),
                nheads,
                seqlen_q,
                seqlen_k,
                seqlen_q_rounded,
                d,
                seqlen_q // 32,
                seqlen_k // 32,  # key for triton cache (limit number of compilations)
                ctx.dropout_p > 0.0,  # USE_DROPOUT
                bias_type,
                ctx.causal,
                BLOCK_HEADDIM,
            )
            dq.copy_(dq_accum)

        return dq, dk, dv, None, None, None, None


flash_attn_triton = FlashAttnTritonFunction.apply