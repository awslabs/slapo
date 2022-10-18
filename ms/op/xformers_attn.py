from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D, prune_conv1d_layer

try:
    import xformers
    import xformers.ops
except ImportError:
    xformers = None


def pt_attention(q, k, v, attn_bias, p=0.0):
    """The native PyTorch implementation of attention with the same signature as the
    FlashAttention implemented in xformers. This is used mainly to check the correctness
    of the xformers implementation, so do not change the functionality of this function.
    """

    def attention_bmk(q, k, v, attn_bias=None, p=0.0):
        # if isinstance(attn_bias, xformers.ops.AttentionMask):
        #     attn_bias = attn_bias.to_tensor().to(q.dtype)
        q = q * (1.0 / q.shape[-1] ** 0.5)
        if attn_bias is None:
            attn = q @ k.transpose(-2, -1)
        else:
            # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
            # but faster, and is what is used in PyTorch now
            attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
        attn = attn.softmax(-1)
        if p > 0:
            attn = torch.nn.functional.dropout(attn, p=p)
        return attn @ v

    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

    out = attention_bmk(T(q), T(k), T(v), attn_bias, p)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


class BertSelfAttentionXFormer(nn.Module):
    """Modified from HuggingFace's BertSelfAttention to use the xformers attention op"""

    def __init__(self, config, position_embedding_type=None, attn_op_name=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            raise NotImplementedError("Not implemented")

        self.is_decoder = config.is_decoder

        if attn_op_name is None:
            self.attn_op = pt_attention
        else:
            assert xformers is not None, "xformers is not installed"

            if attn_op_name == "base":
                op = xformers.ops.MemoryEfficientAttentionOp
            elif attn_op_name == "cutlass":
                op = xformers.ops.MemoryEfficientAttentionCutlassOp
            elif attn_op_name == "triton":
                op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            else:
                raise ValueError(f"Unknown attn_op_name {attn_op_name}")

            self.attn_op = lambda q, k, v, m, p: xformers.ops.memory_efficient_attention(
                q, k, v, m, p, op=op
            )

    def reshape_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Copy from transpose_for_scores but without the transpose"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x

    @staticmethod
    def layout_attention_mask(mask, num_attention_heads):
        # (B, 1, 1, S) -> (B, S)
        mask = mask.squeeze()
        # (B, S) -> (B, 1, S)
        mask = mask.reshape((mask.shape[0], 1, mask.shape[1]))
        # (B, 1, S) -> (B x H, S, S)
        mask = mask.repeat(num_attention_heads, mask.shape[2], 1)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        assert head_mask is None, "head_mask is not supported for now"
        assert not output_attentions, "output_attentions is not supported for now"
        assert past_key_value is None, "past_key_value is not supported for now"

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.reshape_for_scores(self.key(hidden_states))
        value_layer = self.reshape_for_scores(self.value(hidden_states))
        query_layer = self.reshape_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        assert not is_cross_attention, "cross attention is not supported for now"

        # The required attention mask shape is [batch_size x #heads, seq_length, seq_length];
        # while the input shape is [batch_size, 1, 1, seq_length].
        # In other words, we need to broadcast other dimensions manually.
        attention_mask = self.layout_attention_mask(attention_mask, self.num_attention_heads)

        context_layer = self.attn_op(
            query_layer, key_layer, value_layer, attention_mask, p=self.attention_probs_dropout_prob
        )
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class GPT2Attention(nn.Module):
    """FIXME: This is not working yet"""
    def __init__(self, config, is_cross_attention=False, layer_idx=None, attn_op_name="native"):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        assert config.scale_attn_weights, "scale_attn_weights must be True"
        assert not is_cross_attention, "cross attention is not supported for now"

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        assert not self.scale_attn_by_inverse_layer_idx, "scale_attn_by_inverse_layer_idx is not supported for now"
        assert not self.reorder_and_upcast_attn, "reorder_and_upcast_attn is not supported for now"

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_pdrop = config.attn_pdrop
        self.resid_drop = config.resid_pdrop
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(
                torch.bool
            )
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(
            bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device
        )

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
                -1, dk, k_seq_len
            )
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
            )
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor # (batch, seq_length, head, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            assert head_mask is None, "head_mask is not supported for now"
            attn_output = pt_attention(query, key, value, attention_mask)
            attn_weights = None
            #attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            assert attn_weights is not None, "output attention is not supported for now"
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)