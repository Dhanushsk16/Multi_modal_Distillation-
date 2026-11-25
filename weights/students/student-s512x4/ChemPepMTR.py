import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from .config import model_config
from typing import Mapping, Optional
from transformers.tokenization_utils_base import BatchEncoding
from dataclasses import dataclass


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L132

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = self._compute_theta(device=torch.device("cpu"))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def _compute_theta(self, device: torch.device) -> torch.Tensor:
        exponents = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        exponents = exponents[: (self.dim // 2)] / float(self.dim)
        base = torch.tensor(float(self.base), dtype=torch.float32, device=device)
        return torch.pow(base, -exponents)

    def _theta_is_valid(self) -> bool:
        if not hasattr(self, "theta"):
            return False
        t = self.theta
        if not torch.isfinite(t).all():
            return False
        if not ((t > 0.0).all() and (t <= 1.0).all()):
            return False
        return True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        if not self._theta_is_valid():
            self.theta = self._compute_theta(device=torch.device("cpu"))

        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # Some environments load this model with corrupted non-persistent buffers.
        # Regenerate theta/cache from config if either buffer is invalid.
        if (
            (not self._theta_is_valid())
            or (not torch.isfinite(self.cache).all())
            or (self.cache.size(0) < seq_len)
            or (self.theta.device != x.device)
        ):
            theta = self._compute_theta(device=x.device)
            self.theta = theta
            self.build_rope_cache(max(self.max_seq_len, seq_len))

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)




@dataclass
class MLMOutput(ModelOutput):
    logits: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    mean_pool: torch.Tensor = None

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim * 2, bias=True)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
    def forward(self, x):
        # x: (N, input_dim)
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        output = self.linear2(F.silu(x1) * x2)
        return self.dropout(output)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, x, input_pos=None, mask=None):
        B, T, C = x.shape  # Batch, sequence, embedding dim
        
        # project into queries, keys, and values
        q, k, v = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).unbind(2)  # (B, T, num_heads, head_dim)

        # Apply rotary positional embeddings to queries and keys
        q, k = self.rotary(q, input_pos=input_pos), self.rotary(k, input_pos=input_pos)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            # set padding positions to -inf
            mask = mask.to(dtype=torch.float32)  # Ensure mask is float
            mask = (1.0 - mask) * -1e9  # Convert to -inf for padding positions
            
            # mask: (B, T) -> (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask = mask.expand(B, 1, T, T)  # expands to (batch, 1, seqlen, seqlen)
            
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim)

    def forward(self, x, input_pos=None, mask=None):
        x = x + self.attn(self.attn_norm(x), input_pos=input_pos, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class TransformerStack(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            UnifiedTransformerBlock(embed_dim, num_heads, ffn_hidden_dim, max_seq_len)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, input_pos=None, mask=None):
        for block in self.blocks:
            x = block(x, input_pos=input_pos, mask=mask)
        return self.norm(x)

class MLM_core(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_hidden_dim: int,
        output_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = TransformerStack(
            num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len
        )
        self.sequence_head = nn.Linear(embed_dim, output_dim, bias=True)


    def forward(self, ids, mask=None, pad_token_id=0, input_pos=None):
        x = self.embed(ids)
        x = self.transformer(x, mask=mask, input_pos=input_pos)
        # generate logits for MLM
        # print(f"x shape: {x.shape}")  # Debugging line to check the shape of x
        logits = self.sequence_head(x)
        # print(f"logits shape: {logits.shape}")  # Debugging line to check the shape of logits

        # mean pool but remove positions that have pad tokens
        token_mask = (ids != pad_token_id).unsqueeze(-1).type_as(x)
        sum_pool = (x * token_mask).sum(dim=1)
        valid_counts = token_mask.sum(dim=1).clamp(min=1.0)
        mean_pool = sum_pool / valid_counts

        outputs = MLMOutput(
            logits=logits,
            last_hidden_state=x,
            mean_pool=mean_pool,
        )

        return outputs

class MLM_model(PreTrainedModel): # HF-facing class name
    config_class = model_config

    def __init__(self, config):
        super().__init__(config)
        self.model = MLM_core(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            ffn_hidden_dim=config.ffn_hidden_dim,
            output_dim=config.output_dim,
            max_seq_len=config.max_seq_len,
        )
        self.post_init()  # Initialize weights and apply final processing
    
    # if inputs are dictionary
    def forward(self, x=None, **kwargs):
        if isinstance(x, (BatchEncoding, Mapping)):
            return self.model(x.get("input_ids"), mask=x.get("attention_mask"))
        
        if "input_ids" in kwargs or "attention_mask" in kwargs:
            return self.model(kwargs.get("input_ids"), mask=kwargs.get("attention_mask"))
        
        return self.model(x)