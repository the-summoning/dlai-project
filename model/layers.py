import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torch.nn.functional import scaled_dot_product_attention

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached



class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        
        self.head_dim = hidden_size // num_heads
        self.output_size = self.head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(self.output_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        query = qkv[:, :, :self.num_heads]
        key   = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        query, key = apply_rotary_pos_emb(query, key, *cos_sin)
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
        
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = self._find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def _find_multiple(self, a, b):
        return (-(a // -b)) * b


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        expansion: float,
        seq_len: int,
        num_registers: int = 1,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        use_mlp_t: bool = True,
    ):
        super().__init__()
        self.use_mlp_t = use_mlp_t

        if use_mlp_t:
            self.mlp_t = SwiGLU(seq_len + num_registers, expansion) # type: ignore
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=hidden_size // num_heads,
                max_position_embeddings=seq_len + num_registers,
                base=rope_theta
            )
            self.self_attn = Attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_heads,
                causal=False
            )

        self.mlp = SwiGLU(hidden_size, expansion)
        self.norm_eps = norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:        
        if self.use_mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            mlp_out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            cos_sin = self.rotary_emb()
            attn_out = self.self_attn(hidden_states=hidden_states, cos_sin=cos_sin)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        
        mlp_out = self.mlp(hidden_states)
        
        return rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
    

# if __name__ == "__main__":
#     model = nn.Sequential(
#         TransformerBlock(
#             num_heads=8,
#             hidden_size=512,
#             expansion=4.0,
#             max_seq_len=16
#         ),
#         TransformerBlock(
#             num_heads=8,
#             hidden_size=512,
#             expansion=4.0,
#             max_seq_len=16
#         )
#     )
#     x = torch.randn(2, 16, 512)
#     out = model(x)
#     print(out.shape)
