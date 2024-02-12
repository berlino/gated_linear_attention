import torch
from einops import rearrange
import triton 
import triton.language as tl
import torch.nn.functional as F
import torch.nn as nn 

from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla

class GatedLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.n_head
        
        self.gate_fn = nn.functional.silu
        assert config.use_gk and not config.use_gv, "Only use_gk is supported for simplicity."

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
        self.k_gate =  nn.Sequential(nn.Linear(self.embed_dim, 16, bias=False), nn.Linear(16, self.embed_dim // 2))

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.head_dim = self.embed_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.post_init()



    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        if isinstance(self.k_gate, nn.Sequential):
            nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)
        else:
            nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)

    def forward(self, x, hidden_states=None):
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        k_gate = self.k_gate(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        output, new_hidden_states = self.gated_linear_attention(q, k, v, k_gate, hidden_states=hidden_states)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output, new_hidden_states


    def gated_linear_attention(self, q, k, v, gk, normalizer=16, hidden_states=None):
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = F.logsigmoid(gk) / normalizer

        if self.training:
            o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=True)
        else:
            o = fused_recurrent_gla(q, k, v, gk)
            new_hidden_states = None
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states


if __name__ == "__main__":
    BATCH, H, N_CTX, D_MODEL = 32, 4, 2048, 1024

    GLA = GatedLinearAttention(D_MODEL, H, use_gk=True, use_gv=False).cuda().to(torch.bfloat16)

    x = torch.randn((BATCH, N_CTX, D_MODEL), dtype=torch.bfloat16,
     device="cuda", requires_grad=True)
    
    y, _ = GLA(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
    
