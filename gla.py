import torch
from einops import rearrange
import triton 
import triton.language as tl
import torch.nn.functional as F
from kernels.intra_chunk_contribution.fn import intra_chunk_onc
from kernels.inter_chunk_contribution.fn import inter_chunk_onc
import torch.nn as nn 
from fla.ops.triton.gla import fused_chunk_gla


# token mixing layer. ~4D^2. 
class GatedLinearAttention(nn.Module):
    def __init__(self, d_model, n_head, use_gk=True, use_gv=False):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        
        self.gate_fn = nn.functional.silu

        self.use_gk = use_gk
        self.use_gv = use_gv

        self.factor = 1

        #7/8 D^2
        if use_gk and use_gv:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
            self.k_gate = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
            self.v_gate = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 8, bias=False), nn.Linear(self.embed_dim // 8, self.embed_dim * self.factor))
        
        #~D^2
        elif use_gk and not use_gv:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
            self.k_gate =  nn.Sequential(nn.Linear(self.embed_dim, 16, bias=False), nn.Linear(16, self.embed_dim // 2))

        #D^2                               
        elif not use_gk and use_gv:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
            self.v_gate = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False), nn.Linear(self.embed_dim // 4, self.embed_dim * self.factor))            

        #D^2
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim // 2, bias=False)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim // 2, bias=False)


        # D^2
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim * self.factor, bias=False)
        # D^2
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim * self.factor, bias=True)
        # D^2
        self.out_proj = nn.Linear(self.embed_dim * self.factor, self.embed_dim, bias=False)

        self.head_dim = self.embed_dim * self.factor // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.post_init()



    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
                    
        if self.use_gk:
            if isinstance(self.k_gate, nn.Sequential):
                nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
                nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)
            else:
                nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)
                
        if self.use_gv:
            if isinstance(self.v_gate, nn.Sequential):
                nn.init.xavier_uniform_(self.v_gate[0].weight, gain=2 ** -2.5)
                nn.init.xavier_uniform_(self.v_gate[1].weight, gain=2 ** -2.5)
            else:
                nn.init.xavier_uniform_(self.v_gate.weight, gain=2 ** -2.5)


    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        if self.use_gk:
            k_gate = self.k_gate(x)
        else:
            k_gate = None 
        v = self.v_proj(x)
        if self.use_gv:
            v_gate = self.v_gate(x)
        else:
            v_gate = None 
        g = self.g_proj(x)
        output = self.gated_linear_attention(q, k, v, k_gate, v_gate, num_head=self.num_heads)
        output = self.group_norm(output)
        output = rearrange(output, 'b h n c d -> b (n c) (h d)')

        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output


    def gated_linear_attention(self, q, k, v, gk, gv, normalizer_gk=16, normalizer_gv=16,  num_head=8, chunk_size=128):
        # assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        q = rearrange(q, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        k = rearrange(k, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        v = rearrange(v, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()

        if gk is not None:
            gk = rearrange(gk, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()

        if gv is not None:
            gv = rearrange(gv, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()

        # call faster implementation
        if gk is not None and gv is None:
            gk = F.logsigmoid(gk) / normalizer_gk
            o = fused_chunk_gla(q, k, v, gk)
            return o
        # call the general implementation
        else:
            gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv, normalizer_gk, normalizer_gv)        
            o2 = intra_chunk_onc(q, k, v, gk, gv)
            o = (o1 + o2)
            return o

if __name__ == "__main__":
    BATCH, H, N_CTX, D_MODEL = 32, 4, 2048, 1024

    GLA = GatedLinearAttention(D_MODEL, H, use_gk=True, use_gv=True).cuda().to(torch.bfloat16)

    x = torch.randn((BATCH, N_CTX, D_MODEL), dtype=torch.bfloat16,
     device="cuda", requires_grad=True)
    
    y = GLA(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
    