import torch 
import time
import math
from typing import Tuple, Union, Optional
import torch
import torch.nn.functional as F
from einops import rearrange
import torch
import triton
import triton.language as tl
import numpy as np
import math


from .fn_only_gk import FlashGRet
from .fn_only_gv import FlashGRet_O

def intra_chunk_onc(q, k, v, gk, gv):
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    if gk is not None:
        assert gk.is_contiguous()
    if gv is not None:
        assert gv.is_contiguous()

    # q = q.float()
    # k = k.float()
    # v = v.float()

    origin_chunk_size = k.shape[-2]
    assert k.shape[-2] % 16 == 0
    
    if gk is not None:
        A = FlashGRet.apply(q, k, gk)
    else:
        A = q @ k.transpose(-1, -2)

    mask = torch.triu(torch.ones(A.shape[-2], A.shape[-2]), diagonal=1).bool().to(A.device)
    A.masked_fill_(mask, 0)

    if gv is not None:
        O = FlashGRet_O.apply(A, v, gv)
    else:
        O = A.to(v) @ v         

    return O


    

def compute_inner(query, key, value, decay_key, decay_value):
    # query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    # key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    # value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)

    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)


    original_dtype = query.dtype
    decay_key = decay_key.float().exp()
    decay_value = decay_value.float().exp()    

    query = query.float()
    key = key.float()
    value = value.float()

    query = (query * decay_key)
    key = key / decay_key 
    
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    value = value / decay_value 
    return ((qk @ value) * decay_value).to(original_dtype)
    

if __name__ == "__main__":
    B = 32
    H = 2
    L = 2048
    D_QK = 512
    D_V = 128
    requires_grad = True  
    chunk_size = 64
    num_chunk = L // chunk_size

    dtype = torch.float32
    q = ((torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda') / 10).to(dtype)).requires_grad_(requires_grad)  
    k = (torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda') / 10).to(dtype).requires_grad_(requires_grad)
    v = torch.randn(B, H, num_chunk, chunk_size, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda')
    gv =  torch.randn(B, H, num_chunk, chunk_size, D_V, device='cuda')
    

    gk = F.logsigmoid(gk) / 32
    gv = F.logsigmoid(gv) / 32

    gk = (gk.cumsum(-2)).requires_grad_(requires_grad)
    gv = (gv.cumsum(-2)).requires_grad_(requires_grad)

    # gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)     
     

    output = intra_chunk_onc(q, k, v, gk, gv)

    output.sum().backward(retain_graph=True)
    target = [q, k, v, gk, gv]

    grad1= []
    grad2= []
    for s in target:
        grad1.append(s.grad.clone())
        s.grad.zero_()
    
    
    o2 = compute_inner(q, k, v, gk, gv)
    o2.sum().backward(retain_graph=True)

    for s in target:
        grad2.append(s.grad.clone())
        s.grad.zero_()
    
    print( (output - o2).abs().max())

    for a, b in zip(grad1, grad2):
        print( (a  - b).abs().max())



    breakpoint()


