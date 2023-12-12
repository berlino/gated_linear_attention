import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import time

# def stable_logsigmoid(x):
#     # Use the identity log(sigmoid(x)) = -log(1 + exp(-x))
#     # This is stable for large negative values of x
#     neg_abs_x = -torch.abs(x)
#     return torch.where(x < 0, x, neg_abs_x) - torch.log1p(torch.exp(neg_abs_x))

@triton.jit 
def stable_log_sigmoid(x):
    # return     
    max_value = tl.where(x<0, x, 0)
    abs_value = tl.where(x>0, x, -x)
    return max_value - tl.log(1 + tl.exp(-abs_value))
        
@triton.jit
def _fwd_preprocess_cumsum_gk(
    Q, K, GK, GK_cumsum, 
    Q_exp, K_reduce, GK_last_exp, 
    NUM_CHUNK, L, normalizer, clamp_min, 
    D_MODEL_K: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):


    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    Q_exp_ptr = Q_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_last_exp_ptr = GK_last_exp +  offset_bh * NUM_CHUNK * D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_MODEL_K)

    cumsum = tl.zeros([D_MODEL_K], dtype=tl.float32)

    for _ in range(CHUNK_SIZE):
        gk = tl.load(GK_ptr).to(tl.float32) 
        gk = stable_log_sigmoid(gk) / normalizer
        gk = tl.where(gk >= clamp_min, gk, clamp_min)

        cumsum += gk 
        tl.store(GK_cumsum_ptr, cumsum.to(GK_cumsum_ptr.dtype.element_ty))

        cumsum_exp = tl.exp(cumsum)
        
        q = tl.load(Q_ptr)        
        q_exp = q * cumsum_exp
        tl.store(Q_exp_ptr, q_exp)


        Q_ptr += D_MODEL_K
        Q_exp_ptr += D_MODEL_K
        GK_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K

    tl.store(GK_last_exp_ptr, tl.exp(cumsum).to(GK_last_exp_ptr.dtype.element_ty))

    tl.debug_barrier()
    
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_reduce_ptr = K_reduce + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    for _ in range(CHUNK_SIZE):
        gk_cumsum = tl.load(GK_cumsum_ptr)
        k = tl.load(K_ptr)
        k_reduce = k * tl.exp(cumsum - gk_cumsum)
        tl.store(K_reduce_ptr, k_reduce.to(K_reduce_ptr.dtype.element_ty))

        K_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
        K_reduce_ptr += D_MODEL_K




@triton.jit
def _bwd_preprocess_cumsum_gk(
    Q, K, GK, GK_cumsum, 
    
    DQ_exp, DK_reduce, DGK_last_exp, DGK_cumsum, 

    DQ, DK, DGK, 

    NUM_CHUNK, L, normalizer, clamp_min, 
    D_MODEL_K: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    DQ_ptr = DQ + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DK_ptr = DK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DQ_exp_ptr = DQ_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DK_reduce_ptr = DK_reduce + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DGK_cumsum_ptr = DGK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DGK_ptr = DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    D_GK_last_exp_ptr = DGK_last_exp + offset_bh * NUM_CHUNK * D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_MODEL_K) 
    # 
    cumsum_gradient = tl.zeros([D_MODEL_K], dtype=tl.float32)
    grad_gk_last = tl.zeros([D_MODEL_K], dtype=tl.float32)

    gk_last = tl.load(GK_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_K).to(tl.float32)    
    cumsum_gradient += tl.load(D_GK_last_exp_ptr) * tl.exp(gk_last)
    
    GK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    GK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    Q_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    K_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    
    DQ_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K

    
    for idx in range(CHUNK_SIZE -1, -1, -1):
        gk_cs = tl.load(GK_cumsum_ptr).to(tl.float32)
        k = tl.load(K_ptr).to(tl.float32)
        grad_k = tl.exp(gk_last - gk_cs) * tl.load(DK_reduce_ptr).to(tl.float32)
        tl.store(DK_ptr, grad_k.to(DK_ptr.dtype.element_ty))
        grad_k *= k     
        cumsum_gradient -=  grad_k
        grad_gk_last += grad_k

        q = tl.load(Q_ptr).to(tl.float32)
        grad_q = tl.exp(gk_cs) * tl.load(DQ_exp_ptr) 
        tl.store(DQ_ptr, grad_q.to(DK_ptr.dtype.element_ty))
        cumsum_gradient += grad_q * q.to(tl.float32)

        # from intra-chunk contribution.
        cumsum_gradient += tl.load(DGK_cumsum_ptr).to(tl.float32) 
        
        tl.store(DGK_ptr, cumsum_gradient.to(DGK_ptr.dtype.element_ty))

        Q_ptr -= D_MODEL_K
        DQ_exp_ptr -= D_MODEL_K
        K_ptr -= D_MODEL_K
        DK_reduce_ptr -= D_MODEL_K
        GK_cumsum_ptr -= D_MODEL_K
        DGK_cumsum_ptr -= D_MODEL_K
        DQ_ptr -= D_MODEL_K
        DK_ptr -= D_MODEL_K
        DGK_ptr -= D_MODEL_K
    

    DGK_ptr =  DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K
    GK_ptr =  GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K

    # tl.store(D_GK_last_exp_ptr, cumsum_gradient)

    # seems stupid. just workaround some compiler bugs.
    grad_gk_last = grad_gk_last + 0.
    for idx in range(CHUNK_SIZE -1, -1, -1):        
        dgk = tl.load(DGK_ptr).to(tl.float32)
        dgk += grad_gk_last
    
        gk = tl.load(GK_ptr).to(tl.float32) 
        gk_logit = stable_log_sigmoid(gk) / normalizer
        dgk = tl.where(gk_logit >= clamp_min, (dgk / normalizer)  * (1 - tl.sigmoid(gk)), 0.)

        tl.store(DGK_ptr, dgk.to(DGK_ptr.dtype.element_ty))
        DGK_ptr -= D_MODEL_K
        GK_ptr -= D_MODEL_K
    

class PreprocessCumSum_GK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k,  gk,  normalizer_gk=8, clamp_min=-3):
        q = q.contiguous()
        k = k.contiguous()
        gk = gk.contiguous()
    
        B, H, NUM_CHUNK, CHUNK_SIZE, D = q.shape

        D_k = k.shape[-1]
        # D_v = v.shape[-1]
        
    
        # (B, H, L, D_K, D_V)
        # , memory_format=torch.contiguous_format)
        # o = torch.empty_like(v).contiguous()
        # share memory's limit.
        # BLOCK_MODEL_K = 128
        # BLOCK_MODEL_V = 128
        #split k

        grid = (B * H, NUM_CHUNK)
        ctx.grid = grid 

        k_reduce = torch.empty_like(k)

        q_exp = torch.empty_like(q)

        gk_cumsum = torch.empty_like(gk)

        gk_last_exp = torch.empty_like(gk[:, :, :, 0], dtype=torch.float32)

        _fwd_preprocess_cumsum_gk[grid](
            q, k, gk, gk_cumsum, 
            q_exp, k_reduce, gk_last_exp, 
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=normalizer_gk, clamp_min=clamp_min,
            D_MODEL_K=D_k, num_warps=8 if D_k >= 512 else 4
        )
                

        ctx.grid = grid 
        ctx.save_for_backward(q, k, gk, gk_cumsum)
        ctx.normalizer_gk = normalizer_gk
        ctx.clamp_min = clamp_min

        return gk_cumsum, k_reduce, q_exp,  gk_last_exp

    @staticmethod
    def backward(ctx, dgk_cumsum, dk_reduce, dq_exp, dgk_last_exp):

        dgk_cumsum = dgk_cumsum.contiguous()
        dk_reduce = dk_reduce.contiguous()
        dq_exp = dq_exp.contiguous()
        dgk_last_exp = dgk_last_exp.contiguous()

        q, k, gk, gk_cumsum = ctx.saved_tensors
        grid  = ctx.grid

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dgk = torch.empty_like(gk)

        B, H, NUM_CHUNK, CHUNK_SIZE, D_k = q.shape


        # D_v = v.shape[-1]        

        _bwd_preprocess_cumsum_gk[grid](
            q, k, gk, gk_cumsum, 
            dq_exp, dk_reduce, dgk_last_exp, dgk_cumsum,
            dq, dk, dgk,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=ctx.normalizer_gk, clamp_min = ctx.clamp_min,
            D_MODEL_K=D_k, num_warps=8 if D_k >= 512 else 4
        )

        return dq, dk, dgk, None, None, None
    

def prepare_cumsum(query, key, value, g_key, g_value, normalizer_gk=8, normalizer_gv=8, clamp_min=-3):
    g_key = g_key 
    g_value = g_value

    g_key = (F.logsigmoid(g_key) / normalizer_gk).clamp(min=clamp_min)
    g_key_cumsum = g_key.cumsum(-2)
    reduce_chunk_key = (g_key_cumsum[..., -1, None, :] -  g_key_cumsum).exp()
    
    g_value = (F.logsigmoid(g_value) / normalizer_gv).clamp(min=clamp_min)
    g_value_cumsum = g_value.cumsum(-2)
    reduce_chunk_value = (g_value_cumsum[..., -1, None, :] - g_value_cumsum).exp()
    
    reduce_value = value * reduce_chunk_value.to(value)
    reduce_key = key * reduce_chunk_key.to(key)    
    
    g_key_last_exp = g_key_cumsum[:, :, :, -1].exp()
    g_value_last_exp = g_value_cumsum[:, :, :, -1].exp()

    g_value_cumsum_exp = g_value_cumsum.exp().to(key)
    q_exp = query * g_key_cumsum.exp().to(query)

    return  g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp
    
    
    
        
    
# to_add = (key * reduce_chunk_key.to(key)).transpose(-1, -2) @  (value * reduce_chunk_value.to(value))
 
    
    

if __name__ == "__main__":
    B = 32
    H = 4
    L = 2048
    D_K = 256
    D_V = 512
    

    chunk_size = 16
    num_chunk = L // chunk_size
    device = "cuda"
    requires_grad = True
    dtype = torch.float32
    

    v1 = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad)
    v2 = (torch.randn(B, H, num_chunk, chunk_size, D_V)).cuda().to(dtype).requires_grad_(requires_grad) 
    g1 = torch.randn(B, H,  num_chunk, chunk_size, D_K).cuda().to(dtype)
    g1[..., :D_K//2] = -1
    g1[..., D_K//2:] = 1
    g1.requires_grad_(requires_grad)
    g2 = torch.randn(B, H, num_chunk, chunk_size, D_V).cuda().to(dtype).uniform_(0.3, 0.7).log().requires_grad_(requires_grad)
    q = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad) 

    test_speed= False 
    test_gradient = True

    if test_gradient:
        target = [v1, v2, g1, g2, q]
        grad1= [ ]
        g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum_GK.apply(q, v1, v2, g1, g2, 1, 1, -1)
        
        o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
        o.sum().backward(retain_graph=True )
        
        for v in target:   
            grad1.append(v.grad.clone())
            v.grad.zero_()
        
    
        g_key_cumsum2, g_value_cumsum2,  reduce_key2, reduce_value2, q_exp2, g_value_cumsum_exp2, g_key_last_exp2, g_value_last_exp2 = prepare_cumsum(q, v1, v2, g1, g2, 1, 1, -1)
        o = (g_key_cumsum2 ** 2).sum() + (g_value_cumsum2 ** 2).sum() + (reduce_key2 **2).sum() + (q_exp2**2).sum() + (g_value_cumsum_exp2 **2).sum() + (g_key_last_exp2**2).sum() + (g_value_last_exp2 **2).sum()
        o.sum().backward(retain_graph=True )

        grad2= [ ]

        for v in target:
            grad2.append(v.grad.clone())
            v.grad.zero_()
        
        for ss1,ss2 in zip(grad1, grad2):
            print( (ss1 - ss2).abs().max())
    
            
        breakpoint()
  

        
    #### speed testing
    if test_speed:

        for _ in range(100):
            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)

            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )

            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )
        
        # for ss, ss2 in zip(s, s2):
        #     print((ss-ss2).abs().max())
        
        print("Warmup.")
        # print('warm up done')
        torch.cuda.synchronize()

        start = time.time()

        for _ in range(200):
            
            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)
            
            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )

        
        torch.cuda.synchronize()
        end = time.time()
        print("Triton time: ", end - start)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(200):
            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )
                
        torch.cuda.synchronize()
        end = time.time()
        print("Pytorch time: ", end - start)






        
