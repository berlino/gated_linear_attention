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
def _fwd_preprocess_cumsum_gv(
    V, GV,  
    GV_cumsum, GV_exp, V_reduce, GV_last_exp, 
    NUM_CHUNK, L, normalizer, clamp_min,
    D_MODEL_V: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)

    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    
    GV_last_exp_ptr = GV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V)

    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_exp_ptr = GV_exp + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    cumsum = tl.zeros([D_MODEL_V], dtype=tl.float32)
    
    for _ in range(CHUNK_SIZE):
        gv = tl.load(GV_ptr).to(tl.float32) 
        # gv = tl.sigmoid(gv)
        # gv = tl.log(gv + 1e-9) / normalizer
        gv = stable_log_sigmoid(gv) / normalizer
        gv = tl.where(gv >= clamp_min, gv, clamp_min)
        cumsum += gv

        tl.store(GV_cumsum_ptr, cumsum.to(GV_cumsum_ptr.dtype.element_ty))
        tl.store(GV_exp_ptr, tl.exp(cumsum).to(GV_cumsum_ptr.dtype.element_ty))
        
        GV_cumsum_ptr += D_MODEL_V
        GV_exp_ptr += D_MODEL_V
        GV_ptr += D_MODEL_V

    tl.store(GV_last_exp_ptr, tl.exp(cumsum).to(GV_last_exp_ptr.dtype.element_ty))
    
    tl.debug_barrier()
    
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)    
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    V_reduce_ptr = V_reduce + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)    

    for _ in range(CHUNK_SIZE):
        v = tl.load(V_ptr)                
        gv = tl.load(GV_cumsum_ptr)
        v_reduce = v * tl.exp(cumsum - gv)
        tl.store(V_reduce_ptr, v_reduce.to(V_reduce_ptr.dtype.element_ty))
        

        V_ptr += D_MODEL_V
        V_reduce_ptr += D_MODEL_V
        GV_cumsum_ptr += D_MODEL_V
    
@triton.jit
def _bwd_preprocess_cumsum_gv(
    V, GV, GV_cumsum,     

    DGV_cumsum_exp, DV_reduce, DGV_last_exp, DGV_cumsum, 
    DV, DGV, 

    NUM_CHUNK, L, normalizer, clamp_min, 
    D_MODEL_V: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):


    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    DV_ptr = DV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DV_reduce_ptr = DV_reduce + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_ptr = DGV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_exp_ptr = DGV_cumsum_exp + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    DGV_ptr = DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    D_GV_last_exp_ptr = DGV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V) 
     
    cumsum_gradient = tl.zeros([D_MODEL_V], dtype=tl.float32)
    grad_gv_last = tl.zeros([D_MODEL_V], dtype=tl.float32)

    gv_last = tl.load(GV_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_V)    
    cumsum_gradient += tl.load(D_GV_last_exp_ptr) * tl.exp(gv_last).to(tl.float32)
    
    GV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    GV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    V_ptr += (CHUNK_SIZE - 1) * D_MODEL_V 

    DV_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    for idx in range(CHUNK_SIZE -1, -1, -1):
        gv_cs = tl.load(GV_cumsum_ptr).to(tl.float32)
        v = tl.load(V_ptr).to(tl.float32)
        grad_v = tl.exp(gv_last - gv_cs) * tl.load(DV_reduce_ptr).to(tl.float32)
        tl.store(DV_ptr, grad_v.to(DV_ptr.dtype.element_ty))
        grad_v *= v
        cumsum_gradient -= grad_v
        grad_gv_last += grad_v

        # q = tl.load(Q_ptr).to(tl.float32)
        grad_v = tl.exp(gv_cs) * tl.load(DGV_cumsum_exp_ptr) 
        cumsum_gradient += grad_v

        # from intra-chunk contribution.
        cumsum_gradient += tl.load(DGV_cumsum_ptr).to(tl.float32) 
        
        tl.store(DGV_ptr, cumsum_gradient.to(DGV_ptr.dtype.element_ty))

        V_ptr -= D_MODEL_V
        DV_reduce_ptr -= D_MODEL_V
        GV_cumsum_ptr -= D_MODEL_V
        DGV_cumsum_ptr -= D_MODEL_V
        DV_ptr -= D_MODEL_V
        DGV_ptr -= D_MODEL_V
        DGV_cumsum_exp_ptr -= D_MODEL_V
 
    DGV_ptr =  DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    GV_ptr =  GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    
    grad_gv_last = grad_gv_last + 0.

    for idx in range(CHUNK_SIZE -1, -1, -1):        
        dgv = tl.load(DGV_ptr).to(tl.float32)
        dgv += grad_gv_last
        gv = tl.load(GV_ptr).to(tl.float32) 

        gv_logit = stable_log_sigmoid(gv) / normalizer
        gv = tl.sigmoid(gv)    
        dgv = (dgv / normalizer) * (1 - gv)        
        dgv = tl.where(gv_logit >= clamp_min, dgv, 0.)

        tl.store(DGV_ptr, dgv.to(DGV_ptr.dtype.element_ty))
        DGV_ptr -= D_MODEL_V
        GV_ptr -= D_MODEL_V
    


class PreprocessCumSum_GV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, gv, normalizer_gv=8, clamp_min=-3):
        v = v.contiguous()
        gv = gv.contiguous()
    
        B, H, NUM_CHUNK, CHUNK_SIZE, D_v = v.shape


        # D_k = k.shape[-1]
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


        gv_cumsum = torch.empty_like(gv, dtype=torch.float32)                        
        gv_cumsum_exp = torch.empty_like(gv)
        v_reduce = torch.empty_like(v)
        gv_last_exp = torch.empty_like(gv[:, :, :, 0], dtype=torch.float32)
        _fwd_preprocess_cumsum_gv[grid](
            v, gv,  gv_cumsum, gv_cumsum_exp,  
            v_reduce, gv_last_exp, 
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=normalizer_gv, clamp_min=clamp_min,
            D_MODEL_V=D_v, num_warps=8 if D_v >= 512 else 4
        )            
            
        ctx.grid = grid 
        ctx.save_for_backward(v, gv, gv_cumsum)
        ctx.normalizer_gv = normalizer_gv
        ctx.clamp_min = clamp_min

        return gv_cumsum, v_reduce, gv_cumsum_exp, gv_last_exp



    @staticmethod
    def backward(ctx, dgv_cumsum, dv_reduce, dgv_cumsum_exp, dgv_last_exp):

        dgv_cumsum = dgv_cumsum.contiguous()
        dv_reduce = dv_reduce.contiguous()
        dgv_cumsum_exp = dgv_cumsum_exp.contiguous()
        dgv_last_exp = dgv_last_exp.contiguous()
        v, gv, gv_cumsum = ctx.saved_tensors
        grid = ctx.grid

        B, H, NUM_CHUNK, CHUNK_SIZE, D_v = v.shape

        dv = torch.empty_like(v)
        dgv = torch.empty_like(gv)        
        _bwd_preprocess_cumsum_gv[grid](
            v, gv, gv_cumsum,  dgv_cumsum_exp, dv_reduce, dgv_last_exp, dgv_cumsum, 
            dv, dgv, 
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=ctx.normalizer_gv, clamp_min = ctx.clamp_min,
            D_MODEL_V=D_v, num_warps=8 if D_v >= 512 else 4 
        )    
        return dv, dgv, None, None, None



# def prepare_cumsum(query, key, value, g_key, g_value, normalizer_gk=8, normalizer_gv=8, clamp_min=-3):
#     g_key = g_key 
#     g_value = g_value

#     g_key = (F.logsigmoid(g_key) / normalizer_gk).clamp(min=clamp_min)
#     g_key_cumsum = g_key.cumsum(-2)
#     reduce_chunk_key = (g_key_cumsum[..., -1, None, :] -  g_key_cumsum).exp()
    
#     g_value = (F.logsigmoid(g_value) / normalizer_gv).clamp(min=clamp_min)
#     g_value_cumsum = g_value.cumsum(-2)
#     reduce_chunk_value = (g_value_cumsum[..., -1, None, :] - g_value_cumsum).exp()
    
#     reduce_value = value * reduce_chunk_value.to(value)
#     reduce_key = key * reduce_chunk_key.to(key)    
    
#     g_key_last_exp = g_key_cumsum[:, :, :, -1].exp()
#     g_value_last_exp = g_value_cumsum[:, :, :, -1].exp()

#     g_value_cumsum_exp = g_value_cumsum.exp().to(key)
#     q_exp = query * g_key_cumsum.exp().to(query)

#     return  g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp
    
    
    
        
    
# # to_add = (key * reduce_chunk_key.to(key)).transpose(-1, -2) @  (value * reduce_chunk_value.to(value))
 
    
    

# if __name__ == "__main__":
#     B = 32
#     H = 4
#     L = 2048
#     D_K = 256
#     D_V = 512
    

#     chunk_size = 16
#     num_chunk = L // chunk_size
#     device = "cuda"
#     requires_grad = True
#     dtype = torch.float32
    

#     v1 = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad)
#     v2 = (torch.randn(B, H, num_chunk, chunk_size, D_V)).cuda().to(dtype).requires_grad_(requires_grad) 
#     g1 = torch.randn(B, H,  num_chunk, chunk_size, D_K).cuda().to(dtype)
#     g1[..., :D_K//2] = -1
#     g1[..., D_K//2:] = 1
#     g1.requires_grad_(requires_grad)
#     g2 = torch.randn(B, H, num_chunk, chunk_size, D_V).cuda().to(dtype).uniform_(0.3, 0.7).log().requires_grad_(requires_grad)
#     q = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad) 

#     test_speed= False 
#     test_gradient = True

#     if test_gradient:
#         target = [v1, v2, g1, g2, q]
#         grad1= [ ]
#         g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2, 1, 1, -1)
        
#         o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
#         o.sum().backward(retain_graph=True )
        
#         for v in target:   
#             grad1.append(v.grad.clone())
#             v.grad.zero_()
        
    
#         g_key_cumsum2, g_value_cumsum2,  reduce_key2, reduce_value2, q_exp2, g_value_cumsum_exp2, g_key_last_exp2, g_value_last_exp2 = prepare_cumsum(q, v1, v2, g1, g2, 1, 1, -1)
#         o = (g_key_cumsum2 ** 2).sum() + (g_value_cumsum2 ** 2).sum() + (reduce_key2 **2).sum() + (q_exp2**2).sum() + (g_value_cumsum_exp2 **2).sum() + (g_key_last_exp2**2).sum() + (g_value_last_exp2 **2).sum()
#         o.sum().backward(retain_graph=True )

#         grad2= [ ]

#         for v in target:
#             grad2.append(v.grad.clone())
#             v.grad.zero_()
        
#         for ss1,ss2 in zip(grad1, grad2):
#             print( (ss1 - ss2).abs().max())
    
            
#         breakpoint()
  

        
#     #### speed testing
#     if test_speed:

#         for _ in range(100):
#             g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)

#             if requires_grad:
#                 o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
#                 o.sum().backward(retain_graph=True )

#             g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
#             if requires_grad:
#                 o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
#                 o.sum().backward(retain_graph=True )
        
#         # for ss, ss2 in zip(s, s2):
#         #     print((ss-ss2).abs().max())
        
#         print("Warmup.")
#         # print('warm up done')
#         torch.cuda.synchronize()

#         start = time.time()

#         for _ in range(200):
            
#             g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)
            
#             if requires_grad:
#                 o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
#                 o.sum().backward(retain_graph=True )

        
#         torch.cuda.synchronize()
#         end = time.time()
#         print("Triton time: ", end - start)

#         torch.cuda.synchronize()
#         start = time.time()

#         for _ in range(200):
#             g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
#             if requires_grad:
#                 o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
#                 o.sum().backward(retain_graph=True )
                
#         torch.cuda.synchronize()
#         end = time.time()
#         print("Pytorch time: ", end - start)






        
