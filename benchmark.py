import time
import warnings
from contextlib import contextmanager
import torch

import torch
from einops import rearrange
import triton 
import triton.language as tl
import torch.nn.functional as F
from src.intra_chunk_contribution.fn import intra_chunk_onc
from src.inter_chunk_contribution.fn import inter_chunk_onc


## if you don't want to use gate in the K dimension, gk should be None
## if you don't want to use gate in the V dimension, gv should be None
def gated_linear_attention(q, k, v, gk, gv, normalizer_gk=16, normalizer_gv=16, clamp_min=-2, num_head=8, chunk_size=128):

    # (batch, length, D_Model)
    if len(q.shape) == 3:
        q = rearrange(q, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        k = rearrange(k, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        v = rearrange(v, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        
        if gk is not None:
            gk = rearrange(gk, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        if gv is not None:
            gv = rearrange(gv, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()

    # (batch, num_head, length, d_head)
    elif len(q.shape) == 4:
        q = rearrange(q, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        k = rearrange(k, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        v = rearrange(v, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        
        if gk is not None:
            gk = rearrange(gk, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        if gv is not None:
            gv = rearrange(gv, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
    
    gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv, normalizer_gk, normalizer_gv, clamp_min)
    o2 = intra_chunk_onc(q, k, v, gk, gv)

    o = (o1 + o2)
    # o = o1

    return rearrange(o, 'b h n c d -> b (n c) (h d)')


class TimeCounter:
    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls, log_interval=100, warmup_interval=100, with_sync=True):
        assert warmup_interval >= 1


        def _register(func):
            if func.__name__ in cls.names:
                raise RuntimeError(
                    'The registered function name cannot be repeated!')
            # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[func.__name__] = dict(
                count=0,
                pure_inf_time=0,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

            def fun(*args, **kwargs):
                count = cls.names[func.__name__]['count']
                pure_inf_time = cls.names[func.__name__]['pure_inf_time']
                log_interval = cls.names[func.__name__]['log_interval']
                warmup_interval = cls.names[func.__name__]['warmup_interval']
                with_sync = cls.names[func.__name__]['with_sync']

                count += 1
                cls.names[func.__name__]['count'] = count

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time

                if count >= warmup_interval:
                    pure_inf_time += elapsed
                    cls.names[func.__name__]['pure_inf_time'] = pure_inf_time

                    if count % log_interval == 0:
                        times_per_count = 1000 * pure_inf_time / (
                            count - warmup_interval + 1)
                        print(
                            f'[{func.__name__}]-{count} times per count: '
                            f'{times_per_count:.1f} ms',
                            flush=True)

                return result

            return fun

        return _register



    @classmethod
    @contextmanager
    def profile_time(cls,
                     func_name,
                     log_interval=300,
                     warmup_interval=100,
                     with_sync=True):
        assert warmup_interval >= 1
        warnings.warn('func_name must be globally unique if you call '
                      'profile_time multiple times')

        if func_name in cls.names:
            count = cls.names[func_name]['count']
            pure_inf_time = cls.names[func_name]['pure_inf_time']
            log_interval = cls.names[func_name]['log_interval']
            warmup_interval = cls.names[func_name]['warmup_interval']
            with_sync = cls.names[func_name]['with_sync']
        else:
            count = 0
            pure_inf_time = 0
            cls.names[func_name] = dict(
                count=count,
                pure_inf_time=pure_inf_time,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

        count += 1
        cls.names[func_name]['count'] = count

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        yield

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if count >= warmup_interval:
            pure_inf_time += elapsed
            cls.names[func_name]['pure_inf_time'] = pure_inf_time

            if count % log_interval == 0:
                times_per_count = 1000 * pure_inf_time / (
                    count - warmup_interval + 1)
                print(
                    f'[{func_name}]-{count} times per count: '
                    f'{times_per_count:.1f} ms',
                    flush=True)

## if you don't want to use gate in the K dimension, gk should be None
## if you don't want to use gate in the V dimension, gv should be None
def gated_linear_attention(q, k, v, gk, gv, normalizer_gk=16, normalizer_gv=16, clamp_min=-2, num_head=8, chunk_size=128):

    # (batch, length, D_Model)
    if len(q.shape) == 3:
        q = rearrange(q, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        k = rearrange(k, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        v = rearrange(v, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        
        if gk is not None:
            gk = rearrange(gk, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
        if gv is not None:
            gv = rearrange(gv, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()

    # (batch, num_head, length, d_head)
    elif len(q.shape) == 4:
        q = rearrange(q, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        k = rearrange(k, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        v = rearrange(v, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        
        if gk is not None:
            gk = rearrange(gk, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
        if gv is not None:
            gv = rearrange(gv, 'b h (n c) d -> b h n c d', h = num_head, c = chunk_size).contiguous()
    
    with TimeCounter.profile_time("p1"):
        gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv, normalizer_gk, normalizer_gv)
    with TimeCounter.profile_time("p2"):
        o2 = intra_chunk_onc(q, k, v, gk, gv)

    # gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv, normalizer_gk, normalizer_gv, clamp_min)
    # o2 = intra_chunk_onc(q, k, v, gk, gv)

    o = (o1 + o2)

    # o = o1

    return rearrange(o, 'b h n c d -> b (n c) (h d)')


if __name__ == "__main__":
    BATCH, H, N_CTX, D_HEAD = 32, 8, 2048, 128
    dtype = torch.bfloat16
    device = "cuda"
    import time     
    qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    fn = lambda: flash_attn_func(qkv, causal=True)
    BATCH, H, N_CTX, D_HEAD = 32, 4, 2048, 256
    q = torch.randn((BATCH, H, N_CTX, D_HEAD//2), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD//2), dtype=dtype, device="cuda", requires_grad=True)        

    gk = torch.randn((BATCH, H, N_CTX, D_HEAD//2), dtype=dtype, device="cuda", requires_grad=True)        

    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)

    # gv = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)


    fn2 = lambda: gated_linear_attention(q, k, v, gk, None, num_head=H, chunk_size=256)


    for _ in range(1000):
        fn2()


