# Gated Linear Attention Layer


Standalone module of Gated Linear Attention (GLA) from [Gated Linear Attention Transformers with
Hardware-Efficient Training](https://arxiv.org/pdf/2312.06635.pdf) 

## Setup

* torch (tested on 2.1.1+cu121)
* [triton nightly-release](https://github.com/openai/triton)
* einops

## Usage


```python
from gla import GatedLinearAttention

d_model = 1024
num_head = 4
use_gk = True # alpha
use_gv = False # beta
device = "cuda:0"

gla_layer = GatedLinearAttention(d_model, num_head, use_gk, use_gv).to(device)

bsz, seq_len, d_model = 32, 2048, 1024
x = torch.randn(bsz, seq_len, d_model).to(device)
y = gla_layer(x)

asssert y.shape == x.shape
```