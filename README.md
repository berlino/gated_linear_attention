# Gated Linear Attention Layer


Standalone module of Gated Linear Attention (GLA) from [Gated Linear Attention Transformers with
Hardware-Efficient Training](https://arxiv.org/pdf/2312.06635.pdf). 

```
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

Warning: ```fused_chunk``` mode needs Triton2.2 + CUDA12 (See [issue](https://github.com/berlino/gated_linear_attention/issues/8 )). You can use [test](https://github.com/sustcsonglin/flash-linear-attention/blob/main/tests/test_fused_chunk.py) to quickly see if you can use ```fused_chunk``` mode. If cannot, please refer to [link](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/gla.py#L44C1-L45C1) and use ```chunk``` mode instead.

## Usage

Load the checkpoint from huggingface.

```python
from gla_model import GLAForCausalLM
model = GLAForCausalLM.from_pretrained("bailin28/gla-1B-100B")
vocab_size = model.config.vocab_size
bsz, seq_len = 32, 2048
x = torch.randint(high=vocab_size, size=(bsz, seq_len))
model_output = model(x)
loss = model_output.loss
logits = model_output.logits
```
