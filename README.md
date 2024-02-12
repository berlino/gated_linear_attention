# Gated Linear Attention Layer


Standalone module of Gated Linear Attention (GLA) from [Gated Linear Attention Transformers with
Hardware-Efficient Training](https://arxiv.org/pdf/2312.06635.pdf). 

```
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

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