"""gpt2-from-scratch: a verified reimplementation of GPT-2, used as an instrument.

The public surface is deliberately small. Import the model and its config, load
the published weights, tokenize, generate -- everything else (verification,
ablations, induction-head analysis) lives in modules you call explicitly.
"""

from transformer_internals.config import GPTConfig, TrainConfig, gpt2_config
from transformer_internals.model import GPT, Block, CausalSelfAttention, KVCache, gelu_tanh
from transformer_internals.sampling import generate
from transformer_internals.tokenizer import BPETokenizer

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "BPETokenizer",
    "Block",
    "CausalSelfAttention",
    "GPTConfig",
    "KVCache",
    "TrainConfig",
    "gelu_tanh",
    "generate",
    "gpt2_config",
]
