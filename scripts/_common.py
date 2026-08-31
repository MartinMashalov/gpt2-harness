"""Shared setup for the runner scripts: device, tokenizer, corpus, model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from transformer_internals.data import TokenDataset, compact_vocabulary, encode_corpus
from transformer_internals.model import GPT
from transformer_internals.tokenizer import BPETokenizer
from transformer_internals.train import pick_device
from transformer_internals.weights import load_pretrained_gpt2, resolve_checkpoint_dir

RESULTS = Path("results")
ASSETS = Path("assets")


def device_from_arg(arg: str | None) -> torch.device:
    """Resolve a ``--device`` argument, auto-detecting when it is ``auto``."""
    return pick_device(None if arg in (None, "auto") else arg)


def get_tokenizer(local_only: bool = False) -> tuple[BPETokenizer, Path]:
    """Load the GPT-2 tokenizer from the local checkpoint snapshot."""
    d = resolve_checkpoint_dir(local_files_only=local_only)
    return BPETokenizer.from_pretrained(d), d


def get_dataset(
    tokenizer: BPETokenizer,
    block_size: int,
    max_chars: int = 4_000_000,
    local_only: bool = False,
) -> TokenDataset:
    """Tokenize (or load the cached tokenization of) the corpus."""
    tokens = encode_corpus(tokenizer, max_chars=max_chars, local_files_only=local_only)
    return TokenDataset(tokens, block_size=block_size)


def get_compact_dataset(
    tokenizer: BPETokenizer,
    block_size: int,
    vocab_size: int = 4096,
    max_chars: int = 4_000_000,
    local_only: bool = False,
) -> tuple[TokenDataset, float]:
    """Corpus remapped onto a compact vocabulary, for the small training runs."""
    tokens = encode_corpus(tokenizer, max_chars=max_chars, local_files_only=local_only)
    remapped, _, coverage = compact_vocabulary(tokens, vocab_size)
    return TokenDataset(remapped, block_size=block_size), coverage


def get_gpt2(device: str | torch.device = "cpu", local_only: bool = False) -> tuple[GPT, Path]:
    """Load our GPT-2 with the published weights."""
    return load_pretrained_gpt2(device=device, local_files_only=local_only)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write pretty JSON, creating parents."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return p


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a results JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
