"""Corpus loading and batching for the ablation grid.

The corpus is the validation split of TinyStories (Eldan & Li, 2023) -- short,
simple, syntactically clean stories written with a small vocabulary. That choice
is deliberate. The ablation models here are 6-layer, 256-wide toys trained for a
few hundred steps; on WikiText or Gutenberg prose they would all sit at
essentially the same high loss, dominated by irreducible entropy, and every
architectural comparison would read as "no difference" for the boring reason that
nothing had learned anything yet. On TinyStories a model that small genuinely
learns structure inside a minute, so the differences the grid is trying to
measure are actually above the noise floor.

Tokenization uses this repository's own BPE, so the ablations and the
verification are on the same footing. The token stream is cached to ``data/`` as
a ``.npy`` file, keyed by a hash of the source and settings -- encoding 20 MB of
text with a pure-Python BPE takes about a minute and there is no reason to pay it
on every run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from transformer_internals.tokenizer import BPETokenizer

__all__ = [
    "DEFAULT_DATA_DIR",
    "TokenDataset",
    "compact_vocabulary",
    "encode_corpus",
    "load_text",
]

DEFAULT_DATA_DIR = Path("data")

#: TinyStories' validation split: ~19 MB, enough for a 5M-token corpus, and small
#: enough to fetch in seconds. The train split is 2 GB and would be pure waste
#: here -- these runs see at most a few million tokens.
_TINYSTORIES = ("roneneldan/TinyStories", "TinyStories-valid.txt")


def load_text(
    max_chars: int | None = 8_000_000,
    local_files_only: bool = False,
) -> str:
    """Fetch the raw corpus text.

    Args:
        max_chars: Truncate to this many characters. ``None`` keeps everything.
            The default is ~2M tokens, comfortably more than the grid consumes.
        local_files_only: Never touch the network; requires a warm HF cache.

    Returns:
        The corpus as a single string.

    Raises:
        FileNotFoundError: If the corpus is neither cached nor downloadable.
    """
    from huggingface_hub import hf_hub_download

    repo, filename = _TINYSTORIES
    try:
        path = hf_hub_download(
            repo, filename, repo_type="dataset", local_files_only=local_files_only
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"could not obtain {repo}/{filename} ({exc}). Ablation runs need it; "
            f"they are not part of the CI test suite for exactly this reason."
        ) from exc
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text if max_chars is None else text[:max_chars]


def encode_corpus(
    tokenizer: BPETokenizer,
    text: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_chars: int | None = 8_000_000,
    local_files_only: bool = False,
) -> np.ndarray:
    """Tokenize the corpus, caching the result on disk.

    Args:
        tokenizer: Tokenizer to encode with.
        text: Pre-loaded text. If ``None``, :func:`load_text` is called.
        data_dir: Where to write the cache.
        max_chars: Passed to :func:`load_text`.
        local_files_only: Passed to :func:`load_text`.

    Returns:
        A 1-D ``uint16`` array of token ids. ``uint16`` is safe because GPT-2's
        vocabulary is 50257 < 65536, and it halves both the file and the memory.
    """
    if text is None:
        text = load_text(max_chars=max_chars, local_files_only=local_files_only)

    key = hashlib.sha256(
        f"{len(text)}|{text[:4096]}|{tokenizer.vocab_size}".encode()
    ).hexdigest()[:16]
    cache = Path(data_dir) / f"tokens_{key}.npy"
    if cache.exists():
        return np.load(cache)

    # Encode in chunks split on blank lines so no BPE word is cut in half at a
    # chunk boundary -- doing it naively by character count would create a
    # handful of bogus tokens and make the cache depend on the chunk size.
    ids: list[int] = []
    for para in text.split("\n\n"):
        if para:
            ids.extend(tokenizer.encode(para + "\n\n"))
    arr = np.asarray(ids, dtype=np.uint16)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


class TokenDataset:
    """A flat token stream sliced into (input, target) windows.

    Sampling is with replacement from uniformly random offsets, which is the
    standard "sample a random crop" regime. The alternative -- a fixed epoch over
    non-overlapping blocks -- gives every token exactly one position in the
    context, so the model never sees a given token both early and late in a
    window, and short runs then overfit the alignment rather than the data.

    Args:
        tokens: 1-D array of token ids.
        block_size: Context length.
        val_fraction: Tail fraction held out. The split is contiguous, not
            random: with a random split, windows straddling the boundary leak
            training tokens into validation, and the val loss reads low for a
            reason that has nothing to do with generalisation.
    """

    def __init__(self, tokens: np.ndarray, block_size: int, val_fraction: float = 0.1) -> None:
        if tokens.ndim != 1:
            raise ValueError(f"expected a 1-D token stream, got shape {tokens.shape}")
        n_val = int(len(tokens) * val_fraction)
        if len(tokens) - n_val <= block_size + 1:
            raise ValueError(
                f"corpus of {len(tokens)} tokens is too short for block_size={block_size}"
            )
        self.block_size = block_size
        self.train = tokens[: len(tokens) - n_val]
        self.val = tokens[len(tokens) - n_val :]

    def _split(self, split: str) -> np.ndarray:
        if split not in ("train", "val"):
            raise ValueError(f"unknown split {split!r}")
        return self.train if split == "train" else self.val

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a random batch of windows.

        Args:
            split: ``"train"`` or ``"val"``.
            batch_size: Number of windows.
            device: Destination device.
            generator: RNG, so that two arms of an ablation see byte-identical
                batches at every step. This is what makes the comparison a
                comparison rather than two independent noisy runs.

        Returns:
            ``(x, y)``, both ``(batch_size, block_size)`` int64, with ``y`` the
            input shifted one position left.
        """
        data = self._split(split)
        high = len(data) - self.block_size - 1
        ix = torch.randint(high, (batch_size,), generator=generator)
        # np.int64 indexing then a single from_numpy is measurably faster than
        # building a list of tensors and stacking.
        idx = ix.numpy().astype(np.int64)
        offsets = idx[:, None] + np.arange(self.block_size + 1, dtype=np.int64)[None, :]
        window = data[offsets].astype(np.int64)
        batch = torch.from_numpy(window)
        x = batch[:, :-1].contiguous().to(device)
        y = batch[:, 1:].contiguous().to(device)
        return x, y

    def sequential_batches(
        self, split: str, batch_size: int, limit: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic, non-overlapping windows -- used for perplexity.

        Perplexity must not be estimated on random crops: a random-crop estimate
        weights tokens near the start of the stream differently from those near
        the end and is not reproducible run to run. Walking the split in
        non-overlapping blocks gives every token exactly one prediction.

        Args:
            split: ``"train"`` or ``"val"``.
            batch_size: Windows per batch.
            limit: Stop after this many batches.

        Returns:
            A list of ``(x, y)`` pairs on CPU.
        """
        data = self._split(split)
        stride = self.block_size + 1
        n_windows = (len(data) - 1) // self.block_size
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, n_windows, batch_size):
            rows = []
            for w in range(start, min(start + batch_size, n_windows)):
                off = w * self.block_size
                rows.append(data[off : off + stride].astype(np.int64))
            if len(rows[-1]) < stride:
                rows.pop()
            if not rows:
                break
            batch = torch.from_numpy(np.stack(rows))
            out.append((batch[:, :-1].contiguous(), batch[:, 1:].contiguous()))
            if limit is not None and len(out) >= limit:
                break
        return out


def compact_vocabulary(
    tokens: np.ndarray, vocab_size: int = 4096
) -> tuple[np.ndarray, np.ndarray, float]:
    """Remap a token stream onto the ``vocab_size`` most frequent ids.

    The ablation models in this repository are a few million parameters wide, and
    with GPT-2's full 50257-token vocabulary the output projection alone is 12.9M
    parameters -- three quarters of the model -- and dominates both the forward
    FLOPs and the memory traffic. Every architectural comparison would then be
    run at a fraction of the steps the same wall-clock could otherwise buy, and
    most of the compute would be spent on an embedding table that none of the
    ablations touch.

    Restricting to the most frequent ids fixes that without changing what is being
    compared: the tokenizer is unchanged, the text is unchanged, and every arm
    sees exactly the same stream. It is a change to the *benchmark*, not to any
    arm of it.

    Rare tokens are mapped to id 0, which is therefore an ``<unk>``. On
    TinyStories the top 4096 GPT-2 tokens cover the large majority of occurrences,
    and the coverage is returned so the number can be reported rather than
    assumed.

    Args:
        tokens: 1-D array of GPT-2 token ids.
        vocab_size: Size of the compact vocabulary, including the ``<unk>`` slot.

    Returns:
        ``(remapped, id_map, coverage)`` where ``remapped`` holds compact ids,
        ``id_map`` maps compact id -> original GPT-2 id (index 0 is ``<unk>``),
        and ``coverage`` is the fraction of token occurrences that were kept.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    counts = np.bincount(tokens.astype(np.int64))
    keep = np.argsort(counts)[::-1][: vocab_size - 1]
    keep = keep[counts[keep] > 0]

    lookup = np.zeros(counts.shape[0], dtype=np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)

    remapped = lookup[tokens.astype(np.int64)]
    coverage = float(counts[keep].sum() / counts.sum())
    id_map = np.concatenate([np.array([-1], dtype=np.int64), keep.astype(np.int64)])
    return remapped.astype(np.int32), id_map, coverage
