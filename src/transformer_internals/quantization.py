"""Post-training weight quantization, with the arithmetic written out.

Symmetric linear quantization, implemented directly rather than via a library,
because the point is to show the arithmetic is understood:

    scale = max|W| / (2^(b-1) - 1)
    q     = clamp(round(W / scale), -2^(b-1), 2^(b-1) - 1)
    W_hat = q * scale

The single most consequential choice is the **granularity of ``scale``**.

*Per-tensor* uses one scale for the whole matrix. It is the cheapest thing to
store and the easiest to implement, and it is badly hurt by outliers: a single
large weight anywhere in a 768x2304 matrix sets the scale for all 1.8M entries,
and every small weight is then quantized to zero or one step. Transformer weight
matrices are exactly the pathological case, because a handful of channels carry
much larger magnitudes than the rest.

*Per-channel* uses one scale per output channel (per row of an ``nn.Linear``
weight). An outlier now only inflates the step size of its own row. The storage
cost is one extra fp32 per row -- for a 2304x768 matrix that is 2304 floats
against 1.77M weights, i.e. 0.5% overhead -- and the quality difference is large.
Measuring that gap is most of what this module is for.

**On speed.** These are *simulated* quantization measurements: weights are
quantized then dequantized back to fp32, so the forward pass is bit-for-bit what
a real int8 kernel would compute from the same quantized weights, but it runs at
fp32 speed. That is the honest framing. A genuine speedup needs an integer matmul
kernel, and PyTorch 2.2 ships no int4 kernel and no int8 kernel for MPS at all,
so a "tokens/sec after quantization" number produced here would measure nothing
except dequantization overhead. Tokens/sec is therefore reported *with the
measurement conditions stated*, and the size and quality numbers -- which are
real and hardware-independent -- carry the argument. The on-disk sizes are not
estimates either: :func:`packed_size_bytes` serialises genuinely packed tensors
(two int4 values per byte) and reports the resulting file size.
"""

from __future__ import annotations

import copy
import math
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_internals.data import TokenDataset
from transformer_internals.model import GPT

__all__ = [
    "TARGET_SUFFIXES",
    "QuantResult",
    "pack_int4",
    "packed_size_bytes",
    "perplexity_with_error_bars",
    "quantize_dequantize",
    "quantize_model",
    "quantize_tensor",
    "unpack_int4",
]

Granularity = Literal["per_tensor", "per_channel"]

#: Which weights get quantized. The block linears are ~85% of the parameters
#: outside the embedding and are where every real quantization scheme starts.
#: LayerNorm gains and all biases stay fp32: they are a rounding error in size
#: (0.03% of the model) and quantizing them costs real accuracy, which is the
#: worst trade available.
TARGET_SUFFIXES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def quantize_tensor(
    w: torch.Tensor, bits: int, granularity: Granularity = "per_channel"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix to signed integers.

    Args:
        w: ``(out_features, in_features)`` weight.
        bits: Bit width, e.g. 8 or 4.
        granularity: ``per_tensor`` (one scale) or ``per_channel`` (one scale per
            output row).

    Returns:
        ``(q, scale)`` where ``q`` is int8-typed integer codes in
        ``[-2^(b-1), 2^(b-1) - 1]`` and ``scale`` broadcasts against ``w``.

    Note:
        ``q`` is stored as ``int8`` even for 4-bit, because torch has no int4
        dtype. :func:`pack_int4` is what actually halves the bytes; keeping the
        codes in int8 here means the same dequantization path serves both widths.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    if granularity == "per_channel":
        # amax over the input dimension -> one scale per output row.
        amax = w.abs().amax(dim=1, keepdim=True)
    elif granularity == "per_tensor":
        amax = w.abs().amax()
    else:
        raise ValueError(f"unknown granularity {granularity!r}")

    # A row that is exactly zero would give scale 0 and a division by zero. Clamp
    # to the smallest positive normal instead: the row dequantizes back to zeros,
    # which is correct.
    scale = (amax / qmax).clamp(min=torch.finfo(torch.float32).tiny)
    q = torch.round(w / scale).clamp(qmin, qmax).to(torch.int8)
    return q, scale


def quantize_dequantize(
    w: torch.Tensor, bits: int, granularity: Granularity = "per_channel"
) -> torch.Tensor:
    """Round-trip a weight through quantization; returns the fp32 reconstruction.

    This is what "simulated quantization" means: ``W_hat`` is exactly the matrix a
    real integer kernel would be computing with, so the quality numbers are
    genuine even though the speed is not.
    """
    q, scale = quantize_tensor(w, bits, granularity)
    return (q.to(torch.float32) * scale).to(w.dtype)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack int4 codes two-per-byte.

    Args:
        q: Integer codes in ``[-8, 7]``, any shape with an even number of
            elements.

    Returns:
        A flat ``uint8`` tensor of half the length. The low nibble holds the
        even-indexed value, the high nibble the odd-indexed one; values are
        biased by +8 first so they fit an unsigned nibble.
    """
    flat = q.flatten().to(torch.int16) + 8
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int16)])
    low, high = flat[0::2], flat[1::2]
    return ((high << 4) | low).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`."""
    p = packed.to(torch.int16)
    low = (p & 0x0F) - 8
    high = ((p >> 4) & 0x0F) - 8
    out = torch.stack([low, high], dim=1).flatten()[:numel]
    return out.to(torch.int8)


def quantize_model(
    model: GPT,
    bits: int,
    granularity: Granularity = "per_channel",
    include_embedding: bool = False,
) -> tuple[GPT, dict[str, Any]]:
    """Return a copy of the model with its target weights quantize-dequantized.

    Args:
        model: Source model. Not modified.
        bits: Bit width.
        granularity: Scale granularity.
        include_embedding: Also quantize ``wte``. Off by default: under weight
            tying the embedding is also the output head, so quantizing it damages
            both the input representation and every logit, and it is the single
            most sensitive tensor in the model. The flag exists so that claim can
            be *measured* rather than asserted.

    Returns:
        ``(quantized_model, stats)`` where stats records what was quantized and
        the resulting packed size.
    """
    q_model = copy.deepcopy(model)
    quantized: list[str] = []
    n_quant_params = 0
    codes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for name, module in q_model.named_modules():
        is_target = name.endswith(TARGET_SUFFIXES)
        is_emb = include_embedding and name == "wte"
        if not (is_target or is_emb) or not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            w = module.weight.data
            q, scale = quantize_tensor(w.float(), bits, granularity)
            codes[name] = (q, scale)
            module.weight.data = (q.to(torch.float32) * scale).to(w.dtype)
        quantized.append(name)
        n_quant_params += w.numel()

    total_params = sum(
        p.numel() for p in {id(p): p for p in model.parameters()}.values()
    )
    stats = {
        "bits": bits,
        "granularity": granularity,
        "include_embedding": include_embedding,
        "n_quantized_tensors": len(quantized),
        "n_quantized_params": n_quant_params,
        "n_total_params": total_params,
        "fraction_quantized": n_quant_params / total_params,
        "packed_bytes": packed_size_bytes(q_model, codes, bits),
        "fp32_bytes": total_params * 4,
    }
    stats["compression_ratio"] = stats["fp32_bytes"] / stats["packed_bytes"]
    return q_model, stats


def packed_size_bytes(
    model: GPT, codes: dict[str, tuple[torch.Tensor, torch.Tensor]], bits: int
) -> int:
    """Serialise a genuinely packed checkpoint and return its real file size.

    Quantized tensors are written as packed integers plus fp32 scales; everything
    else (LayerNorm gains, biases, and any tensor left unquantized) is written as
    fp32. The file is written to a temporary path and measured, so the number in
    the results table is a size that actually exists on disk rather than a
    hand-computed estimate.

    Args:
        model: The quantized model, used for the tensors that were *not*
            quantized.
        codes: ``name -> (int codes, scale)`` for the tensors that were.
        bits: Bit width, which decides whether packing applies.

    Returns:
        File size in bytes.
    """
    payload: dict[str, torch.Tensor] = {}
    for name, (q, scale) in codes.items():
        payload[f"{name}.q"] = pack_int4(q) if bits == 4 else q
        payload[f"{name}.scale"] = scale.to(torch.float32)
        payload[f"{name}.shape"] = torch.tensor(list(q.shape), dtype=torch.int32)

    quantized_weight_names = {f"{n}.weight" for n in codes}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if name in quantized_weight_names or id(p) in seen:
            continue
        seen.add(id(p))
        payload[name] = p.detach().to(torch.float32)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        torch.save(payload, tmp)
        return tmp.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)


@torch.no_grad()
def perplexity_with_error_bars(
    model: GPT,
    dataset: TokenDataset,
    n_chunks: int = 8,
    batches_per_chunk: int = 2,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Held-out perplexity, with a spread computed over disjoint chunks.

    A single perplexity number has no error bar and therefore cannot support a
    claim like "int8 costs 0.3% quality". Splitting the held-out split into
    disjoint chunks and reporting the mean and standard deviation *across* chunks
    gives the scale of the variation the estimate itself carries, so a
    quantization delta can be compared against it.

    Args:
        model: Model to score.
        dataset: Data source.
        n_chunks: Number of disjoint chunks.
        batches_per_chunk: Batches per chunk.
        batch_size: Windows per batch.
        device: Device.

    Returns:
        Mean/std of loss and perplexity, plus the per-chunk values.
    """
    model = model.to(device).eval()
    batches = dataset.sequential_batches(
        "val", batch_size=batch_size, limit=n_chunks * batches_per_chunk
    )
    chunk_losses: list[float] = []
    for c in range(0, len(batches), batches_per_chunk):
        group = batches[c : c + batches_per_chunk]
        if not group:
            continue
        total, n = 0.0, 0
        for x, y in group:
            logits = model(x.to(device))["logits"].float().cpu()
            total += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            ).item()
            n += y.numel()
        chunk_losses.append(total / n)

    mean = statistics.fmean(chunk_losses)
    std = statistics.stdev(chunk_losses) if len(chunk_losses) > 1 else 0.0
    ppls = [math.exp(x) for x in chunk_losses]
    return {
        "loss_mean": mean,
        "loss_std": std,
        "ppl_mean": statistics.fmean(ppls),
        "ppl_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
        "ppl_of_mean_loss": math.exp(mean),
        "n_chunks": len(chunk_losses),
        "chunk_losses": chunk_losses,
    }


@dataclass
class QuantResult:
    """One row of the quantization table."""

    label: str
    bits: int
    granularity: str
    include_embedding: bool
    stats: dict[str, Any]
    quality: dict[str, Any]
    speed: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bits": self.bits,
            "granularity": self.granularity,
            "include_embedding": self.include_embedding,
            **self.stats,
            "quality": self.quality,
            "speed": self.speed,
        }
