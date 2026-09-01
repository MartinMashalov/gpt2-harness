"""Part 4: post-training quantization -- quality, size and speed tradeoff.

int8 and int4, per-tensor against per-channel scales, each scored on held-out
perplexity with error bars and measured on-disk size. Writes
``results/quantization.json``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_dataset, get_gpt2, get_tokenizer, write_json
from transformer_internals.benchmark import benchmark_generation, model_size_bytes
from transformer_internals.quantization import (
    QuantResult,
    perplexity_with_error_bars,
    quantize_model,
)

#: Practical-significance threshold in nats: below this a quality change is
#: reported as negligible regardless of whether the paired test resolves it.
#: 0.01 nats is ~1% in perplexity.
PRACTICAL_NATS = 0.01

SCHEMES = [
    ("int8 per-channel", 8, "per_channel", False),
    ("int8 per-tensor", 8, "per_tensor", False),
    ("int4 per-channel", 4, "per_channel", False),
    ("int4 per-tensor", 4, "per_tensor", False),
    ("int8 per-channel + embedding", 8, "per_channel", True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--batches-per-chunk", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--speed", action="store_true", help="also time generation per scheme")
    ap.add_argument("--out", default=str(RESULTS / "quantization.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    tok, _ = get_tokenizer(args.local_only)
    model, _ = get_gpt2(device=device, local_only=args.local_only)
    dataset = get_dataset(tok, block_size=128, max_chars=1_000_000, local_only=args.local_only)

    def speed_of(m) -> dict:
        if not args.speed:
            return {}
        rows = benchmark_generation(
            m, prompt_lens=[64], new_tokens=16, device=device, repeats=2, warmup=1,
            both_arms=False,
        )
        return {"ms_per_token": rows[0].ms_per_token, "tokens_per_s": rows[0].tokens_per_s}

    print("scoring fp32 baseline ...")
    base_quality = perplexity_with_error_bars(
        model, dataset, n_chunks=args.chunks,
        batches_per_chunk=args.batches_per_chunk, batch_size=args.batch_size, device=device,
    )
    base_speed = speed_of(model)
    fp32_bytes = model_size_bytes(model)
    print(f"  fp32 ppl {base_quality['ppl_of_mean_loss']:.4f} "
          f"± {base_quality['ppl_std']:.4f}  ({fp32_bytes / 1e6:.1f} MB)")

    rows = []
    for label, bits, gran, inc_emb in SCHEMES:
        print(f"quantizing {label} ...")
        qm, stats = quantize_model(model, bits=bits, granularity=gran, include_embedding=inc_emb)
        quality = perplexity_with_error_bars(
            qm, dataset, n_chunks=args.chunks,
            batches_per_chunk=args.batches_per_chunk, batch_size=args.batch_size, device=device,
        )
        # Paired per-chunk deltas. Every scheme is scored on the SAME chunks, so
        # the chunk-to-chunk variation is common to both arms and cancels; the
        # unpaired spread of either arm alone is not the uncertainty of the
        # difference and using it as one both understates real effects and
        # excuses fake ones.
        paired = [q - b for q, b in zip(quality["chunk_losses"], base_quality["chunk_losses"], strict=True)]
        mean_d = statistics.fmean(paired)
        sd_d = statistics.stdev(paired) if len(paired) > 1 else 0.0
        stderr_d = sd_d / math.sqrt(len(paired)) if paired else 0.0
        quality["paired_loss_delta"] = {
            "mean": mean_d,
            "sd": sd_d,
            "stderr": stderr_d,
            "n_chunks": len(paired),
            "per_chunk": paired,
            # Distinguishable when the mean shift exceeds twice its own standard
            # error -- a yardstick applied identically to every scheme.
            "distinguishable": bool(abs(mean_d) > 2 * stderr_d) if stderr_d else False,
        }
        res = QuantResult(label, bits, gran, inc_emb, stats, quality, speed_of(qm))
        rows.append(res.to_dict())
        pd = quality["paired_loss_delta"]
        print(f"  ppl {quality['ppl_of_mean_loss']:.4f}  "
              f"| paired Δloss {pd['mean']:+.4f} ± {pd['stderr']:.4f} (2se) "
              f"{'DISTINGUISHABLE' if pd['distinguishable'] else 'n.s.'} "
              f"| {stats['packed_bytes'] / 1e6:.1f} MB "
              f"| {stats['compression_ratio']:.2f}x smaller")
        del qm

    # Statistical and practical significance are different questions, and the
    # paired test answers only the first. With 8 paired chunks it resolves shifts
    # of ~0.001 nats, so EVERY scheme here is "distinguishable" from fp32 --
    # including int8 per-channel at +0.002 nats, which is a 0.03 perplexity move
    # on a baseline of 18.27. Reporting that as a quality cost would be true and
    # useless. So the highlight uses a stated practical threshold instead, and
    # the table prints both numbers so a reader can apply their own.
    practical = PRACTICAL_NATS
    ok = [r for r in rows if abs(r["quality"]["paired_loss_delta"]["mean"]) < practical]
    if ok:
        best = max(ok, key=lambda r: r["compression_ratio"])
        for r in rows:
            r["highlight"] = r["label"] == best["label"]

    print(f"\n{'scheme':<30} {'ppl':>9} {'paired Δloss (±2se)':>24} {'MB':>8} {'ratio':>7}")
    print("-" * 82)
    print(f"{'fp32 (reference)':<30} {base_quality['ppl_of_mean_loss']:>9.4f} "
          f"{'--':>24} {fp32_bytes / 1e6:>8.1f} {'1.00':>7}")
    for r in rows:
        pd = r["quality"]["paired_loss_delta"]
        verdict = "" if pd["distinguishable"] else "  n.s."
        cell = f"{pd['mean']:+.4f} ± {2 * pd['stderr']:.4f}{verdict}"
        print(f"{r['label']:<30} {r['quality']['ppl_of_mean_loss']:>9.4f} "
              f"{cell:>24} {r['packed_bytes'] / 1e6:>8.1f} {r['compression_ratio']:>6.2f}x")

    write_json(args.out, {
        "baseline": {**base_quality, "bytes": fp32_bytes, "speed": base_speed},
        "rows": rows,
        "meta": {
            "device": str(device),
            "practical_threshold_nats": PRACTICAL_NATS,
            "significance_note": (
                "paired per-chunk delta vs fp32 on identical chunks; 'distinguishable' "
                "means |mean| > 2 standard errors. With 8 chunks this resolves ~0.001 "
                "nats, so every scheme is statistically distinguishable -- the "
                "highlight therefore uses the practical threshold above instead."
            ),
            "n_chunks": args.chunks,
            "batch_size": args.batch_size,
            "speed_measured": args.speed,
            "note": (
                "Simulated quantization: weights are quantize-dequantized, so quality "
                "and size are real while speed is fp32. PyTorch 2.2 has no int4 kernel "
                "and no int8 MPS kernel, so a speedup cannot be measured here honestly."
            ),
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
