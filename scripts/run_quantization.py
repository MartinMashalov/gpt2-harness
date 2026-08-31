"""Part 4: post-training quantization -- quality, size and speed tradeoff.

int8 and int4, per-tensor against per-channel scales, each scored on held-out
perplexity with error bars and measured on-disk size. Writes
``results/quantization.json``.
"""

from __future__ import annotations

import argparse
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
        res = QuantResult(label, bits, gran, inc_emb, stats, quality, speed_of(qm))
        rows.append(res.to_dict())
        print(f"  ppl {quality['ppl_of_mean_loss']:.4f} ± {quality['ppl_std']:.4f}  "
              f"| {stats['packed_bytes'] / 1e6:.1f} MB "
              f"| {stats['compression_ratio']:.2f}x smaller")
        del qm

    # Highlight the best scheme that stays within one baseline sd of fp32.
    tol = base_quality["ppl_std"]
    ok = [r for r in rows
          if r["quality"]["ppl_of_mean_loss"] - base_quality["ppl_of_mean_loss"] <= tol]
    if ok:
        best = max(ok, key=lambda r: r["compression_ratio"])
        for r in rows:
            r["highlight"] = r["label"] == best["label"]

    print(f"\n{'scheme':<30} {'ppl':>16} {'Δ ppl':>8} {'MB':>8} {'ratio':>7}")
    print("-" * 74)
    print(f"{'fp32 (reference)':<30} {base_quality['ppl_of_mean_loss']:>8.4f} "
          f"± {base_quality['ppl_std']:<5.4f} {'--':>8} {fp32_bytes / 1e6:>8.1f} {'1.00':>7}")
    for r in rows:
        d = r["quality"]["ppl_of_mean_loss"] - base_quality["ppl_of_mean_loss"]
        print(f"{r['label']:<30} {r['quality']['ppl_of_mean_loss']:>8.4f} "
              f"± {r['quality']['ppl_std']:<5.4f} {d:>+8.3f} "
              f"{r['packed_bytes'] / 1e6:>8.1f} {r['compression_ratio']:>6.2f}x")

    write_json(args.out, {
        "baseline": {**base_quality, "bytes": fp32_bytes, "speed": base_speed},
        "rows": rows,
        "meta": {
            "device": str(device),
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
