"""Part 7: measure what the KV cache actually buys.

Latency and throughput with and without the cache across context length, the
exact cache-memory growth against model size, and the GQA/MQA reduction.
Writes ``results/kv_cache.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_gpt2, write_json
from transformer_internals.benchmark import (
    attention_variant_memory,
    benchmark_generation,
    kv_cache_memory,
    measure_cache_tensor_bytes,
    model_size_bytes,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 128, 256, 512, 768])
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(RESULTS / "kv_cache.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    model, _ = get_gpt2(device=device, local_only=args.local_only)
    mbytes = model_size_bytes(model)
    print(f"device {device} | model {mbytes / 1e6:.1f} MB fp32")

    rows = benchmark_generation(
        model,
        prompt_lens=args.prompt_lens,
        new_tokens=args.new_tokens,
        batch_size=args.batch_size,
        device=device,
        repeats=args.repeats,
    )
    print(f"\n{'prompt':>7} {'cache':>7} {'ms/token':>10} {'tok/s':>9} {'speedup':>8}")
    print("-" * 46)
    by_len: dict[int, dict[bool, float]] = {}
    for r in rows:
        by_len.setdefault(r.prompt_len, {})[r.use_cache] = r.ms_per_token
    for r in rows:
        pair = by_len[r.prompt_len]
        speed = (pair[False] / pair[True]) if (True in pair and False in pair) else float("nan")
        tag = f"{speed:.2f}x" if r.use_cache else ""
        print(f"{r.prompt_len:>7} {'yes' if r.use_cache else 'no':>7} "
              f"{r.ms_per_token:>10.2f} {r.tokens_per_s:>9.1f} {tag:>8}")

    mem = kv_cache_memory(
        model.config,
        seq_lens=[128, 256, 512, 1024],
        batch_sizes=[1, 4, 8, 32],
        dtype_bytes=4,
        model_bytes=mbytes,
    )
    print(f"\n{'seq':>6} {'batch':>6} {'cache MB':>10} {'x model':>9}")
    print("-" * 34)
    for m in mem:
        if m["batch_size"] in (1, 8, 32):
            print(f"{m['seq_len']:>6} {m['batch_size']:>6} {m['cache_mb']:>10.1f} "
                  f"{m['cache_fraction_of_model']:>9.2f}")

    variants = attention_variant_memory(
        model.config, [None, 4, 2, 1], seq_len=1024, batch_size=8, dtype_bytes=4,
        model_bytes=mbytes,
    )
    print(f"\n{'variant':>9} {'kv heads':>9} {'cache MB':>10} {'reduction':>10}")
    print("-" * 42)
    for v in variants:
        print(f"{v['variant']:>9} {v['n_kv_head']:>9} {v['cache_mb']:>10.1f} "
              f"{v['reduction_vs_mha']:>9.1f}x")

    check = measure_cache_tensor_bytes(model, 64, 32, 2, device=str(device))
    print(f"\ncache formula check: measured {check['measured_bytes']:,} B vs "
          f"predicted {check['predicted_bytes']:,} B (ratio {check['ratio']:.4f})")

    write_json(args.out, {
        "generation": [r.to_dict() for r in rows],
        "cache_memory": mem,
        "attention_variants": variants,
        "formula_check": check,
        "meta": {
            "device": str(device),
            "model_bytes": mbytes,
            "new_tokens": args.new_tokens,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "n_params": model.num_parameters(),
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
