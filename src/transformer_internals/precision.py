"""Mixed precision: which dtype the matmuls run in, and which the weights live in.

Three separate decisions get collapsed into the phrase "mixed precision", and
keeping them apart is most of the value of this module.

1. **Compute dtype.** What the GEMMs and the activations use. bf16 on any
   Ampere-or-newer GPU, fp16 on older tensor-core hardware.
2. **Master weight dtype.** What the optimiser updates. fp32, always, on every
   path here.
3. **Reduction dtype.** What the gradient all-reduce carries. A separate,
   independent choice, exposed as :data:`REDUCE_DTYPES` and defaulted to fp32.

Why bf16 rather than fp16
-------------------------
bf16 and fp32 have the same eight exponent bits, so they have the same dynamic
range: roughly 1e-38 to 3e38. fp16 has five exponent bits, and its smallest
normal value is about 6.1e-05. Real gradient magnitudes deep in a transformer
sit below that, so under fp16 they flush to zero and the parameter never
updates. The standard workaround is loss scaling -- multiply the loss by a large
constant before the backward pass, divide it out of the gradients before the
optimiser step, and back off whenever the scaled gradient overflows -- which is
what ``GradScaler`` implements and what makes an fp16 run a dynamical system
with its own failure mode.

bf16 needs none of that: no scaler, no overflow retries, no scale schedule. It
buys that with precision. bf16 keeps 8 significand bits against fp16's 11, so it
resolves about two to three decimal digits.

Why the master weights stay fp32
--------------------------------
That precision is the reason the weights cannot live in bf16. A late-training
AdamW update is commonly 1e-4 to 1e-6 times the weight it is applied to. Adding
a number 10^4 smaller than its neighbour in a format with three decimal digits
of resolution rounds straight back to the neighbour: the update is silently
discarded, and the model stops learning while the loss curve still looks
plausible. Keeping the parameters in fp32 keeps the update representable, and
the cost is one extra copy of the parameters -- 4N bytes rather than 2N, which
next to Adam's 8N of moments is not the term that decides whether a model fits.

Under ``torch.autocast`` the fp32 parameters **are** the master weights. Autocast
casts each weight to the compute dtype at the matmul, caches that cast for the
duration of the autocast region, and leaves the original alone; the gradients
land in fp32 on the fp32 parameters, and the optimiser updates those. There is
no second copy to maintain, which is worth saying because writing one by hand is
a common and unnecessary reimplementation of what autocast already does.

Reduction dtype
---------------
The gradient all-reduce is not covered by autocast, and it is a real choice.
Reducing in bf16 halves the bytes on the wire, which matters exactly when the
step is communication-bound. It costs accuracy twice over: the cast to bf16
truncates each rank's gradient to 8 significand bits, and the sum over ``p``
ranks then accumulates in that narrower format. PyTorch exposes the same knob as
``FullyShardedDataParallel(mixed_precision=MixedPrecision(reduce_dtype=...))``
and as DDP's ``_MixedPrecision.reduce_dtype``. Here it is
:func:`~transformer_internals.parallel.data_parallel.average_gradients`'s
``reduce_dtype``, it defaults to fp32, and
``tests/test_precision.py`` measures what choosing bf16 actually costs rather
than asserting that it is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from transformer_internals.hardware import Capabilities, HardwareError

__all__ = [
    "AMP_DTYPES",
    "REDUCE_DTYPES",
    "AmpPolicy",
    "autocast_context",
    "make_grad_scaler",
    "master_weight_report",
    "reduce_dtype_of",
    "resolve_amp",
]

#: Compute dtypes the training loop accepts, by their config spelling.
AMP_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

#: Dtypes the gradient all-reduce may carry. fp32 is the default and the one
#: every equivalence proof in this repository uses.
REDUCE_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def reduce_dtype_of(name: str | torch.dtype | None) -> torch.dtype:
    """Resolve a reduction dtype from its config spelling.

    Raises:
        ValueError: On an unknown name, listing the ones that exist.
    """
    if name is None:
        return torch.float32
    if isinstance(name, torch.dtype):
        return name
    try:
        return REDUCE_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown reduction dtype {name!r}; expected one of {sorted(REDUCE_DTYPES)}"
        ) from None


@dataclass(frozen=True)
class AmpPolicy:
    """The resolved mixed-precision decision for one run, and why.

    Attributes:
        enabled: Whether autocast wraps the forward pass at all.
        device_type: ``"cuda"``, ``"cpu"`` or ``"mps"``.
        dtype: The compute dtype, or None when autocast is off.
        needs_scaler: Whether a ``GradScaler`` is required. True only for fp16.
            bf16 has fp32's exponent range, so there is nothing to scale.
        master_dtype: What the parameters and the optimiser state stay in. fp32
            on every path; present as a field so a result file records it
            instead of a reader having to trust this docstring.
        reason: A sentence explaining the decision, carried into result files so
            a run that quietly did not use mixed precision says so.
    """

    enabled: bool
    device_type: str
    dtype: torch.dtype | None
    needs_scaler: bool
    reason: str
    master_dtype: torch.dtype = torch.float32

    @property
    def name(self) -> str:
        """``"bf16"``, ``"fp16"`` or ``"fp32"``."""
        if not self.enabled or self.dtype is None:
            return "fp32"
        return {torch.bfloat16: "bf16", torch.float16: "fp16"}[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "compute_dtype": self.name,
            "master_weight_dtype": str(self.master_dtype).replace("torch.", ""),
            "device_type": self.device_type,
            "grad_scaler": self.needs_scaler,
            "reason": self.reason,
        }


def resolve_amp(
    enabled: bool,
    dtype: str = "bf16",
    device: torch.device | str = "cpu",
    caps: Capabilities | None = None,
) -> AmpPolicy:
    """Decide the mixed-precision policy for a device, or refuse with a reason.

    Pure apart from the optional capability lookup, so the whole decision table
    can be tested against a fabricated GPU. The refusals are deliberate: a run
    that silently drops to fp32 on one rank and not another is not one run, and
    a comparison measured under different numerics from its baseline is not a
    comparison.

    Args:
        enabled: The config's ``amp`` flag.
        dtype: ``"bf16"`` or ``"fp16"``.
        device: Where the run will happen.
        caps: Machine description; detected when omitted and needed.

    Returns:
        An :class:`AmpPolicy`. ``enabled`` may be False even when asked for, on
        the one device where it is refused rather than raising (MPS).

    Raises:
        ValueError: On an unknown dtype name.
        HardwareError: When the request is impossible on this device, with the
            device, the request, and what to change.
    """
    dev = torch.device(device)
    off = AmpPolicy(False, dev.type, None, False, "autocast off, everything in fp32")
    if not enabled:
        return off
    if dtype not in AMP_DTYPES:
        raise ValueError(f"unknown amp dtype {dtype!r}; expected one of {sorted(AMP_DTYPES)}")
    want = AMP_DTYPES[dtype]

    if dev.type == "mps":
        # Not an oversight and not a raise. torch 2.2's MPS bf16 autocast
        # changes LayerNorm numerics, and the ablation grid on this machine is
        # small enough that fp32 costs little; an arm run under different
        # numerics from its baseline is not a comparison.
        return AmpPolicy(
            False,
            "mps",
            None,
            False,
            "MPS: bf16 autocast changes LayerNorm numerics on torch 2.2, so the "
            "run stays in fp32 rather than being incomparable with its baseline",
        )

    if dev.type == "cuda":
        if want is torch.bfloat16:
            caps = caps if caps is not None else Capabilities.detect()
            if not caps.bf16_supported:
                names = ", ".join(f"sm_{a}{b}" for a, b in caps.compute_capabilities) or "unknown"
                raise HardwareError(
                    f"bf16 autocast needs compute capability 8.0 (Ampere) or newer; "
                    f"this node reports {names}. Set amp_dtype='fp16' to use fp16 "
                    f"with a GradScaler, or amp=False to stay in fp32."
                )
            return AmpPolicy(
                True,
                "cuda",
                torch.bfloat16,
                False,
                "bf16 autocast with fp32 master weights; no GradScaler, because "
                "bf16 carries fp32's exponent range and nothing underflows",
            )
        return AmpPolicy(
            True,
            "cuda",
            torch.float16,
            True,
            "fp16 autocast with fp32 master weights and dynamic loss scaling; "
            "fp16's smallest normal is 6.1e-05, so small gradients underflow "
            "without a scaler",
        )

    if dev.type == "cpu":
        if want is torch.float16:
            raise HardwareError(
                "fp16 autocast is not supported on CPU in this torch range, and "
                "GradScaler is CUDA-only. Use amp_dtype='bf16' on CPU, or run on "
                "a CUDA device."
            )
        return AmpPolicy(
            True,
            "cpu",
            torch.bfloat16,
            False,
            "bf16 autocast on CPU with fp32 master weights. Numerically the same "
            "path as the CUDA one, which is what makes it testable without a GPU; "
            "it is not faster without AVX512-BF16 or AMX",
        )

    raise HardwareError(f"no mixed-precision policy for device type {dev.type!r}")


def autocast_context(policy: AmpPolicy) -> Any:
    """The autocast context for a policy, or a null context when it is off.

    One function so the training loop has no ``if use_amp`` around its forward
    pass. A conditional there is how the fp32 and mixed-precision paths drift
    apart.
    """
    if not policy.enabled or policy.dtype is None:
        import contextlib

        return contextlib.nullcontext()
    return torch.autocast(device_type=policy.device_type, dtype=policy.dtype)


def make_grad_scaler(policy: AmpPolicy) -> Any | None:
    """A ``GradScaler`` when the policy needs one, else None.

    Returns None rather than a disabled scaler. A disabled scaler on a path that
    never runs is how this repository previously carried a fp16 code path that
    had never executed; making its absence explicit means the training loop's
    fp16 branch is either real or not there.

    ``torch.amp.GradScaler`` is the current spelling and landed in torch 2.3;
    ``torch.cuda.amp.GradScaler`` is the older one and is deprecated from 2.4.
    This package supports 2.2 through 2.5, so both are tried.
    """
    if not policy.needs_scaler:
        return None
    try:
        return torch.amp.GradScaler(policy.device_type)  # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()  # torch 2.2


def master_weight_report(model: torch.nn.Module) -> dict[str, Any]:
    """What dtypes the parameters are actually in, counted by bytes.

    Recorded in the result files so "fp32 master weights" is a measurement of
    the run rather than a claim in a docstring. Under a correct autocast setup
    this reports 100% fp32 whatever the compute dtype was.
    """
    by_dtype: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype).replace("torch.", "")
        by_dtype[key] = by_dtype.get(key, 0) + p.numel() * p.element_size()
    total = sum(by_dtype.values())
    return {
        "bytes_by_dtype": by_dtype,
        "total_bytes": total,
        "all_fp32": set(by_dtype) == {"float32"},
        "fp32_fraction": by_dtype.get("float32", 0) / total if total else 0.0,
    }
