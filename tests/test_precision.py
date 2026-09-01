"""Mixed precision: the policy table, the bf16 path, and what a bf16 wire costs.

Two halves.

The first is the policy: which compute dtype each device gets, which refusals
are refusals rather than silent downgrades, and whether a ``GradScaler`` exists.
Those are pure decisions and are tested against a fabricated GPU.

The second is arithmetic that a laptop can settle. CPU bf16 autocast is the same
code path as CUDA bf16 autocast -- the same ``torch.autocast``, the same fp32
parameters, the same absence of a scaler -- so the claim "the bf16 path matches
the fp32 path to within X" is testable here, and the number is measured rather
than assumed. It is not a *speed* claim: CPU bf16 is not faster without
AVX512-BF16 or AMX, and no speed is asserted anywhere in this file.
"""

from __future__ import annotations

import pytest
import torch

from transformer_internals.config import GPTConfig, TrainConfig
from transformer_internals.data import TokenDataset
from transformer_internals.hardware import Capabilities, HardwareError
from transformer_internals.precision import (
    AmpPolicy,
    autocast_context,
    make_grad_scaler,
    master_weight_report,
    reduce_dtype_of,
    resolve_amp,
)
from transformer_internals.train import train

AMPERE = Capabilities.stub(device_count=8, capability=(8, 0))
VOLTA = Capabilities.stub(device_count=8, capability=(7, 0))


# --------------------------------------------------------------------------- #
# the policy table
# --------------------------------------------------------------------------- #


def test_amp_off_is_fp32_everywhere():
    for device in ("cpu", "cuda", "mps"):
        policy = resolve_amp(False, "bf16", device)
        assert policy.enabled is False
        assert policy.name == "fp32"
        assert policy.needs_scaler is False


def test_bf16_on_ampere_needs_no_scaler():
    policy = resolve_amp(True, "bf16", "cuda", AMPERE)
    assert policy.enabled and policy.name == "bf16"
    # The whole reason bf16 is preferred: it carries fp32's exponent range, so
    # there is nothing to loss-scale.
    assert policy.needs_scaler is False
    assert make_grad_scaler(policy) is None


def test_fp16_on_cuda_gets_a_real_scaler():
    policy = resolve_amp(True, "fp16", "cuda", AMPERE)
    assert policy.enabled and policy.name == "fp16"
    assert policy.needs_scaler is True


def test_bf16_on_pre_ampere_is_refused_and_names_the_alternative():
    with pytest.raises(HardwareError, match="amp_dtype='fp16'"):
        resolve_amp(True, "bf16", "cuda", VOLTA)


def test_fp16_on_cpu_is_refused_rather_than_silently_ignored():
    with pytest.raises(HardwareError, match="not supported on CPU"):
        resolve_amp(True, "fp16", "cpu")


def test_mps_stays_fp32_and_says_why():
    policy = resolve_amp(True, "bf16", "mps")
    assert policy.enabled is False
    assert "LayerNorm" in policy.reason


def test_master_weights_are_fp32_on_every_policy():
    for policy in (
        resolve_amp(True, "bf16", "cuda", AMPERE),
        resolve_amp(True, "fp16", "cuda", AMPERE),
        resolve_amp(True, "bf16", "cpu"),
        resolve_amp(False, "bf16", "cpu"),
    ):
        assert policy.master_dtype is torch.float32
        assert policy.to_dict()["master_weight_dtype"] == "float32"


def test_unknown_dtype_names_are_rejected():
    with pytest.raises(ValueError, match="fp8"):
        resolve_amp(True, "fp8", "cuda", AMPERE)
    with pytest.raises(ValueError, match="fp8"):
        reduce_dtype_of("fp8")


def test_the_fp16_scaler_is_a_real_scaler_with_the_whole_interface():
    """As far as this machine can check it, which is the construction and the API.

    The fp16 branch cannot be executed here: fp16 autocast needs CUDA, and a
    GradScaler on a CPU-only build disables itself. What can be checked is that
    the factory finds a scaler at all, which is not free -- torch.amp.GradScaler
    is the current spelling and arrived in 2.3, torch.cuda.amp.GradScaler is the
    old one and is deprecated from 2.4, and this package supports 2.2 through
    2.5 -- and that the object it returns has the four methods the training loop
    calls on it.
    """
    import warnings

    policy = resolve_amp(True, "fp16", "cuda", AMPERE)
    with warnings.catch_warnings():
        # "CUDA is not available. Disabling." on this machine, which is the point.
        warnings.simplefilter("ignore")
        scaler = make_grad_scaler(policy)

    assert scaler is not None
    for method in ("scale", "unscale_", "step", "update"):
        assert callable(getattr(scaler, method)), method
    # Self-disabled here, because there is no CUDA. Asserted rather than left
    # implicit, so the limit of what this test covers is written down.
    assert scaler.is_enabled() is torch.cuda.is_available()


def test_no_scaler_is_ever_built_for_a_path_that_does_not_need_one():
    """The bug this replaces: a disabled GradScaler on a bf16 path."""
    for policy in (
        resolve_amp(True, "bf16", "cuda", AMPERE),
        resolve_amp(True, "bf16", "cpu"),
        resolve_amp(False, "bf16", "cpu"),
    ):
        assert make_grad_scaler(policy) is None


# --------------------------------------------------------------------------- #
# autocast actually casting
# --------------------------------------------------------------------------- #


def test_autocast_context_changes_the_matmul_dtype_and_not_the_weights():
    layer = torch.nn.Linear(32, 32)
    x = torch.randn(4, 32)
    policy = resolve_amp(True, "bf16", "cpu")
    with autocast_context(policy):
        y = layer(x)
    assert y.dtype is torch.bfloat16
    # The parameter itself was never cast. This is what "fp32 master weights"
    # means under autocast, and it is the reason no second copy is needed.
    assert layer.weight.dtype is torch.float32
    assert master_weight_report(layer)["all_fp32"] is True


def test_autocast_off_leaves_everything_in_fp32():
    layer = torch.nn.Linear(32, 32)
    with autocast_context(AmpPolicy(False, "cpu", None, False, "off")):
        y = layer(torch.randn(4, 32))
    assert y.dtype is torch.float32


# --------------------------------------------------------------------------- #
# the bf16 training path against the fp32 one
# --------------------------------------------------------------------------- #
#
# What is compared here is the gradient of one step and the loss over many, and
# deliberately *not* the parameters after many steps. Adam divides by
# sqrt(g^2), so a half-percent perturbation of a gradient that is near zero
# turns into an update of the opposite sign, and after forty steps on a random
# corpus two runs that differ only in bf16 rounding have parameter vectors that
# differ by 59% of how far either of them moved. That number is real and it is
# about Adam's normalisation, not about bf16; it is the same effect that sets
# the 1e-5 tolerance on the ZeRO trajectory comparison. The gradient and the
# loss are the quantities where a bf16-vs-fp32 comparison means something.


TINY = GPTConfig(vocab_size=128, n_positions=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


def _tiny_dataset() -> TokenDataset:
    g = torch.Generator().manual_seed(3)
    return TokenDataset(torch.randint(0, 128, (20_000,), generator=g).numpy(), block_size=32)


def _one_step_gradient(amp: bool) -> tuple[torch.Tensor, float]:
    """The flattened gradient of one forward and backward, fp32 or bf16."""
    from transformer_internals.model import GPT

    torch.manual_seed(0)
    model = GPT(TINY).train()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 128, (8, 32), generator=g)
    y = torch.randint(0, 128, (8, 32), generator=g)
    with autocast_context(resolve_amp(amp, "bf16", "cpu")):
        loss = model(x, targets=y)["loss"]
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]), float(loss)


def test_bf16_gradients_match_fp32_to_bf16_resolution():
    """The stated tolerance is bf16's own resolution, and nothing looser.

    bf16 keeps 8 significand bits, so it resolves 2^-8 = 3.9e-03 relatively. A
    gradient that has been through a dozen bf16 matmuls should therefore differ
    from the fp32 one by a small multiple of that and by no more. 1e-02 is
    roughly two and a half times 2^-8, which is the room the accumulation over
    the block stack needs and no more than that.
    """
    g32, _ = _one_step_gradient(amp=False)
    g16, _ = _one_step_gradient(amp=True)

    relative = float((g16 - g32).norm() / g32.norm())
    assert relative < 1e-02, f"bf16 gradient off by {relative:.2e} relative"
    # Not zero. A bf16 run whose autocast silently did nothing would pass every
    # tolerance in this file, so the test asserts that something changed.
    assert relative > 1e-04, "autocast appears not to have engaged at all"
    # The gradients arrive in fp32 because the parameters are in fp32. This is
    # the observable consequence of fp32 master weights.
    assert g16.dtype is torch.float32


def _run(amp: bool, steps: int = 40) -> tuple[float, dict]:
    tcfg = TrainConfig(
        steps=steps,
        batch_size=8,
        block_size=32,
        lr=3e-3,
        warmup_steps=4,
        eval_interval=steps,
        eval_batches=4,
        seed=0,
        amp=amp,
        amp_dtype="bf16",
    )
    _model, result = train(TINY, tcfg, _tiny_dataset(), device="cpu")
    return result.final_val_loss, result.to_dict()["meta"]


@pytest.mark.slow
def test_bf16_and_fp32_reach_the_same_loss():
    """Forty steps apart, the two runs must agree on the loss to 0.1%.

    The loss is a scalar average over 4096 tokens, so the per-element bf16
    rounding largely cancels in it, which is why the tolerance here is far
    tighter than the one on the gradient above.
    """
    fp32_loss, fp32_meta = _run(amp=False)
    bf16_loss, bf16_meta = _run(amp=True)

    assert fp32_meta["precision"]["compute_dtype"] == "fp32"
    assert bf16_meta["precision"]["compute_dtype"] == "bf16"
    assert bf16_meta["precision"]["grad_scaler"] is False

    relative = abs(bf16_loss - fp32_loss) / fp32_loss
    assert relative < 1e-03, f"bf16 {bf16_loss:.6f} vs fp32 {fp32_loss:.6f} ({relative:.4%})"


@pytest.mark.slow
def test_the_bf16_run_keeps_its_weights_in_fp32():
    """Measured from the run's own metadata, not asserted in a docstring."""
    _loss, meta = _run(amp=True, steps=8)
    assert meta["master_weights"]["all_fp32"] is True
    assert meta["master_weights"]["fp32_fraction"] == 1.0
