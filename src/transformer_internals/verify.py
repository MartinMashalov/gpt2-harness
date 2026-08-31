"""Numerical equivalence against HuggingFace's ``GPT2LMHeadModel``.

This module is the reason the repository exists. A from-scratch transformer that
merely trains is not evidence of anything: a transposed projection, an off-by-one
causal mask, or the wrong GELU all still give fluent samples and a falling loss.
The only way to know an implementation is *correct* is to load weights that were
trained by someone else and check that it computes the same function they do.

``transformers`` is imported here, and only here, as the reference oracle. It is
never on the forward path of the model under test.

Four levels of evidence, each strictly harder to pass than the last:

1. **Layer-by-layer activations.** Every sub-module of every block, compared on a
   fixed batch. This localises a discrepancy instead of just detecting one -- if
   block 7's MLP is wrong, the table says so.
2. **Final logits.** One number, asserted in a test.
3. **Greedy generation, token-exact.** Hundreds of tokens from several prompts.
   Far more sensitive than the logit check: a 1e-4 logit error is invisible until
   two candidate tokens are within 1e-4 of each other, at which point the
   sequences diverge and never re-converge. Passing this over 300 tokens means
   the argmax agreed 300 consecutive times.
4. **Perplexity.** Both implementations on identical held-out windows.

**On tolerances.** These are floating-point computations, not symbolic ones. The
same maths in a different association order gives different answers: we compute
``(q @ k^T) / sqrt(d)`` where HF computes ``(q @ k^T) * (1/sqrt(d))``, our GELU is
a different expression tree, and matmul reductions are non-deterministic in
their blocking. In fp32 the accumulated difference on 124M parameters and 12
layers lands around 1e-4 absolute on logits whose own scale is ~1e2 -- i.e. a
relative error near 1e-6, which is the fp32 epsilon times the depth. The suite
asserts ``max |delta| < 1e-3`` on logits, roughly an order of magnitude above what
is observed, so it fails on a real bug and not on a BLAS version change. A
*bit-exact* assertion would be the wrong test: it would fail on hardware that is
perfectly correct.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from transformer_internals.data import TokenDataset
from transformer_internals.model import GPT
from transformer_internals.sampling import generate
from transformer_internals.tokenizer import BPETokenizer

__all__ = [
    "DEFAULT_PROMPTS",
    "LOGIT_TOLERANCE",
    "LayerDiff",
    "VerificationReport",
    "compare_activations",
    "compare_greedy_generation",
    "compare_logits",
    "compare_perplexity",
    "load_reference_model",
    "run_verification",
]

#: Absolute tolerance asserted on the final logits, in fp32. See the module
#: docstring for why this is 1e-3 and not 0.
LOGIT_TOLERANCE = 1e-3

#: Prompts for the token-exact generation check. Chosen to exercise different
#: regimes: factual recall, a repeated-structure prompt (which is where induction
#: heads engage and where a mask bug would show), code-ish text, and a prompt
#: whose continuation is high-entropy so the argmax margins are small.
DEFAULT_PROMPTS = [
    "The capital of France is",
    "In 1969, humans first walked on the",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Alice went to the store. Bob went to the store. Alice went to the",
    "The meaning of the word 'serendipity' is",
]


@dataclass
class LayerDiff:
    """One row of the layer-by-layer comparison table.

    Attributes:
        name: Dotted activation name, e.g. ``h.7.mlp``.
        max_abs: ``max |ours - reference|``.
        max_rel: ``max |ours - reference| / (|reference| + eps)``, computed only
            where ``|reference|`` exceeds a floor. Relative error on an activation
            that is legitimately ~0 is meaningless and would dominate the table
            with noise, so those entries are excluded rather than reported as
            enormous.
        mean_abs: Mean absolute difference, which says whether a large ``max_abs``
            is one outlier element or a systematic offset.
        ref_scale: ``max |reference|``, so the reader can judge the absolute
            numbers against the scale of the thing being compared.
        shape: Activation shape.
    """

    name: str
    max_abs: float
    max_rel: float
    mean_abs: float
    ref_scale: float
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "mean_abs": self.mean_abs,
            "ref_scale": self.ref_scale,
            "shape": list(self.shape),
        }


@dataclass
class VerificationReport:
    """Everything the verification run measured."""

    device: str
    dtype: str
    n_params: int
    layers: list[LayerDiff] = field(default_factory=list)
    logits_max_abs: float = float("nan")
    logits_max_rel: float = float("nan")
    logits_scale: float = float("nan")
    logit_tolerance: float = LOGIT_TOLERANCE
    generation: list[dict[str, Any]] = field(default_factory=list)
    perplexity: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def all_generations_match(self) -> bool:
        return bool(self.generation) and all(g["match"] for g in self.generation)

    @property
    def worst_layer(self) -> LayerDiff | None:
        return max(self.layers, key=lambda d: d.max_abs) if self.layers else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "n_params": self.n_params,
            "logit_tolerance": self.logit_tolerance,
            "logits": {
                "max_abs": self.logits_max_abs,
                "max_rel": self.logits_max_rel,
                "scale": self.logits_scale,
                "within_tolerance": bool(self.logits_max_abs < self.logit_tolerance),
            },
            "layers": [d.to_dict() for d in self.layers],
            "generation": self.generation,
            "all_generations_match": self.all_generations_match,
            "perplexity": self.perplexity,
            "timing": self.timing,
        }


def _diff(name: str, ours: torch.Tensor, ref: torch.Tensor, rel_floor: float = 1e-2) -> LayerDiff:
    """Compute one comparison row.

    Args:
        name: Activation name.
        ours: Our activation.
        ref: Reference activation.
        rel_floor: Reference magnitudes below this are excluded from the relative
            error, which is otherwise dominated by division by near-zero.

    Returns:
        The populated :class:`LayerDiff`.
    """
    if ours.shape != ref.shape:
        raise ValueError(f"{name}: shape {tuple(ours.shape)} vs reference {tuple(ref.shape)}")
    a = ours.detach().to(torch.float64).cpu()
    b = ref.detach().to(torch.float64).cpu()
    delta = (a - b).abs()
    mask = b.abs() > rel_floor
    rel = (delta[mask] / b.abs()[mask]).max().item() if bool(mask.any()) else 0.0
    return LayerDiff(
        name=name,
        max_abs=delta.max().item(),
        max_rel=rel,
        mean_abs=delta.mean().item(),
        ref_scale=b.abs().max().item(),
        shape=tuple(ours.shape),
    )


def load_reference_model(model_id: str = "openai-community/gpt2") -> Any:
    """Load the HuggingFace reference model in fp32 eval mode.

    Args:
        model_id: HuggingFace repo id.

    Returns:
        A ``GPT2LMHeadModel``. Imported lazily so that the rest of the package
        never depends on ``transformers`` being installed.
    """
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return model


def _reference_taps(ref: Any, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture the reference model's intermediate activations.

    Forward hooks are used rather than ``output_hidden_states`` alone, because
    the interesting rows -- the *sub-module* outputs inside a block -- are not
    exposed by the public API at all.

    The one row that needs care is ``resid_mid``, the residual stream after the
    attention add but before the MLP. HF never returns it, but it is exactly the
    tensor fed into ``ln_2``, so it is captured as ``ln_2``'s *input*.

    Args:
        ref: The reference model.
        x: ``(B, T)`` input ids.

    Returns:
        Activations keyed to match ours.
    """
    taps: dict[str, torch.Tensor] = {}
    handles = []

    def save_out(key: str, unwrap: bool = False) -> Callable[..., None]:
        def hook(_m: Any, _i: Any, out: Any) -> None:
            taps[key] = (out[0] if unwrap else out).detach()

        return hook

    def save_in(key: str) -> Callable[..., None]:
        def hook(_m: Any, inp: Any, _o: Any) -> None:
            taps[key] = inp[0].detach()

        return hook

    tr = ref.transformer
    for i, block in enumerate(tr.h):
        handles.append(block.ln_1.register_forward_hook(save_out(f"h.{i}.ln_1")))
        handles.append(block.attn.register_forward_hook(save_out(f"h.{i}.attn", unwrap=True)))
        # ln_2's input is the post-attention residual stream.
        handles.append(block.ln_2.register_forward_hook(save_in(f"h.{i}.resid_mid")))
        handles.append(block.ln_2.register_forward_hook(save_out(f"h.{i}.ln_2")))
        handles.append(block.mlp.register_forward_hook(save_out(f"h.{i}.mlp")))
    handles.append(tr.ln_f.register_forward_hook(save_out("ln_f")))
    # ln_f's INPUT is the last block's residual output. This matters: HF's
    # ``output_hidden_states`` applies ln_f before appending the final entry, so
    # ``hidden_states[-1]`` is the normalised stream, not the raw one. Comparing
    # our raw block-11 output against it reports a difference of ~1e2 on a
    # perfectly correct model -- a harness bug that looks exactly like a real one.
    handles.append(tr.ln_f.register_forward_hook(save_in("final_resid")))

    try:
        with torch.no_grad():
            out = ref(x, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    hidden = out.hidden_states
    taps["embed"] = hidden[0].detach()
    n_layer = len(tr.h)
    for i in range(n_layer):
        # hidden_states[i + 1] is the output of block i for every block except the
        # last, where the ln_f-input hook above holds the un-normalised tensor.
        taps[f"h.{i}.resid_out"] = (
            taps.pop("final_resid") if i == n_layer - 1 else hidden[i + 1].detach()
        )
    taps["logits"] = out.logits.detach()
    return taps


def compare_activations(
    model: GPT, ref: Any, x: torch.Tensor
) -> tuple[list[LayerDiff], torch.Tensor, torch.Tensor]:
    """Compare every named activation on one batch.

    Args:
        model: Our model.
        ref: The reference model.
        x: ``(B, T)`` input ids.

    Returns:
        ``(rows, our_logits, ref_logits)``. Rows are ordered by depth so the
        table reads as a walk through the network.
    """
    with torch.no_grad():
        ours = model(x, collect_taps=True)
    our_taps = ours["taps"]
    ref_taps = _reference_taps(ref, x)

    order = ["embed"]
    for i in range(model.config.n_layer):
        order += [f"h.{i}.{s}" for s in ("ln_1", "attn", "resid_mid", "ln_2", "mlp", "resid_out")]
    order += ["ln_f", "logits"]

    rows = [_diff(k, our_taps[k], ref_taps[k]) for k in order if k in our_taps and k in ref_taps]
    return rows, ours["logits"], ref_taps["logits"]


def compare_logits(ours: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """Return ``(max_abs, max_rel, ref_scale)`` for the final logits."""
    d = _diff("logits", ours, ref)
    return d.max_abs, d.max_rel, d.ref_scale


def compare_greedy_generation(
    model: GPT,
    ref: Any,
    tokenizer: BPETokenizer,
    prompts: list[str] | None = None,
    max_new_tokens: int = 300,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Greedy-decode the same prompts with both models and compare token by token.

    Args:
        model: Our model.
        ref: The reference model.
        tokenizer: Our tokenizer (also used to encode the reference's input, so a
            tokenizer bug would show up as a generation mismatch here too).
        prompts: Prompt strings. Defaults to :data:`DEFAULT_PROMPTS`.
        max_new_tokens: Tokens to generate per prompt.
        device: Device for our model.

    Returns:
        One record per prompt with the match flag and, on a mismatch, the index
        of the first differing token -- which is the single most useful number
        for debugging, since it says how deep into the sequence the error took to
        surface.
    """
    prompts = list(DEFAULT_PROMPTS if prompts is None else prompts)
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        ours = generate(model, x, max_new_tokens, do_sample=False, use_cache=True)[0].tolist()
        with torch.no_grad():
            theirs = ref.generate(
                torch.tensor([ids], dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eot_token_id,
            )[0].tolist()

        match = ours == theirs
        first_div = None
        if not match:
            for i, (a, b) in enumerate(zip(ours, theirs, strict=False)):
                if a != b:
                    first_div = i
                    break
            if first_div is None:
                first_div = min(len(ours), len(theirs))
        records.append(
            {
                "prompt": prompt,
                "n_prompt_tokens": len(ids),
                "n_new_tokens": len(ours) - len(ids),
                "match": match,
                "first_divergence": first_div,
                "text": tokenizer.decode(ours),
            }
        )
    return records


def compare_perplexity(
    model: GPT,
    ref: Any,
    dataset: TokenDataset,
    n_batches: int = 8,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Perplexity of both models on identical held-out windows.

    Token-level perplexity is ``exp`` of the mean next-token negative
    log-likelihood. The mean is taken over *tokens*, not over batches, so a short
    final batch cannot skew it.

    Args:
        model: Our model.
        ref: The reference model.
        dataset: Provides the held-out split.
        n_batches: Number of non-overlapping batches to score.
        batch_size: Windows per batch.
        device: Device for our model.

    Returns:
        Both perplexities, both mean losses, the token count, and the absolute
        difference.
    """
    import torch.nn.functional as F

    batches = dataset.sequential_batches("val", batch_size=batch_size, limit=n_batches)
    ours_sum = ref_sum = 0.0
    n_tokens = 0
    for x, y in batches:
        with torch.no_grad():
            our_logits = model(x.to(device))["logits"].float().cpu()
            ref_logits = ref(x).logits.float()
        n = y.numel()
        ours_sum += F.cross_entropy(
            our_logits.reshape(-1, our_logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        ref_sum += F.cross_entropy(
            ref_logits.reshape(-1, ref_logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n_tokens += n

    ours_loss = ours_sum / n_tokens
    ref_loss = ref_sum / n_tokens
    import math

    return {
        "n_tokens": n_tokens,
        "n_batches": len(batches),
        "ours_loss": ours_loss,
        "reference_loss": ref_loss,
        "ours_ppl": math.exp(ours_loss),
        "reference_ppl": math.exp(ref_loss),
        "abs_ppl_diff": abs(math.exp(ours_loss) - math.exp(ref_loss)),
    }


def run_verification(
    model: GPT,
    tokenizer: BPETokenizer,
    dataset: TokenDataset | None = None,
    prompts: list[str] | None = None,
    max_new_tokens: int = 300,
    batch: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    ref: Any = None,
) -> VerificationReport:
    """Run every level of the verification and return the full report.

    Args:
        model: Our model, with the published weights loaded.
        tokenizer: Our tokenizer.
        dataset: Held-out data for the perplexity comparison. Skipped if absent.
        prompts: Generation prompts.
        max_new_tokens: Tokens per generation.
        batch: ``(B, T)`` fixed batch for the activation comparison. Built from
            the prompts if not supplied.
        device: Device for our model. The verification runs on CPU by default:
            MPS is fp32-but-not-quite in places, and a tolerance that has to
            absorb that is a tolerance that cannot detect a real bug.
        ref: A preloaded reference model, if you have one.

    Returns:
        The :class:`VerificationReport`.
    """
    ref = load_reference_model() if ref is None else ref

    if batch is None:
        # A fixed, reproducible batch: two prompts padded to a common length by
        # truncation, so no padding token is ever attended to.
        seqs = [tokenizer.encode(p) for p in (prompts or DEFAULT_PROMPTS)[:3]]
        n = min(len(s) for s in seqs)
        batch = torch.tensor([s[:n] for s in seqs], dtype=torch.long)
    batch = batch.to(device)

    report = VerificationReport(
        device=str(device),
        dtype=str(next(model.parameters()).dtype),
        n_params=model.num_parameters(),
    )

    t0 = time.perf_counter()
    rows, our_logits, ref_logits = compare_activations(model, ref, batch)
    report.layers = rows
    report.timing["activations_s"] = time.perf_counter() - t0

    max_abs, max_rel, scale = compare_logits(our_logits, ref_logits)
    report.logits_max_abs = max_abs
    report.logits_max_rel = max_rel
    report.logits_scale = scale

    t0 = time.perf_counter()
    report.generation = compare_greedy_generation(
        model, ref, tokenizer, prompts=prompts, max_new_tokens=max_new_tokens, device=device
    )
    report.timing["generation_s"] = time.perf_counter() - t0

    if dataset is not None:
        t0 = time.perf_counter()
        report.perplexity = compare_perplexity(model, ref, dataset, device=device)
        report.timing["perplexity_s"] = time.perf_counter() - t0

    return report
