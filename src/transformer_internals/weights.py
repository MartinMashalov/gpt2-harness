"""Load the published OpenAI GPT-2 weights into :class:`transformer_internals.model.GPT`.

``huggingface_hub`` is used to *fetch bytes*, and nothing else -- the tensors are
read straight out of the safetensors file and copied into our own parameters.
No ``transformers`` modelling code is imported here or anywhere on the forward
path. (``transformers`` appears exactly once in this repository, in
:mod:`transformer_internals.verify`, where ``GPT2LMHeadModel`` is the reference oracle
being compared *against*.)

The one real transformation is a transpose. GPT-2's original TensorFlow code
used 1-D convolutions for its projections, and HuggingFace preserved that as its
``Conv1D`` class, which stores weights as ``(in_features, out_features)`` and
computes ``x @ W``. ``nn.Linear`` stores ``(out_features, in_features)`` and
computes ``x @ W.T``. So every ``c_attn`` / ``c_fc`` / ``c_proj`` weight is
transposed on the way in; the biases and the embeddings are copied verbatim.

Getting this wrong is the classic silent failure: for the square ``attn.c_proj``
(768x768) a missing transpose still *runs*, still produces fluent-looking text,
and is completely wrong. That is precisely the class of bug the verification
suite in :mod:`transformer_internals.verify` exists to catch, and it is why "the loss went
down and the samples look fine" is not evidence of correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from transformer_internals.config import GPTConfig, gpt2_config
from transformer_internals.model import GPT

__all__ = [
    "CONV1D_TRANSPOSED",
    "convert_hf_state_dict",
    "load_gpt2_state_dict",
    "load_pretrained_gpt2",
    "resolve_checkpoint_dir",
]

#: Suffixes whose weights are stored ``(in, out)`` by HuggingFace's ``Conv1D``
#: and must be transposed for ``nn.Linear``. Biases are 1-D and never transposed.
CONV1D_TRANSPOSED = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

#: Buffers in the checkpoint that are not parameters: the causal mask HF bakes
#: into every block, and the large-negative constant it masks with. We build our
#: own mask from the config, so these are dropped.
#:
#: The leading dots matter. Matching on the bare suffix ``"attn.bias"`` also
#: swallows ``attn.c_attn.bias`` -- the qkv projection bias -- and a model
#: missing its qkv bias still loads, still runs, and is wrong.
_IGNORED_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def resolve_checkpoint_dir(
    model_id: str = "openai-community/gpt2", local_files_only: bool = False
) -> Path:
    """Find a local directory holding the GPT-2 checkpoint, downloading if needed.

    Args:
        model_id: HuggingFace repo id.
        local_files_only: Never touch the network; raise if the cache is cold.

    Returns:
        Path to a snapshot directory containing ``model.safetensors``,
        ``vocab.json`` and ``merges.txt``.

    Raises:
        FileNotFoundError: If the weights are not cached and cannot be fetched.
    """
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            model_id,
            allow_patterns=["config.json", "model.safetensors", "vocab.json", "merges.txt"],
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"GPT-2 weights for {model_id!r} are not available offline and could "
            f"not be downloaded ({exc}). Weight-dependent tests are marked "
            f"`weights` and can be deselected with `-m 'not weights'`."
        ) from exc
    return Path(path)


def load_gpt2_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read the raw tensors out of ``model.safetensors``.

    Args:
        checkpoint_dir: Directory containing ``model.safetensors``.

    Returns:
        The checkpoint's tensors, untouched, keyed as stored.
    """
    from safetensors.torch import load_file

    path = Path(checkpoint_dir) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"no model.safetensors under {checkpoint_dir}")
    return load_file(str(path))


def convert_hf_state_dict(
    hf_state: dict[str, torch.Tensor], config: GPTConfig
) -> dict[str, torch.Tensor]:
    """Map HuggingFace GPT-2 tensor names/layouts onto ours.

    Two checkpoint conventions are accepted: the bare ``GPT2Model`` layout used by
    the released ``openai-community/gpt2`` safetensors (``h.0.attn...``) and the
    ``GPT2LMHeadModel`` layout (``transformer.h.0.attn...``), which is what
    ``model.state_dict()`` gives you in code.

    Args:
        hf_state: Tensors as stored by HuggingFace.
        config: Target config, used to sanity-check shapes.

    Returns:
        A state dict loadable by :class:`GPT` with ``strict=False`` (strict only
        because ``lm_head.weight`` is tied and therefore absent).

    Raises:
        ValueError: If a tensor's shape does not match the target model, which is
            the earliest possible point to catch a size mismatch.
    """
    out: dict[str, torch.Tensor] = {}
    for raw_key, tensor in hf_state.items():
        key = raw_key.removeprefix("transformer.")
        if key.endswith(_IGNORED_SUFFIXES):
            continue
        if key == "lm_head.weight":
            # Tied to wte in the released checkpoints; carrying it would only
            # duplicate 38.6M parameters. Our model ties them itself.
            continue
        if key.endswith(CONV1D_TRANSPOSED):
            tensor = tensor.t().contiguous()
        out[key] = tensor

    expected = {
        "wte.weight": (config.vocab_size, config.n_embd),
        "wpe.weight": (config.n_positions, config.n_embd),
        "ln_f.weight": (config.n_embd,),
    }
    for key, shape in expected.items():
        if key in out and tuple(out[key].shape) != shape:
            raise ValueError(f"{key}: checkpoint has {tuple(out[key].shape)}, model wants {shape}")
    return out


def load_pretrained_gpt2(
    size: str = "gpt2",
    checkpoint_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
    **config_overrides: Any,
) -> tuple[GPT, Path]:
    """Build a :class:`GPT` and load the published weights into it.

    Args:
        size: One of ``gpt2``, ``gpt2-medium``, ``gpt2-large``, ``gpt2-xl``.
        checkpoint_dir: Use this directory instead of resolving through the hub.
        device: Device to move the model to.
        dtype: Parameter dtype. fp32 by default -- the verification tolerances in
            this repository are fp32 tolerances, and fp16 would blow through them
            by three orders of magnitude for reasons that have nothing to do with
            whether the implementation is correct.
        local_files_only: Never touch the network.
        **config_overrides: Passed to :func:`transformer_internals.config.gpt2_config`.

    Returns:
        ``(model, checkpoint_dir)``. The model is in ``eval()`` mode with dropout
        already disabled by config, so a forward pass is deterministic.
    """
    model_id = "openai-community/gpt2" if size == "gpt2" else f"openai-community/{size}"
    path = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else resolve_checkpoint_dir(model_id, local_files_only=local_files_only)
    )

    # dropout=0.0: the released model is only ever used for inference here, and a
    # non-zero rate would make the verification numbers stochastic.
    config = gpt2_config(size, dropout=0.0, **config_overrides)
    model = GPT(config)
    converted = convert_hf_state_dict(load_gpt2_state_dict(path), config)

    missing, unexpected = model.load_state_dict(converted, strict=False)
    # ``lm_head.weight`` is the only legitimate absence: it is the same tensor as
    # ``wte.weight``, which we did load.
    unexplained = [k for k in missing if k != "lm_head.weight"]
    if unexplained or unexpected:
        raise RuntimeError(
            f"state dict mismatch: missing={unexplained} unexpected={list(unexpected)}"
        )

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, path
