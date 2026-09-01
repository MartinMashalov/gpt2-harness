"""What accelerator is present, which distributed backend to use, and where tensors go.

Everything in this repository was written and measured on a machine with no
CUDA. The code still has to run correctly on an eight-GPU node without being
rewritten there, and the first run on rented hardware is the worst possible
place to discover a typo in a branch that has never executed. This module exists
so that the CUDA-specific decisions are made in one place, by pure functions,
against an explicit description of the machine.

The shape is deliberate:

* :class:`Capabilities` is a plain snapshot of what the machine has. It is built
  either by asking torch (:meth:`Capabilities.detect`) or by fabrication
  (:meth:`Capabilities.stub`).
* :func:`select_backend`, :func:`select_device` and :func:`check_placement` are
  pure functions of a :class:`Capabilities`. They contain the whole of the
  "which backend, which device, is this layout even possible" logic, and they
  can therefore be exercised on a laptop against a fabricated eight-GPU machine.
  That is what ``scripts/gpu_preflight.py --dry-run`` does.
* The impure part -- calling ``torch.cuda.set_device``, running ``nvidia-smi`` --
  is small, is at the bottom of this file, and every failure in it is turned
  into a :class:`HardwareError` carrying a sentence about what to do, rather
  than a traceback from three libraries down.

Nothing here decides anything by itself. A caller that wants gloo on a CUDA box
passes ``requested="gloo"`` and gets it, because the equivalence proofs are
worth running on both backends on the same hardware.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "BACKENDS",
    "Capabilities",
    "HardwareError",
    "check_placement",
    "describe",
    "environment_payload",
    "nvidia_smi",
    "select_backend",
    "select_device",
    "set_visible_device",
    "synchronize",
]

#: The two backends this repository supports. NCCL for CUDA, gloo for
#: everything else. MPI is not supported: it needs a torch built against an MPI
#: installation, which no published wheel is.
BACKENDS = ("nccl", "gloo")


class HardwareError(RuntimeError):
    """A hardware or placement problem, phrased as an instruction.

    Raised instead of letting a CUDA call fail three libraries deep. Every
    message says what was asked for, what the machine has, and what to change.
    """


@dataclass(frozen=True)
class Capabilities:
    """A snapshot of the accelerators and distributed backends this machine has.

    Attributes:
        cuda_available: ``torch.cuda.is_available()``.
        device_count: Visible CUDA devices. Zero when CUDA is unavailable.
        nccl_available: Whether this torch build can open a NCCL process group.
        gloo_available: Whether this torch build can open a gloo process group.
            False only on unusual builds, but checked rather than assumed.
        mps_available: Apple Metal, which is what the CPU-side development
            machine has and which supports neither NCCL nor bf16 autocast.
        device_names: One name per visible CUDA device.
        compute_capabilities: ``(major, minor)`` per visible CUDA device. bf16
            tensor cores need major >= 8 (Ampere), which is what decides whether
            the mixed-precision path can use bf16 or has to fall back to fp16.
        driver_version: NVIDIA driver version string, or None.
        cuda_runtime_version: The CUDA version torch was compiled against.
        nccl_version: NCCL version as reported by torch, or None.
        torch_version: For the record; a result JSON is worthless without it.
        source: ``"detected"`` or ``"stub"``. Present so nothing can quietly
            report a fabricated machine as a measured one.
    """

    cuda_available: bool = False
    device_count: int = 0
    nccl_available: bool = False
    gloo_available: bool = True
    mps_available: bool = False
    device_names: tuple[str, ...] = ()
    compute_capabilities: tuple[tuple[int, int], ...] = ()
    total_memory_bytes: tuple[int, ...] = ()
    driver_version: str | None = None
    cuda_runtime_version: str | None = None
    nccl_version: str | None = None
    torch_version: str = ""
    platform: str = ""
    source: str = "detected"
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- construction ------------------------------------------------------

    @classmethod
    def detect(cls) -> Capabilities:
        """Ask torch what is present, without letting any probe raise.

        Every accessor here is wrapped. ``torch.cuda.get_device_name`` on a box
        whose driver is mismatched raises, and a preflight check that dies while
        finding out what the machine is would be worse than useless.
        """
        notes: list[str] = []
        cuda = _safe(lambda: bool(torch.cuda.is_available()), False, notes, "cuda.is_available")
        count = _safe(lambda: int(torch.cuda.device_count()), 0, notes, "cuda.device_count") if cuda else 0
        names: list[str] = []
        caps: list[tuple[int, int]] = []
        mem: list[int] = []
        for i in range(count):
            names.append(_safe(lambda i=i: str(torch.cuda.get_device_name(i)), "unknown", notes, f"get_device_name({i})"))
            caps.append(_safe(lambda i=i: tuple(torch.cuda.get_device_capability(i)), (0, 0), notes, f"get_device_capability({i})"))
            mem.append(_safe(lambda i=i: int(torch.cuda.get_device_properties(i).total_memory), 0, notes, f"total_memory({i})"))

        nccl_ver = None
        if cuda:
            raw = _safe(lambda: torch.cuda.nccl.version(), None, notes, "nccl.version")
            if isinstance(raw, tuple):
                nccl_ver = ".".join(str(p) for p in raw)
            elif raw is not None:
                nccl_ver = str(raw)

        return cls(
            cuda_available=cuda,
            device_count=count,
            nccl_available=_safe(
                lambda: bool(torch.distributed.is_nccl_available()), False, notes, "is_nccl_available"
            ),
            gloo_available=_safe(
                lambda: bool(torch.distributed.is_gloo_available()), True, notes, "is_gloo_available"
            ),
            mps_available=_safe(
                lambda: bool(torch.backends.mps.is_available()), False, notes, "mps.is_available"
            ),
            device_names=tuple(names),
            compute_capabilities=tuple(caps),
            total_memory_bytes=tuple(mem),
            driver_version=_driver_version(),
            cuda_runtime_version=_safe(lambda: torch.version.cuda, None, notes, "version.cuda"),
            nccl_version=nccl_ver,
            torch_version=torch.__version__,
            platform=platform.platform(),
            source="detected",
            notes=tuple(notes),
        )

    @classmethod
    def stub(
        cls,
        device_count: int = 8,
        name: str = "NVIDIA A100-SXM4-80GB",
        capability: tuple[int, int] = (8, 0),
        total_memory_bytes: int = 80 * 1024**3,
        nccl_version: str = "2.19.3",
    ) -> Capabilities:
        """A fabricated CUDA machine, for exercising the CUDA branches with no CUDA.

        Used by ``--dry-run``. ``source`` is ``"stub"`` so that anything which
        serialises a :class:`Capabilities` into a result file carries the fact
        that the machine was invented.
        """
        return cls(
            cuda_available=device_count > 0,
            device_count=device_count,
            nccl_available=device_count > 0,
            gloo_available=True,
            mps_available=False,
            device_names=tuple([name] * device_count),
            compute_capabilities=tuple([capability] * device_count),
            total_memory_bytes=tuple([total_memory_bytes] * device_count),
            driver_version="stub",
            cuda_runtime_version="stub",
            nccl_version=nccl_version,
            torch_version=torch.__version__,
            platform=platform.platform(),
            source="stub",
            notes=("fabricated capabilities: no CUDA device was queried",),
        )

    # -- derived properties ------------------------------------------------

    @property
    def bf16_supported(self) -> bool:
        """Whether every visible device has bf16 tensor cores (Ampere or newer).

        ``all``, not ``any``: a job that runs bf16 on some ranks and fp16 on
        others is not one job, and the ranks would disagree numerically.
        """
        if not self.cuda_available or not self.compute_capabilities:
            return False
        return all(major >= 8 for major, _ in self.compute_capabilities)

    @property
    def accelerator(self) -> str:
        """``"cuda"``, ``"mps"`` or ``"cpu"``."""
        if self.cuda_available:
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, for embedding in a result file's ``meta``."""
        return {
            "source": self.source,
            "accelerator": self.accelerator,
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "device_names": list(self.device_names),
            "compute_capabilities": [list(c) for c in self.compute_capabilities],
            "total_memory_bytes": list(self.total_memory_bytes),
            "bf16_supported": self.bf16_supported,
            "nccl_available": self.nccl_available,
            "nccl_version": self.nccl_version,
            "gloo_available": self.gloo_available,
            "mps_available": self.mps_available,
            "driver_version": self.driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "torch_version": self.torch_version,
            "platform": self.platform,
            "notes": list(self.notes),
        }


def _safe(fn: Any, default: Any, notes: list[str], label: str) -> Any:
    """Call ``fn``; on any exception record it and return ``default``."""
    try:
        return fn()
    except Exception as exc:
        # A probe must never be the thing that fails: record and carry on.
        notes.append(f"{label} raised {type(exc).__name__}: {exc}")
        return default


def _driver_version() -> str | None:
    """The NVIDIA driver version, from ``nvidia-smi``. None if it is not there."""
    out = nvidia_smi("--query-gpu=driver_version", "--format=csv,noheader")
    if out is None:
        return None
    first = out.strip().splitlines()
    return first[0].strip() if first else None


# --------------------------------------------------------------------------- #
# pure decisions
# --------------------------------------------------------------------------- #


def select_backend(caps: Capabilities, requested: str | None = None) -> str:
    """Which ``torch.distributed`` backend to open.

    NCCL when there is CUDA and the build has NCCL, gloo otherwise. An explicit
    request is honoured if the machine can satisfy it and refused with a
    sentence if it cannot, because silently downgrading NCCL to gloo on a rented
    GPU box would produce numbers that look like collective measurements and are
    not.

    Args:
        caps: What the machine has.
        requested: ``"nccl"``, ``"gloo"``, ``"auto"`` or ``None``.

    Returns:
        ``"nccl"`` or ``"gloo"``.

    Raises:
        HardwareError: If the request cannot be satisfied.
    """
    if requested in (None, "auto"):
        if caps.cuda_available and caps.nccl_available and caps.device_count > 0:
            return "nccl"
        return "gloo"
    if requested not in BACKENDS:
        raise HardwareError(f"unknown backend {requested!r}; expected one of {BACKENDS} or 'auto'")
    if requested == "nccl":
        if not caps.cuda_available:
            raise HardwareError(
                "backend 'nccl' was requested but torch reports no CUDA device. "
                "Run with --backend auto to fall back to gloo, or fix the CUDA "
                "install (check `nvidia-smi` and that torch was installed from a "
                "cu-suffixed index)."
            )
        if not caps.nccl_available:
            raise HardwareError(
                "backend 'nccl' was requested but this torch build reports "
                "is_nccl_available() == False. This is a CPU-only or MPS wheel; "
                "reinstall torch from https://download.pytorch.org/whl/cu121 or "
                "equivalent."
            )
        return "nccl"
    if not caps.gloo_available:
        raise HardwareError("backend 'gloo' was requested but this torch build has no gloo")
    return "gloo"


def select_device(caps: Capabilities, rank: int, backend: str) -> str:
    """Which device string rank ``rank`` should place its tensors on.

    One GPU per rank, assigned by ``rank % device_count``. The modulo is not a
    convenience: it is what makes a world size larger than the GPU count run at
    all, which is how a smoke test with 8 ranks works on a 2-GPU box. It is also
    a performance trap on a real run, which is why :func:`check_placement`
    refuses it unless it was asked for explicitly.

    Args:
        caps: What the machine has.
        rank: Global rank.
        backend: The backend chosen by :func:`select_backend`.

    Returns:
        ``"cuda:<n>"`` or ``"cpu"``.

    Raises:
        HardwareError: If NCCL was chosen with no visible CUDA device, which
            means the backend selection and the device count disagree.
    """
    if backend != "nccl":
        return "cpu"
    if caps.device_count <= 0:
        raise HardwareError("backend 'nccl' with zero visible CUDA devices")
    return f"cuda:{rank % caps.device_count}"


def check_placement(
    caps: Capabilities, world_size: int, backend: str, allow_oversubscribe: bool = False
) -> list[str]:
    """Validate a world size against the machine, and return the per-rank devices.

    Args:
        caps: What the machine has.
        world_size: Ranks about to be launched.
        backend: The chosen backend.
        allow_oversubscribe: Permit more ranks than GPUs. Two ranks sharing a
            GPU serialise on it, so every timing measured that way is a
            measurement of contention. Allowed for smoke tests, refused by
            default.

    Returns:
        A device string per rank, in rank order.

    Raises:
        HardwareError: On an impossible or misleading placement.
    """
    if world_size < 1:
        raise HardwareError(f"world_size must be at least 1, got {world_size}")
    if backend == "nccl":
        if caps.device_count <= 0:
            raise HardwareError(
                "backend 'nccl' with zero visible CUDA devices. Check "
                "CUDA_VISIBLE_DEVICES; an empty string hides every GPU."
            )
        if world_size > caps.device_count and not allow_oversubscribe:
            raise HardwareError(
                f"world size {world_size} exceeds the {caps.device_count} visible CUDA "
                f"device(s). NCCL ranks sharing a GPU serialise on it, so every "
                f"timing from such a run measures contention. Reduce the world "
                f"size, or pass --allow-oversubscribe if you only want the "
                f"correctness proofs and not the timings."
            )
    return [select_device(caps, rank, backend) for rank in range(world_size)]


def describe(caps: Capabilities, topology: str | None = None) -> str:
    """A human-readable report of the machine, for the top of a run log."""
    lines: list[str] = []
    w = lines.append
    if caps.source == "stub":
        w("*** STUBBED CAPABILITIES: no CUDA device was queried ***")
    w(f"torch            {caps.torch_version}")
    w(f"platform         {caps.platform}")
    w(f"accelerator      {caps.accelerator}")
    w(f"CUDA available   {caps.cuda_available}  (devices: {caps.device_count})")
    for i, name in enumerate(caps.device_names):
        cap = caps.compute_capabilities[i] if i < len(caps.compute_capabilities) else (0, 0)
        gib = (caps.total_memory_bytes[i] / 1024**3) if i < len(caps.total_memory_bytes) else 0.0
        w(f"  [{i}] {name}  sm_{cap[0]}{cap[1]}  {gib:.0f} GiB")
    w(f"bf16 tensor core {caps.bf16_supported}")
    w(f"driver           {caps.driver_version or 'not reported'}")
    w(f"CUDA runtime     {caps.cuda_runtime_version or 'not reported'}")
    w(f"NCCL             available={caps.nccl_available} version={caps.nccl_version or 'n/a'}")
    w(f"gloo             available={caps.gloo_available}")
    for note in caps.notes:
        w(f"note: {note}")
    if topology:
        w("")
        w("nvidia-smi topo -m:")
        for line in topology.rstrip().splitlines():
            w(f"  {line}")
    return "\n".join(lines)


def environment_payload(caps: Capabilities | None = None, topology: bool = True) -> dict[str, Any]:
    """The machine description that every GPU-run result file carries.

    A number measured on an eight-GPU node and a number measured on a laptop are
    different numbers even when the code is identical, so the result files say
    which machine produced them.
    """
    caps = caps or Capabilities.detect()
    payload = caps.to_dict()
    payload["env"] = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "OMP_NUM_THREADS",
        )
        if os.environ.get(key) is not None
    }
    if topology:
        payload["nvlink_topology"] = nvidia_smi("topo", "-m")
    return payload


# --------------------------------------------------------------------------- #
# the impure edge
# --------------------------------------------------------------------------- #


def nvidia_smi(*args: str, timeout: float = 20.0) -> str | None:
    """Run ``nvidia-smi`` with the given arguments; None if it is absent or fails.

    Absent is the normal case on the development machine, so this returns None
    rather than raising. A GPU box where ``nvidia-smi`` is missing has a broken
    driver install, and the preflight script says so.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def set_visible_device(device: str) -> torch.device:
    """Bind this process to ``device``, returning it as a ``torch.device``.

    For CUDA this calls ``torch.cuda.set_device`` **before** any process group is
    opened. NCCL derives the device a rank owns from the current device at
    initialisation; getting this wrong puts two ranks on GPU 0 and hangs the
    first collective with no error message at all, which is the single most
    common way a first multi-GPU run fails.

    Raises:
        HardwareError: If the device cannot be selected, with the index and the
            count in the message.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return dev
    index = dev.index if dev.index is not None else 0
    try:
        count = torch.cuda.device_count()
    except Exception as exc:
        raise HardwareError(f"could not count CUDA devices: {exc}") from exc
    if index >= count:
        raise HardwareError(
            f"device {device} requested but only {count} CUDA device(s) are visible "
            f"(indices 0..{max(count - 1, 0)}). Check CUDA_VISIBLE_DEVICES."
        )
    try:
        torch.cuda.set_device(index)
    except Exception as exc:
        raise HardwareError(f"torch.cuda.set_device({index}) failed: {exc}") from exc
    return torch.device("cuda", index)


def synchronize(device: torch.device | str) -> None:
    """Block until queued work on ``device`` has finished.

    Every timing in this repository needs this and CPU needs none of it, so it
    is one function rather than a conditional at each call site. Without it a
    CUDA timing measures how fast Python can enqueue kernels.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()
