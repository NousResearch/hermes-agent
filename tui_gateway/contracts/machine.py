"""This machine (``methods_machine.py``): host facts from ``hermes_platform.host.summary``.

Host-level by design: there is no ``profile`` param because the answer is the same for every
profile one backend serves.
"""

from __future__ import annotations

from .base import Params, Result
from .registry import method


class MachineInfo(Result):
    """Measured hardware; a key that was not measured is absent."""

    os_family: str | None = None
    os_release: str | None = None
    native_arch: str | None = None
    cpu_model: str | None = None
    ram_gb: int | None = None
    gpu_class: str | None = None
    vendor: str | None = None
    wsl: bool | None = None
    container: bool | None = None


class MachineFactsResult(Result):
    """``machine_kind`` is ``Spark``, ``Mac``, ``PC`` or ``computer``. ``full_name`` is the OS account's
    real name (never a login handle) and ``locale`` its first UI language; each is null when the OS has none."""

    machine: MachineInfo
    machine_kind: str
    has_nvidia_gpu: bool
    is_spark: bool
    locale: str | None = None
    full_name: str | None = None


method("machine.facts", params=Params, result=MachineFactsResult,
       doc="Facts about the machine running this backend and its OS account, for the desktop first run.")
