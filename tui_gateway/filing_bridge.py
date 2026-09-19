"""Bridge to the correction-aware Hermes Projects filing contract.

The contract itself (rules / exclusions / append-only audit) is a user-home
artifact written by the ``project-filing`` gateway hook. This module gives the
gateway RPC surface a read/manage path over it without importing the user file
at module load:

* preferred library: a repo-local ``filing_contract`` module (tests, dev tree);
* fallback: the deployed ``~/project_filing_contract.py`` next to the user's
  home, loaded by path — the same file the live hook writes;
* contract file: ``<profile home>/project_filing.json`` (profile-aware via
  ``get_hermes_home()``), overridable with the ``PROJECT_FILING_CONTRACT``
  env var exactly like the deployed library.

Everything here is best-effort: when neither library is importable the RPC
handlers answer "filing unavailable" rather than raising, so an older or
stripped install degrades the More-pane entry instead of erroring.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

_CONTRACT_ENV = "PROJECT_FILING_CONTRACT"


def contract_path() -> Path:
    """The profile-aware contract file (env override wins)."""
    raw = os.environ.get(_CONTRACT_ENV, "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "project_filing.json"


def load_contract_lib() -> Optional[Any]:
    """The filing-contract library module, or None when nothing is installed.

    Order: an already-imported ``filing_contract`` (repo-local/test seam), then
    an importable ``filing_contract``, then the deployed user-home copy by path.
    """
    try:
        return sys.modules["filing_contract"]
    except KeyError:
        pass
    try:
        return importlib.import_module("filing_contract")
    except ImportError:
        pass
    deployed = Path(os.path.expanduser("~")) / "project_filing_contract.py"
    if not deployed.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("filing_contract", deployed)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["filing_contract"] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop("filing_contract", None)
        return None


def load_filing_hook_lib() -> Optional[Any]:
    """The live ``project-filing`` hook module (for its projects.db mirror
    helpers), or None when the hook is not installed."""
    try:
        return sys.modules["hermes_filing_hook"]
    except KeyError:
        pass
    hook_file = Path(os.path.expanduser("~")) / ".hermes" / "hooks" / "project-filing" / "handler.py"
    if not hook_file.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("hermes_filing_hook", hook_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["hermes_filing_hook"] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop("hermes_filing_hook", None)
        return None


def hook_installed() -> bool:
    """True when the project-filing gateway hook is present and declares events."""
    hook_dir = Path(os.path.expanduser("~")) / ".hermes" / "hooks" / "project-filing"
    manifest = hook_dir / "HOOK.yaml"
    if not (hook_dir / "handler.py").is_file() or not manifest.is_file():
        return False
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return False
    return "events:" in text and "agent:start" in text


def read_contract() -> dict:
    """Contract dict from the profile's contract file (empty when absent)."""
    pfc = load_contract_lib()
    if pfc is None:
        return {"version": 1, "rules": [], "exclusions": [], "audit": []}
    return pfc.load_contract(contract_path())
