"""vault-retrieval plugin (token-efficient Vault context retrieval).

Profile-scoped user plugin. Enforces the locked contract in
``99 System/Hermes Token-Efficient Vault Retrieval Architecture - 2026-10-01.md``.

Subsurfaces:
- Tool: ``vault_context`` (read-only, bounded budgets, large-file refusal).
- System-prompt section: ``vault-retrieval-usage`` (bounded, after_memory).
- ``pre_tool_call`` hook: blocks direct ``read_file`` / ``search_files`` inside
  the configured Vault in enforce mode; audit-only in audit mode; ignored in
  off mode.

Operational guard only — arbitrary terminal/Python can still read files.
Adversarial containment requires future core/sandbox work.
"""
from __future__ import annotations

from .plugin import register

__all__ = ["plugin_id", "register"]


plugin_id = "vault-retrieval"
