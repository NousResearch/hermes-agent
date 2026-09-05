"""Workflow documents and runs — HERMES_HOME-backed, gateway-readable.

The desktop canvas authors a scenario; this package is the durable half:
documents live under ``~/.hermes/workflows/``, a run walks that graph and
writes the same event log the canvas already folds (``protocol.ts``).
"""
