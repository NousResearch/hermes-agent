"""PLUGIN-COMPAT stub (revert-scheduled; see COMPAT_MANIFEST.md).

``tools.environments.modal_utils`` was deleted in the Sep 2026 decomposition (its callers were folded into the
execution backends). Importing it no longer provides anything; this stub exists only so an external
plugin's ``import tools.environments.modal_utils`` does not raise at import time.
"""
