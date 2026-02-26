"""Tools Package

Individual tool implementations for the Hermes Agent.
Import from specific submodules (e.g., ``from tools.terminal_tool import ...``).

Tool discovery is handled by model_tools._discover_tools() which uses
importlib.import_module with try/except for graceful degradation.
"""
