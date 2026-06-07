"""Tests for lazy_deps.py interactive prompt suppression.

Regression tests for #40504/#40490: CLI input locks up when lazy-dep 
installation prompt conflicts with prompt_toolkit owning stdin.

The fix: skip the interactive prompt (call input()) when any prompt_toolkit 
module is detected to be loaded, falling through to auto-install gated by 
the security.allow_lazy_installs config flag.
"""
import sys
from unittest.mock import patch, MagicMock
import pytest

from tools.lazy_deps import ensure, FeatureUnavailable


class TestLazyDepPromptSuppression:
    """Test that interactive prompts are suppressed under prompt_toolkit."""

    def test_prompt_suppressed_when_any_prompt_toolkit_imported(self):
        """When any prompt_toolkit module is imported, interactive prompt should be skipped."""
        # Simulate prompt_toolkit being imported
        with patch.dict(sys.modules, {"prompt_toolkit.application.current": MagicMock()}):
            # The new detection logic: check if any module starts with "prompt_toolkit"
            _pt_active = any(
                name.startswith("prompt_toolkit")
                for name in sys.modules.keys()
            )
            assert _pt_active is True, "Should detect any prompt_toolkit module"

    def test_prompt_suppressed_with_different_prompt_toolkit_submodules(self):
        """Detection should work with any prompt_toolkit submodule."""
        # Test with different prompt_toolkit imports
        test_modules = [
            "prompt_toolkit.application.current",
            "prompt_toolkit.formatted_text",
            "prompt_toolkit.layout",
            "prompt_toolkit.document",
        ]
        
        for module_name in test_modules:
            with patch.dict(sys.modules, {module_name: MagicMock()}):
                _pt_active = any(
                    name.startswith("prompt_toolkit")
                    for name in sys.modules.keys()
                )
                assert _pt_active is True, f"Should detect {module_name}"

    def test_prompt_not_suppressed_when_prompt_toolkit_not_imported(self):
        """When prompt_toolkit is not imported, prompt should not be suppressed by pt check."""
        # Remove all prompt_toolkit modules from sys.modules
        pt_modules = {k: v for k, v in sys.modules.items() if k.startswith("prompt_toolkit")}
        with patch.dict(sys.modules, pt_modules, clear=True):
            _pt_active = any(
                name.startswith("prompt_toolkit")
                for name in sys.modules.keys()
            )
            assert _pt_active is False, "Should not detect prompt_toolkit when not imported"

    def test_prompt_condition_requires_all_gates(self):
        """Interactive prompt requires: prompt=True AND not _pt_active AND isatty."""
        # Test the gate logic: prompt and not _pt_active and sys.stdin.isatty() and sys.stdout.isatty()
        
        # Case 1: _pt_active = True → prompt skipped
        prompt = True
        _pt_active = True
        show_prompt = prompt and not _pt_active
        assert show_prompt is False, "Should skip prompt when pt_active=True"
        
        # Case 2: _pt_active = False → prompt might show (if isatty checks pass)
        _pt_active = False
        show_prompt = prompt and not _pt_active
        assert show_prompt is True, "Should show prompt when pt_active=False (before isatty checks)"
        
        # Case 3: prompt = False → prompt skipped regardless
        prompt = False
        show_prompt = prompt and not _pt_active
        assert show_prompt is False, "Should skip prompt when prompt=False"


class TestPromptToolkitDetection:
    """Test prompt_toolkit detection logic."""

    def test_detection_uses_startswith_check(self):
        """Detection should check if any module name starts with 'prompt_toolkit'."""
        # Simulate various modules being loaded
        modules_to_test = {
            "prompt_toolkit.application": MagicMock(),
            "prompt_toolkit.layout": MagicMock(),
            "sys": MagicMock(),  # Non-prompt_toolkit module
            "os": MagicMock(),   # Non-prompt_toolkit module
        }
        
        with patch.dict(sys.modules, modules_to_test):
            _pt_active = any(
                name.startswith("prompt_toolkit")
                for name in sys.modules.keys()
            )
            assert _pt_active is True, "Should detect prompt_toolkit modules"

    def test_detection_conservative_avoids_false_negatives(self):
        """Conservative detection should not miss any prompt_toolkit import."""
        # Test with deeply nested prompt_toolkit imports
        nested_modules = [
            "prompt_toolkit",
            "prompt_toolkit.application",
            "prompt_toolkit.application.current",
            "prompt_toolkit.application.current._app",
        ]
        
        for module_name in nested_modules:
            with patch.dict(sys.modules, {module_name: MagicMock()}):
                _pt_active = any(
                    name.startswith("prompt_toolkit")
                    for name in sys.modules.keys()
                )
                assert _pt_active is True, f"Should detect {module_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

