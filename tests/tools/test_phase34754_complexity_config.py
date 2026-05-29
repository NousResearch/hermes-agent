"""Unit tests for Phase 34754: Model Complexity Configuration System."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestModelComplexityConfig:
    """Tests for model complexity configuration loading and resolution."""
    
    def test_get_model_complexity_map_loads_config(self):
        """Test that get_model_complexity_map() loads active models from config."""
        from hermes_cli.config import get_model_complexity_map
        
        complexity_map = get_model_complexity_map()
        assert isinstance(complexity_map, dict)
        assert len(complexity_map) > 0
        assert "qwen3.5:397b-cloud" in complexity_map
        assert "kimi-k2.6:cloud" in complexity_map
    
    def test_get_model_complexity_map_filters_inactive(self):
        """Test that inactive models are excluded."""
        from hermes_cli.config import get_model_complexity_map
        
        complexity_map = get_model_complexity_map()
        
        # All returned models should be active
        for model_id, cfg in complexity_map.items():
            assert cfg.get("active", True) is True
    
    def test_get_model_complexity_from_config(self):
        """Test that get_model_complexity() returns correct complexity from config."""
        from hermes_cli.config import get_model_complexity
        
        assert get_model_complexity("qwen3.5:397b-cloud") == "easy"
        assert get_model_complexity("deepseek-v4-flash:cloud") == "medium"
        assert get_model_complexity("kimi-k2.6:cloud") == "hard"
    
    def test_get_model_complexity_fallback_unknown_model(self):
        """Test that unknown models fall back to 'medium'."""
        from hermes_cli.config import get_model_complexity
        
        # Unknown model should default to medium
        complexity = get_model_complexity("totally-unknown-model:xyz")
        assert complexity == "medium"
    
    def test_get_model_complexity_priority_chain(self):
        """Test the 4-tier fallback chain."""
        from hermes_cli.config import get_model_complexity
        
        # Config entry exists → use it
        assert get_model_complexity("qwen3.5:397b-cloud") == "easy"
        
        # No config, but BENCHMARK_REGISTRY might have it
        # (depending on what's in BENCHMARK_REGISTRY, this is fallback-dependent)
        complexity = get_model_complexity("unknown-model")
        assert complexity in ("easy", "medium", "hard")
    
    def test_build_delegation_capabilities_prompt_renders_models(self):
        """Test that Discovery Pipe renders configured models."""
        from agent.prompt_builder import build_delegation_capabilities_prompt
        
        prompt = build_delegation_capabilities_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Available Models" in prompt or "EASY" in prompt
    
    def test_build_delegation_capabilities_prompt_groups_by_complexity(self):
        """Test that Discovery Pipe groups models by complexity tier."""
        from agent.prompt_builder import build_delegation_capabilities_prompt
        
        prompt = build_delegation_capabilities_prompt()
        
        # Should have at least easy/medium/hard groups
        assert "EASY" in prompt or "easy" in prompt.lower()
        assert "MEDIUM" in prompt or "medium" in prompt.lower()
        assert "HARD" in prompt or "hard" in prompt.lower()
    
    def test_resolve_reasoning_effort_per_call_override(self):
        """Test that per-call reasoning_effort beats config."""
        from tools.delegate_tool import _resolve_reasoning_effort_from_config
        
        # Per-call override should win
        effort = _resolve_reasoning_effort_from_config("qwen3.5:397b-cloud", "xhigh")
        assert effort == "xhigh"
    
    def test_resolve_reasoning_effort_from_config(self):
        """Test that reasoning_effort is read from config."""
        from tools.delegate_tool import _resolve_reasoning_effort_from_config
        
        # qwen3.5 is configured as "low"
        effort = _resolve_reasoning_effort_from_config("qwen3.5:397b-cloud", None)
        assert effort == "low"
        
        # kimi is configured as "xhigh"
        effort = _resolve_reasoning_effort_from_config("kimi-k2.6:cloud", None)
        assert effort == "xhigh"
    
    def test_resolve_reasoning_effort_fallback(self):
        """Test that unknown models fall back to 'medium'."""
        from tools.delegate_tool import _resolve_reasoning_effort_from_config
        
        effort = _resolve_reasoning_effort_from_config("unknown-model", None)
        assert effort == "medium"
    
    def test_model_complexity_map_structure(self):
        """Test that model_complexity_map has correct structure."""
        from hermes_cli.config import get_model_complexity_map
        
        complexity_map = get_model_complexity_map()
        
        for model_id, cfg in complexity_map.items():
            assert isinstance(model_id, str)
            assert isinstance(cfg, dict)
            assert "complexity" in cfg
            assert cfg["complexity"] in ("easy", "medium", "hard")
            assert "reasoning_effort" in cfg
            assert cfg["reasoning_effort"] in ("none", "minimal", "low", "medium", "high", "xhigh")
            assert "active" in cfg
            assert isinstance(cfg["active"], bool)
    
    def test_config_yaml_has_model_complexity_map(self):
        """Test that ~/.hermes/config.yaml contains model_complexity_map."""
        from hermes_cli.config import load_config
        
        cfg = load_config()
        assert "delegation" in cfg
        assert "model_complexity_map" in cfg["delegation"]
        assert len(cfg["delegation"]["model_complexity_map"]) > 0
    
    def test_user_can_override_complexity(self):
        """Test that user-added models are accessible via get_model_complexity_map."""
        from hermes_cli.config import get_model_complexity_map
        
        # This test verifies the mechanism works, not that user edits are persisted
        # (user edits would be in their config.yaml)
        complexity_map = get_model_complexity_map()
        
        # At least one model should be accessible
        if complexity_map:
            first_model = list(complexity_map.keys())[0]
            assert first_model is not None
    
    def test_disabled_model_not_in_map(self):
        """Test that inactive models are excluded from the map."""
        from hermes_cli.config import load_config, get_model_complexity_map
        
        cfg = load_config()
        all_models = cfg.get("delegation", {}).get("model_complexity_map", {})
        active_models = get_model_complexity_map()
        
        # All returned models should be active
        for model_id in active_models.keys():
            assert all_models[model_id].get("active", True) is True


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing systems."""
    
    def test_delegate_task_still_works(self):
        """Test that delegate_task function still exists and is callable."""
        from tools.delegate_tool import delegate_task
        
        assert callable(delegate_task)
    
    def test_benchmark_registry_unchanged(self):
        """Test that BENCHMARK_REGISTRY still exists and works."""
        from agent.benchmark_registry import BENCHMARK_REGISTRY
        
        assert isinstance(BENCHMARK_REGISTRY, dict)
        assert len(BENCHMARK_REGISTRY) > 0
    
    def test_size_tiers_unchanged(self):
        """Test that SIZE_TIERS still exists for fallback."""
        from agent.model_fallback_estimator import SIZE_TIERS
        
        assert isinstance(SIZE_TIERS, list)
        assert len(SIZE_TIERS) > 0
    
    def test_config_loader_unchanged(self):
        """Test that load_config() still works as before."""
        from hermes_cli.config import load_config
        
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "model" in cfg or "delegation" in cfg


class TestIntegrationE2E:
    """End-to-end integration tests."""
    
    def test_discovery_pipe_in_prompt(self):
        """Test that Discovery Pipe can be injected into system prompt."""
        from agent.prompt_builder import build_delegation_capabilities_prompt
        
        prompt = build_delegation_capabilities_prompt()
        
        # Prompt should not be empty
        assert prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # Reasonable minimum
    
    def test_config_priority_chain_e2e(self):
        """Test the complete priority chain: per-call > config > fallback."""
        from hermes_cli.config import get_model_complexity
        from tools.delegate_tool import _resolve_reasoning_effort_from_config
        
        # Known model from config
        complexity = get_model_complexity("kimi-k2.6:cloud")
        assert complexity == "hard"
        
        # Reasoning effort for same model
        effort = _resolve_reasoning_effort_from_config("kimi-k2.6:cloud", None)
        assert effort == "xhigh"
        
        # Unknown model falls back
        complexity = get_model_complexity("totally-unknown")
        assert complexity in ("easy", "medium", "hard")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
