"""Tests for cost_optimizer tool."""

import json
import pytest


class TestCostOptimizer:
    def test_check_requirements(self):
        from tools.cost_optimizer import check_cost_optimizer_requirements
        assert check_cost_optimizer_requirements() is True

    def test_compare_all_models(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer()
        data = json.loads(output)
        assert data["success"] is True
        assert "comparison" in data
        assert len(data["comparison"]) >= 4
        assert "summary" in data

    def test_single_model_analysis(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(model="claude-sonnet-4", input_tokens=5000, output_tokens=2000)
        data = json.loads(output)
        assert data["success"] is True
        assert data["analysis"]["model"] == "claude-sonnet-4"
        assert data["analysis"]["costs"]["total_cost"] > 0

    def test_single_model_with_alternatives(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(model="claude-sonnet-4")
        data = json.loads(output)
        assert data["success"] is True
        assert "alternatives" in data
        assert len(data["alternatives"]) >= 1

    def test_with_provider_filter(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(provider="google")
        data = json.loads(output)
        assert data["success"] is True
        assert "provider_filter" in data
        assert data["provider_filter"]["provider"] == "google"

    def test_model_with_provider(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(model="gpt-4o-mini", provider="openrouter")
        data = json.loads(output)
        assert data["success"] is True
        assert "via_provider" in data["analysis"]["costs"]

    def test_unknown_model(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(model="fake-model-v1")
        data = json.loads(output)
        assert data["success"] is False

    def test_different_token_counts(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(model="gemini-flash", input_tokens=100000, output_tokens=50000)
        data = json.loads(output)
        assert data["success"] is True
        cost = data["analysis"]["costs"]["total_cost"]
        assert cost > 0.01

    def test_cheapest_identified(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer()
        data = json.loads(output)
        assert data["summary"]["cheapest"]["cost"] <= data["summary"]["most_expensive"]["cost"]
        assert data["summary"]["cost_ratio"] >= 1

    def test_recommendation_present(self):
        from tools.cost_optimizer import cost_optimizer
        output = cost_optimizer(model="claude-sonnet-4")
        data = json.loads(output)
        assert data["success"] is True
        assert "recommendation" in data


class TestCostOptimizerSchema:
    def test_schema_has_required_fields(self):
        from tools.cost_optimizer import COST_OPTIMIZER_SCHEMA
        assert COST_OPTIMIZER_SCHEMA["name"] == "cost_optimizer"
        props = COST_OPTIMIZER_SCHEMA["parameters"]["properties"]
        assert "model" in props
        assert "input_tokens" in props
        assert "output_tokens" in props
        assert "task_id" in props