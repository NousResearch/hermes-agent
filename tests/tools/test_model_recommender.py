"""Tests for model_recommender tool."""

import json
import pytest


class TestModelRecommender:
    def test_check_requirements(self):
        from tools.model_recommender import check_model_recommend_requirements
        assert check_model_recommend_requirements() is True

    def test_recommend_code_gen(self):
        from tools.model_recommender import model_recommend
        output = model_recommend("Write a Python function to sort a list of dictionaries by a key")
        data = json.loads(output)
        assert data["success"] is True
        assert data["task_type"] in ("code-gen", "explain")
        assert len(data["recommendations"]) >= 1

    def test_recommend_debug(self):
        from tools.model_recommender import model_recommend
        output = model_recommend("Fix this bug: the login function crashes when password is empty")
        data = json.loads(output)
        assert data["success"] is True
        assert data["task_type"] == "debug"

    def test_recommend_simple_task_auto_downgrade(self):
        from tools.model_recommender import model_recommend
        output = model_recommend("hi")
        data = json.loads(output)
        assert data["success"] is True

    def test_recommend_with_provider_filter(self):
        from tools.model_recommender import model_recommend
        output = model_recommend("Write a complex algorithm", preferred_provider="google")
        data = json.loads(output)
        assert data["success"] is True

    def test_recommend_with_max_cost(self):
        from tools.model_recommender import model_recommend
        output = model_recommend("Write code", max_cost_per_mtok=1.0)
        data = json.loads(output)
        assert data["success"] is True
        assert all(r["avg_cost_per_mtok"] <= 1.0 for r in data["recommendations"])

    def test_recommend_prefer_speed(self):
        from tools.model_recommender import model_recommend
        output = model_recommend("Quick task", prefer_speed=True)
        data = json.loads(output)
        assert data["success"] is True

    def test_classify_task_code(self):
        from tools.model_recommender import _classify_task
        task_type, complexity = _classify_task("Write a function to parse JSON")
        assert task_type in ("code-gen",)
        assert 0 <= complexity <= 1

    def test_classify_task_debug(self):
        from tools.model_recommender import _classify_task
        task_type, complexity = _classify_task("Fix the crash when loading config file with missing keys")
        assert task_type == "debug"

    def test_classify_task_empty(self):
        from tools.model_recommender import _classify_task
        task_type, complexity = _classify_task("hello world")
        assert task_type == "explain"
        assert complexity <= 0.5


class TestModelRecommenderSchema:
    def test_schema_has_required_fields(self):
        from tools.model_recommender import MODEL_RECOMMEND_SCHEMA
        assert MODEL_RECOMMEND_SCHEMA["name"] == "model_recommend"
        props = MODEL_RECOMMEND_SCHEMA["parameters"]["properties"]
        assert "prompt" in props
        assert "preferred_provider" in props
        assert "task_id" in props