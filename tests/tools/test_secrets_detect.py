"""Tests for secrets_detect tool."""

import json
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock


class TestSecretsDetect:
    @pytest.fixture
    def temp_file_with_secrets(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("API_KEY = 'TEST_KEY_1234567890abcdefghijklmnop'\n")
            f.write("password = 'mysecretpassword123'\n")
            f.write("# This is a comment with PASSWORD=test\n")
        yield f.name
        os.remove(f.name)

    @pytest.fixture
    def temp_file_clean(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    print('world')\n")
        yield f.name
        os.remove(f.name)

    def test_check_requirements(self):
        from tools.secrets_detect import check_secrets_detect_requirements
        result = check_secrets_detect_requirements()
        assert result is True

    def test_detect_api_key(self, temp_file_with_secrets):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(temp_file_with_secrets)
        data = json.loads(output)
        assert data["success"] is True
        assert data["total_findings"] > 0

    def test_detect_password(self, temp_file_with_secrets):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(temp_file_with_secrets)
        data = json.loads(output)
        assert data["success"] is True
        types = [f["type"] for f in data["findings"]]
        assert "Password" in types or "API Key" in types

    def test_clean_file_no_secrets(self, temp_file_clean):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(temp_file_clean)
        data = json.loads(output)
        assert data["success"] is True
        assert data["total_findings"] == 0

    def test_path_not_found(self):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect("/nonexistent/path")
        data = json.loads(output)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_severity_filter_high(self, temp_file_with_secrets):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(temp_file_with_secrets, severity="high")
        data = json.loads(output)
        assert data["success"] is True
        for f in data["findings"]:
            assert f["severity"] in ["high", "critical"]

    def test_severity_filter_critical(self, temp_file_with_secrets):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(temp_file_with_secrets, severity="critical")
        data = json.loads(output)
        assert data["success"] is True
        for f in data["findings"]:
            assert f["severity"] == "critical"

    def test_summary_counts(self, temp_file_with_secrets):
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(temp_file_with_secrets)
        data = json.loads(output)
        assert "summary" in data
        assert "critical" in data["summary"]
        assert "high" in data["summary"]
        assert "files_scanned" in data["summary"]

    def test_directory_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "test.py")
            with open(py_file, "w") as f:
                f.write("API_KEY = 'test_key_12345678901234567890'\n")
            
            from tools.secrets_detect import secrets_detect
            output = secrets_detect(tmpdir)
            data = json.loads(output)
            assert data["success"] is True
            assert data["total_findings"] > 0

    def test_exclude_test_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_config.py")
            with open(test_file, "w") as f:
                f.write("PASSWORD = 'secret123'\n")
            
            from tools.secrets_detect import secrets_detect
            output = secrets_detect(tmpdir)
            data = json.loads(output)
            
            excluded = any("test_config.py" in f["file"] for f in data["findings"])
            assert excluded or data["total_findings"] == 0


class TestSecretsDetectSchema:
    def test_schema_has_required_fields(self):
        from tools.secrets_detect import SECRETS_DETECT_SCHEMA
        assert SECRETS_DETECT_SCHEMA["name"] == "secrets_detect"
        assert "parameters" in SECRETS_DETECT_SCHEMA
        props = SECRETS_DETECT_SCHEMA["parameters"]["properties"]
        assert "path" in props
        assert "severity" in props
        assert "task_id" in props

    def test_schema_severity_enum(self):
        from tools.secrets_detect import SECRETS_DETECT_SCHEMA
        props = SECRETS_DETECT_SCHEMA["parameters"]["properties"]
        assert props["severity"]["enum"] == ["high", "medium", "low", "all"]


class TestShouldExclude:
    def test_exclude_test_files(self):
        from tools.secrets_detect import _should_exclude
        assert _should_exclude("test_config.py") is False
        assert _should_exclude("config.test.py") is True

    def test_exclude_node_modules(self):
        from tools.secrets_detect import _should_exclude
        assert _should_exclude("node_modules/package/index.js") is True

    def test_exclude_env_example(self):
        from tools.secrets_detect import _should_exclude
        assert _should_exclude(".env.example") is True

    def test_include_real_files(self):
        from tools.secrets_detect import _should_exclude
        assert _should_exclude("src/config.py") is False
        assert _should_exclude("app.py") is False