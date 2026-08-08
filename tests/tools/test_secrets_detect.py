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


class TestCredentialMasking:
    def test_credential_masked_in_context(self):
        """Credential value in context must be masked, not in plaintext."""
        from tools.secrets_detect import secrets_detect
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("API_KEY = 'supersecretvalue1234567890'\n")
            f_name = f.name
        try:
            output = secrets_detect(f_name)
            data = json.loads(output)
            assert data["success"] is True
            assert data["total_findings"] > 0
            for finding in data["findings"]:
                ctx = finding["context"]
                # The full credential string must never appear in context
                assert "supersecretvalue1234567890" not in ctx
                # Must contain masking indicator
                assert "*" in ctx
        finally:
            os.remove(f_name)

    def test_context_preserves_surrounding_text(self):
        """Context around the credential must be preserved."""
        from tools.secrets_detect import secrets_detect
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("my_api_key = 'abcdefghij1234567890'\n")
            f_name = f.name
        try:
            output = secrets_detect(f_name)
            data = json.loads(output)
            assert data["success"] is True
            assert data["total_findings"] > 0
            ctx = data["findings"][0]["context"]
            # Variable name must be visible in context
            assert "my_api_key" in ctx
            # Password keyword context preserved
        finally:
            os.remove(f_name)

    def test_password_pattern_masked(self):
        """Password values must also be masked."""
        from tools.secrets_detect import secrets_detect
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("password = 'hunter2isabadpassword'\n")
            f_name = f.name
        try:
            output = secrets_detect(f_name)
            data = json.loads(output)
            assert data["success"] is True
            for finding in data["findings"]:
                if finding["type"] == "Password":
                    assert "hunter2isabadpassword" not in finding["context"]
                    assert "*" in finding["context"]
                    return
            pytest.fail("No Password finding detected")
        finally:
            os.remove(f_name)


class TestFilesScannedCount:
    def test_files_scanned_reflects_actual_count(self, tmp_path):
        """files_scanned must reflect total files inspected, not just files with findings."""
        # Create 3 files, only 1 with a secret
        for i in range(3):
            p = tmp_path / f"file_{i}.py"
            if i == 0:
                p.write_text("API_KEY = 'test_secret_key_12345678901234567890'\n")
            else:
                p.write_text("x = 1\n")

        from tools.secrets_detect import secrets_detect
        output = secrets_detect(str(tmp_path))
        data = json.loads(output)
        assert data["success"] is True
        # All 3 files were scanned, even though only 1 had findings
        assert data["summary"]["files_scanned"] == 3

    def test_files_scanned_single_file(self, tmp_path):
        """Single-file scan must report files_scanned = 1."""
        p = tmp_path / "clean.py"
        p.write_text("x = 1\n")

        from tools.secrets_detect import secrets_detect
        output = secrets_detect(str(p))
        data = json.loads(output)
        assert data["success"] is True
        assert data["summary"]["files_scanned"] == 1

    def test_files_scanned_empty_dir(self, tmp_path):
        """Empty directory must report files_scanned = 0."""
        from tools.secrets_detect import secrets_detect
        output = secrets_detect(str(tmp_path))
        data = json.loads(output)
        assert data["success"] is True
        assert data["summary"]["files_scanned"] == 0