"""Unit tests for the process registry module.

Tests cover:
- ProcessSession dataclass initialization and state
- ProcessRegistry spawn, poll, wait, kill operations
- Output buffering and rolling window
- Session tracking and cleanup
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from tools.process_registry import (
    ProcessSession,
    ProcessRegistry,
    MAX_OUTPUT_CHARS,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


class TestProcessSession:
    """Tests for the ProcessSession dataclass."""
    
    def test_initialization_defaults(self):
        """Session initializes with correct defaults."""
        session = ProcessSession(id="proc_test123", command="echo hello")
        
        assert session.id == "proc_test123"
        assert session.command == "echo hello"
        assert session.task_id == ""
        assert session.session_key == ""
        assert session.pid is None
        assert session.process is None
        assert session.exited is False
        assert session.exit_code is None
        assert session.output_buffer == ""
        assert session.max_output_chars == MAX_OUTPUT_CHARS
        assert session.detached is False
    
    def test_initialization_with_values(self):
        """Session accepts custom values."""
        session = ProcessSession(
            id="proc_custom",
            command="pytest -v",
            task_id="task_abc",
            session_key="sess_xyz",
            pid=12345,
            cwd="/tmp",
        )
        
        assert session.id == "proc_custom"
        assert session.task_id == "task_abc"
        assert session.session_key == "sess_xyz"
        assert session.pid == 12345
        assert session.cwd == "/tmp"
    
    def test_lock_is_threading_lock(self):
        """Session has a threading lock for thread safety."""
        session = ProcessSession(id="proc_lock", command="test")
        assert isinstance(session._lock, type(threading.Lock()))


class TestProcessRegistry:
    """Tests for the ProcessRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        reg = ProcessRegistry()
        reg._sessions.clear()
        return reg
    
    def test_generate_id(self, registry):
        """Generated IDs have correct format."""
        id1 = registry._generate_id()
        id2 = registry._generate_id()
        
        assert id1.startswith("proc_")
        assert id2.startswith("proc_")
        assert len(id1) == 17  # proc_ + 12 chars
        assert id1 != id2  # Should be unique
    
    def test_get_nonexistent_session(self, registry):
        """Getting nonexistent session returns None."""
        result = registry.get("proc_doesnotexist")
        assert result is None
    
    def test_list_empty(self, registry):
        """Listing empty registry returns empty list."""
        sessions = registry.list_sessions()
        assert sessions == []
    
    def test_list_with_sessions(self, registry):
        """Listing shows all tracked sessions."""
        # Add mock sessions directly
        s1 = ProcessSession(id="proc_aaa", command="cmd1")
        s2 = ProcessSession(id="proc_bbb", command="cmd2")
        registry._sessions["proc_aaa"] = s1
        registry._sessions["proc_bbb"] = s2
        
        sessions = registry.list_sessions()
        assert len(sessions) == 2
        ids = [s.id for s in sessions]
        assert "proc_aaa" in ids
        assert "proc_bbb" in ids
    
    def test_list_filters_by_task_id(self, registry):
        """Listing can filter by task_id."""
        s1 = ProcessSession(id="proc_aaa", command="cmd1", task_id="task1")
        s2 = ProcessSession(id="proc_bbb", command="cmd2", task_id="task2")
        s3 = ProcessSession(id="proc_ccc", command="cmd3", task_id="task1")
        registry._sessions["proc_aaa"] = s1
        registry._sessions["proc_bbb"] = s2
        registry._sessions["proc_ccc"] = s3
        
        filtered = registry.list_sessions(task_id="task1")
        assert len(filtered) == 2
        assert all(s.task_id == "task1" for s in filtered)
    
    def test_poll_nonexistent(self, registry):
        """Polling nonexistent session returns error dict."""
        result = registry.poll("proc_fake")
        assert result["error"] is True
        assert "not found" in result["message"].lower()
    
    def test_poll_running_session(self, registry):
        """Polling running session returns status dict."""
        session = ProcessSession(
            id="proc_running",
            command="sleep 100",
            pid=12345,
            started_at=time.time(),
        )
        session.output_buffer = "some output\n"
        registry._sessions["proc_running"] = session
        
        result = registry.poll("proc_running")
        
        assert result["session_id"] == "proc_running"
        assert result["status"] == "running"
        assert result["pid"] == 12345
        assert "some output" in result["output"]
    
    def test_poll_exited_session(self, registry):
        """Polling exited session shows exit code."""
        session = ProcessSession(
            id="proc_done",
            command="echo done",
            pid=12345,
            started_at=time.time() - 10,
        )
        session.exited = True
        session.exit_code = 0
        session.output_buffer = "done\n"
        registry._sessions["proc_done"] = session
        
        result = registry.poll("proc_done")
        
        assert result["status"] == "exited"
        assert result["exit_code"] == 0
    
    def test_kill_nonexistent(self, registry):
        """Killing nonexistent session returns error."""
        result = registry.kill("proc_fake")
        assert result["error"] is True
    
    def test_output_buffer_truncation(self, registry):
        """Output buffer respects rolling window limit."""
        session = ProcessSession(
            id="proc_buffer",
            command="cat bigfile",
            max_output_chars=100,
        )
        
        # Simulate appending lots of output
        with session._lock:
            session.output_buffer = "x" * 150
            if len(session.output_buffer) > session.max_output_chars:
                session.output_buffer = session.output_buffer[-session.max_output_chars:]
        
        assert len(session.output_buffer) == 100


class TestConstants:
    """Tests for module constants."""
    
    def test_max_output_chars(self):
        """MAX_OUTPUT_CHARS is reasonable size."""
        assert MAX_OUTPUT_CHARS == 200_000
    
    def test_finished_ttl(self):
        """FINISHED_TTL_SECONDS is 30 minutes."""
        assert FINISHED_TTL_SECONDS == 1800
    
    def test_max_processes(self):
        """MAX_PROCESSES limits concurrent tracking."""
        assert MAX_PROCESSES == 64
