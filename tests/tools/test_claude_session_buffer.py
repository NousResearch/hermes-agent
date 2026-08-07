"""Tests for tools/claude_session/output_buffer.py"""

import time
import pytest
from tools.claude_session.output_buffer import OutputBuffer, OutputLine


class TestOutputLine:
    def test_creation(self):
        ol = OutputLine(text="hello", timestamp=1.0)
        assert ol.text == "hello"
        assert ol.timestamp == 1.0

    def test_content_hash(self):
        ol1 = OutputLine(text="hello", timestamp=1.0)
        ol2 = OutputLine(text="hello", timestamp=2.0)
        assert ol1.content_hash() == ol2.content_hash()

    def test_content_hash_differs(self):
        ol1 = OutputLine(text="hello", timestamp=1.0)
        ol2 = OutputLine(text="world", timestamp=1.0)
        assert ol1.content_hash() != ol2.content_hash()


class TestOutputBuffer:
    def test_append_and_read(self):
        buf = OutputBuffer(max_lines=10)
        buf.append("line1")
        buf.append("line2")
        lines = buf.read(offset=0, limit=10)
        assert len(lines) == 2
        assert lines[0].text == "line1"
        assert lines[1].text == "line2"

    def test_ring_overflow(self):
        buf = OutputBuffer(max_lines=3)
        buf.append("line1")
        buf.append("line2")
        buf.append("line3")
        buf.append("line4")
        lines = buf.read(offset=0, limit=10)
        assert len(lines) == 3
        assert lines[0].text == "line2"
        assert lines[2].text == "line4"

    def test_offset_limit(self):
        buf = OutputBuffer(max_lines=100)
        for i in range(10):
            buf.append(f"line{i}")
        lines = buf.read(offset=5, limit=3)
        assert len(lines) == 3
        assert lines[0].text == "line5"

    def test_total_count(self):
        buf = OutputBuffer(max_lines=100)
        for i in range(10):
            buf.append(f"line{i}")
        assert buf.total_count() == 10

    def test_total_count_with_overflow(self):
        buf = OutputBuffer(max_lines=5)
        for i in range(10):
            buf.append(f"line{i}")
        assert buf.total_count() == 10

    def test_since_marker(self):
        buf = OutputBuffer(max_lines=100)
        buf.append("before1")
        buf.append("before2")
        marker = buf.append("marker_line")
        buf.append("after1")
        buf.append("after2")
        lines = buf.since(marker)
        assert len(lines) == 2
        assert lines[0].text == "after1"

    def test_since_marker_overflow(self):
        buf = OutputBuffer(max_lines=3)
        m = buf.append("m")
        buf.append("a")
        buf.append("b")
        buf.append("c")  # overflow, "m" evicted
        lines = buf.since(m)
        # marker was evicted, return all current content
        assert len(lines) == 3

    def test_last_n_chars(self):
        buf = OutputBuffer(max_lines=100)
        for i in range(20):
            buf.append(f"line{i}")
        tail = buf.last_n_chars(30)
        assert len(tail) <= 30

    def test_append_batch_dedup(self):
        buf = OutputBuffer(max_lines=100)
        buf.append("line1")
        buf.append("line2")
        # Simulate re-capturing same pane output
        added = buf.append_batch(["line1", "line2", "line3"])
        assert added == 1  # Only line3 is new

    def test_clear(self):
        buf = OutputBuffer(max_lines=100)
        buf.append("line1")
        buf.clear()
        # Counter is preserved to keep markers valid
        assert buf.total_count() == 1
        assert buf.read() == []

    def test_thread_safety(self):
        import threading
        buf = OutputBuffer(max_lines=100)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    buf.append(f"thread{start}-{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=(0,))
        t2 = threading.Thread(target=writer, args=(100,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
        assert buf.total_count() == 200
