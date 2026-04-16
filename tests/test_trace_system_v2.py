"""
Test file for Trace System V2.

Tests the enhanced trace system with three-level indexing,
intelligent compression, and session boundary detection.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase, mock

from core.trace_system_v2 import (
    TraceSystemV2,
    TraceEvent,
    TraceSession,
    EventType,
    EventPriority,
    IntelligentCompressor,
    SessionBoundaryDetector,
    AsyncStorageManager
)


class TestTraceSystemV2(TestCase):
    """Test cases for Trace System V2."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "test_trace.db"
        
        # Mock the database path
        self.patcher = mock.patch('core.trace_system_v2.TRACE_DB_PATH', self.test_db_path)
        self.patcher.start()
        
        # Create trace system
        self.system = TraceSystemV2()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()
        shutil.rmtree(self.test_dir)
    
    def test_singleton_pattern(self):
        """Test that TraceSystemV2 follows singleton pattern."""
        system1 = TraceSystemV2()
        system2 = TraceSystemV2()
        self.assertIs(system1, system2)
    
    def test_create_session(self):
        """Test session creation."""
        session_id = self.system.create_session(metadata={'test': True})
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.system._sessions)
        
        session = self.system.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.status, "active")
    
    def test_end_session(self):
        """Test session ending."""
        session_id = self.system.create_session()
        self.system.end_session(session_id, status="completed")
        
        session = self.system.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.status, "completed")
        self.assertIsNotNone(session.ended_at)
    
    def test_record_tool_start(self):
        """Test recording tool start event."""
        session_id = self.system.create_session()
        
        self.system.record_tool_start(
            session_id=session_id,
            trace_id='trace-123',
            tool_call_id='tool-001',
            tool_name='read_file',
            tool_args={'path': '/test/file.txt'}
        )
        
        session = self.system.get_session(session_id)
        self.assertEqual(session.event_count, 2)  # session_start + tool_start
        self.assertIn('trace-123', session.trace_ids)
        self.assertEqual(session.tool_calls['read_file'], 1)
    
    def test_record_tool_complete(self):
        """Test recording tool complete event."""
        session_id = self.system.create_session()
        
        self.system.record_tool_complete(
            session_id=session_id,
            trace_id='trace-123',
            tool_call_id='tool-001',
            tool_name='read_file',
            tool_args={'path': '/test/file.txt'},
            tool_result='File contents',
            duration_ms=123.45
        )
        
        session = self.system.get_session(session_id)
        self.assertEqual(session.event_count, 2)  # session_start + tool_complete
    
    def test_record_tool_error(self):
        """Test recording tool error event."""
        session_id = self.system.create_session()
        
        self.system.record_tool_complete(
            session_id=session_id,
            trace_id='trace-123',
            tool_call_id='tool-001',
            tool_name='read_file',
            tool_args={'path': '/test/file.txt'},
            tool_result='',
            error='File not found'
        )
        
        session = self.system.get_session(session_id)
        self.assertEqual(len(session.errors), 1)
        self.assertEqual(session.errors[0]['error'], 'File not found')
    
    def test_record_llm_events(self):
        """Test recording LLM request and response."""
        session_id = self.system.create_session()
        
        self.system.record_llm_request(
            session_id=session_id,
            trace_id='trace-123',
            model='gpt-4',
            message_count=10
        )
        
        self.system.record_llm_response(
            session_id=session_id,
            trace_id='trace-123',
            model='gpt-4',
            response_preview='Test response preview',
            duration_ms=1234.5
        )
        
        session = self.system.get_session(session_id)
        self.assertEqual(session.event_count, 3)  # session_start + request + response
    
    def test_analyze_session(self):
        """Test session analysis."""
        session_id = self.system.create_session()
        
        # Add some events
        self.system.record_tool_start(
            session_id=session_id,
            trace_id='trace-123',
            tool_call_id='tool-001',
            tool_name='read_file',
            tool_args={'path': '/test/file.txt'}
        )
        
        self.system.record_tool_complete(
            session_id=session_id,
            trace_id='trace-123',
            tool_call_id='tool-001',
            tool_name='read_file',
            tool_args={'path': '/test/file.txt'},
            tool_result='File contents',
            duration_ms=123.45
        )
        
        analysis = self.system.analyze_session(session_id)
        
        self.assertEqual(analysis['session_id'], session_id)
        self.assertIn('tool_analysis', analysis)
        self.assertIn('read_file', analysis['tool_analysis'])
    
    def test_list_sessions(self):
        """Test listing sessions."""
        # Create multiple sessions with different timestamps
        import time
        
        session1 = self.system.create_session(metadata={'index': 1})
        time.sleep(0.001)  # Small delay to ensure different timestamps
        session2 = self.system.create_session(metadata={'index': 2})
        time.sleep(0.001)
        session3 = self.system.create_session(metadata={'index': 3})
        
        sessions = self.system.list_sessions(limit=2)
        self.assertEqual(len(sessions), 2)
        
        # Should be sorted by start time (most recent first)
        # session3 was created last, so should be first
        self.assertEqual(sessions[0].session_id, session3)
        self.assertEqual(sessions[1].session_id, session2)
    
    def test_three_level_indexing(self):
        """Test three-level indexing: session_id -> trace_id -> tool_call_id."""
        session_id = self.system.create_session()
        trace_id = 'trace-123'
        tool_call_id = 'tool-001'
        
        # Record events with different tool_call_ids
        self.system.record_tool_start(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name='read_file',
            tool_args={'path': '/test/file1.txt'}
        )
        
        self.system.record_tool_start(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id='tool-002',
            tool_name='write_file',
            tool_args={'path': '/test/file2.txt'}
        )
        
        # Check session in-memory data
        session = self.system.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.event_count, 3)  # session_start + 2 tool_starts
        self.assertIn(trace_id, session.trace_ids)
        self.assertEqual(session.tool_calls['read_file'], 1)
        self.assertEqual(session.tool_calls['write_file'], 1)


class TestIntelligentCompressor(TestCase):
    """Test cases for IntelligentCompressor."""
    
    def test_compress_tool_result_preserves_important(self):
        """Test that important content is preserved."""
        result = """Error: Something went wrong!
Warning: This is a warning.
Debug: This is debug info.
Regular output line 1
Regular output line 2
Success: Operation completed."""
        
        compressed = IntelligentCompressor.compress_tool_result(result, max_length=150)
        
        # Important patterns should be preserved
        self.assertIn("Error:", compressed)
        self.assertIn("Warning:", compressed)
        self.assertIn("Success:", compressed)
    
    def test_compress_tool_args(self):
        """Test tool args compression."""
        args = {'key': 'value', 'nested': {'data': 'test'}}
        compressed = IntelligentCompressor.compress_tool_args(args)
        
        self.assertIsInstance(compressed, str)
        self.assertIn('key', compressed)
        self.assertIn('value', compressed)
    
    def test_compress_long_result(self):
        """Test compression of long results."""
        long_result = "Line " + "x" * 1000 + "\n" * 100
        compressed = IntelligentCompressor.compress_tool_result(long_result, max_length=200)
        
        self.assertLessEqual(len(compressed), 200)
        self.assertIn("[... truncated ...]", compressed)


class TestSessionBoundaryDetector(TestCase):
    """Test cases for SessionBoundaryDetector."""
    
    def test_compression_boundary(self):
        """Test compression boundary detection."""
        detector = SessionBoundaryDetector()
        
        # Compression start should trigger boundary
        is_boundary = detector.detect_boundary(EventType.COMPRESSION_START)
        self.assertTrue(is_boundary)
    
    def test_message_count_boundary(self):
        """Test message count boundary detection."""
        detector = SessionBoundaryDetector()
        
        # Update message count
        detector.update_message_count(100)
        
        # Should trigger boundary after 50 messages
        is_boundary = detector.detect_boundary(EventType.CUSTOM)
        self.assertTrue(is_boundary)
    
    def test_no_boundary(self):
        """Test when no boundary should be detected."""
        detector = SessionBoundaryDetector()
        
        # Normal event should not trigger boundary
        is_boundary = detector.detect_boundary(EventType.TOOL_START)
        self.assertFalse(is_boundary)


class TestTraceEvent(TestCase):
    """Test cases for TraceEvent."""
    
    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = TraceEvent(
            session_id='session-123',
            trace_id='trace-456',
            event_type=EventType.TOOL_START,
            tool_name='test_tool'
        )
        
        event_dict = event.to_dict()
        
        self.assertEqual(event_dict['session_id'], 'session-123')
        self.assertEqual(event_dict['trace_id'], 'trace-456')
        self.assertEqual(event_dict['event_type'], EventType.TOOL_START)
        self.assertEqual(event_dict['tool_name'], 'test_tool')
        self.assertIn('timestamp', event_dict)
    
    def test_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            'session_id': 'session-123',
            'trace_id': 'trace-456',
            'event_type': EventType.TOOL_START,
            'tool_name': 'test_tool',
            'timestamp': '2024-01-01T00:00:00'
        }
        
        event = TraceEvent.from_dict(data)
        
        self.assertEqual(event.session_id, 'session-123')
        self.assertEqual(event.trace_id, 'trace-456')
        self.assertEqual(event.event_type, EventType.TOOL_START)
        self.assertEqual(event.tool_name, 'test_tool')
    
    def test_short_ids(self):
        """Test short ID generation."""
        event = TraceEvent(
            session_id='session-1234567890',
            trace_id='trace-1234567890',
            event_type=EventType.TOOL_START
        )
        
        self.assertEqual(event.trace_id_short, 'trace-12')
        self.assertIsNone(event.tool_call_id_short)
        
        event.tool_call_id = 'tool-1234567890'
        self.assertEqual(event.tool_call_id_short, 'tool-123')


if __name__ == '__main__':
    import unittest
    unittest.main()