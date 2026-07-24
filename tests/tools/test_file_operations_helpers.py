#!/usr/bin/env python3
"""
Tests for ShellFileOperations helper methods.

Focuses on testing the _ends_with_newline() method and wc -l line count
correction logic to ensure accurate line counting for all file types.
"""

import os
import pytest
from pathlib import Path
from tools.file_operations import ShellFileOperations


class MockTerminal:
    """Mock terminal environment for testing."""
    def __init__(self, cwd: str = "/tmp"):
        self.cwd = cwd
        self.executed_commands = []
    
    def execute(self, command: str, cwd: str = None, **kwargs):
        """Execute a shell command and return result."""
        self.executed_commands.append({
            'command': command,
            'cwd': cwd,
            'stdin': kwargs.get('stdin_data', '')
        })
        
        # Actually execute the command to get real results
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.cwd,
                input=kwargs.get('stdin_data', None),
                capture_output=True,
                timeout=30
            )
            return {
                'output': result.stdout.decode('utf-8', errors='replace'),
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'output': '', 'returncode': 1}
        except Exception as e:
            return {'output': str(e), 'returncode': 1}


@pytest.fixture
def temp_files(tmp_path):
    """Create test files with various newline configurations."""
    
    # File with trailing newline
    file_with_newline = tmp_path / "with_newline.txt"
    file_with_newline.write_text("line1\nline2\nline3\n")
    
    # File without trailing newline
    file_without_newline = tmp_path / "without_newline.txt"
    file_without_newline.write_text("line1\nline2\nline3")
    
    # Single line with newline
    single_line_with_newline = tmp_path / "single_with_newline.txt"
    single_line_with_newline.write_text("single line\n")
    
    # Single line without newline
    single_line_without_newline = tmp_path / "single_without_newline.txt"
    single_line_without_newline.write_text("single line")
    
    # Empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    
    # Large file with trailing newline (500+ lines)
    large_file_with_newline = tmp_path / "large_with_newline.txt"
    lines_with_newline = [f"line{i}\n" for i in range(1, 501)]
    large_file_with_newline.write_text("".join(lines_with_newline))
    
    # Large file without trailing newline (500+ lines)
    large_file_without_newline = tmp_path / "large_without_newline.txt"
    lines_without_newline = [f"line{i}\n" for i in range(1, 500)]
    lines_without_newline.append("line500")  # Last line has no newline
    large_file_without_newline.write_text("".join(lines_without_newline))
    
    return {
        'with_newline': str(file_with_newline),
        'without_newline': str(file_without_newline),
        'single_with_newline': str(single_line_with_newline),
        'single_without_newline': str(single_line_without_newline),
        'empty': str(empty_file),
        'large_with_newline': str(large_file_with_newline),
        'large_without_newline': str(large_file_without_newline),
    }


class TestEndsWithNewline:
    """Test the _ends_with_newline() helper method."""
    
    def test_file_with_trailing_newline(self, temp_files, tmp_path):
        """Files ending with newline should return True."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        assert file_ops._ends_with_newline(temp_files['with_newline']) is True
    
    def test_file_without_trailing_newline(self, temp_files, tmp_path):
        """Files not ending with newline should return False."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        assert file_ops._ends_with_newline(temp_files['without_newline']) is False
    
    def test_single_line_with_newline(self, temp_files, tmp_path):
        """Single line file with newline should return True."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        assert file_ops._ends_with_newline(temp_files['single_with_newline']) is True
    
    def test_single_line_without_newline(self, temp_files, tmp_path):
        """Single line file without newline should return False."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        assert file_ops._ends_with_newline(temp_files['single_without_newline']) is False
    
    def test_empty_file(self, temp_files, tmp_path):
        """Empty file should return True (considered to end with newline)."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        assert file_ops._ends_with_newline(temp_files['empty']) is True


class TestLineCountAccuracy:
    """Test that read_file() returns accurate line counts."""
    
    def test_file_with_trailing_newline(self, temp_files, tmp_path):
        """File with trailing newline: wc -l count should be accurate."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['with_newline'])
        
        # "line1\nline2\nline3\n" has 3 lines, wc -l reports 3
        assert result.total_lines == 3, f"Expected 3 lines, got {result.total_lines}"
    
    def test_file_without_trailing_newline(self, temp_files, tmp_path):
        """File without trailing newline: should add 1 to wc -l count."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['without_newline'])
        
        # "line1\nline2\nline3" has 3 lines, wc -l reports 2, we should report 3
        assert result.total_lines == 3, f"Expected 3 lines, got {result.total_lines}"
    
    def test_single_line_with_newline(self, temp_files, tmp_path):
        """Single line file with newline should report 1 line."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['single_with_newline'])
        
        # "single line\n" has 1 line, wc -l reports 1
        assert result.total_lines == 1, f"Expected 1 line, got {result.total_lines}"
    
    def test_single_line_without_newline(self, temp_files, tmp_path):
        """Single line file without newline should report 1 line."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['single_without_newline'])
        
        # "single line" has 1 line, wc -l reports 0, we should report 1
        assert result.total_lines == 1, f"Expected 1 line, got {result.total_lines}"
    
    def test_empty_file(self, temp_files, tmp_path):
        """Empty file should report 0 lines."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['empty'])
        
        # Empty file has 0 lines
        assert result.total_lines == 0, f"Expected 0 lines, got {result.total_lines}"
    
    def test_large_file_with_trailing_newline(self, temp_files, tmp_path):
        """Large file (500+ lines) with trailing newline should be accurate."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['large_with_newline'])
        
        # 500 lines with trailing newline, wc -l reports 500
        assert result.total_lines == 500, f"Expected 500 lines, got {result.total_lines}"
    
    def test_large_file_without_trailing_newline(self, temp_files, tmp_path):
        """Large file (500+ lines) without trailing newline should add 1."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(temp_files['large_without_newline'])
        
        # 500 lines without trailing newline, wc -l reports 499, we should report 500
        assert result.total_lines == 500, f"Expected 500 lines, got {result.total_lines}"
    
    def test_pagination_preserves_total_count(self, temp_files, tmp_path):
        """Pagination should not affect total line count."""
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        
        # Read first 10 lines
        result1 = file_ops.read_file(temp_files['large_with_newline'], offset=1, limit=10)
        # Read next 10 lines
        result2 = file_ops.read_file(temp_files['large_with_newline'], offset=11, limit=10)
        
        # Both should report same total
        assert result1.total_lines == result2.total_lines == 500


class TestEdgeCases:
    """Test additional edge cases."""
    
    def test_only_newlines(self, tmp_path):
        """File containing only newlines."""
        file_path = tmp_path / "only_newlines.txt"
        file_path.write_text("\n\n\n")
        
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(str(file_path))
        
        # 3 newlines = 3 empty lines + 1 final empty line after last newline = 4 lines
        # Actually wc -l counts newlines, so 3 newlines = 3 lines
        assert result.total_lines == 3, f"Expected 3 lines, got {result.total_lines}"
    
    def test_mixed_line_endings(self, tmp_path):
        """File with mixed content."""
        file_path = tmp_path / "mixed.txt"
        file_path.write_text("line1\nline2\r\nline3")  # Unix, Windows, no newline
        
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(str(file_path))
        
        # wc -l counts \n characters, so "line1\nline2\r\nline3" has 2 newlines
        # But we should count 3 lines (the last line without newline counts as 1)
        assert result.total_lines == 3, f"Expected 3 lines, got {result.total_lines}"
    
    def test_unicode_content(self, tmp_path):
        """File with unicode content and no trailing newline."""
        file_path = tmp_path / "unicode.txt"
        file_path.write_text("日本語\n中文\n한국어")  # No trailing newline
        
        file_ops = ShellFileOperations(MockTerminal(cwd=str(tmp_path)))
        result = file_ops.read_file(str(file_path))
        
        # 3 lines, wc -l reports 2 (2 newlines), we should report 3
        assert result.total_lines == 3, f"Expected 3 lines, got {result.total_lines}"