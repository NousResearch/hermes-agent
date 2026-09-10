"""Tests for distutils.text_file."""

from distutils.tests import support
from distutils.text_file import TextFile

import jaraco.path
import path

TEST_DATA = """# test file

line 3 \\
# intervening comment
  continues on next line
"""


class TestTextFile(support.TempdirManager):
    def test_class(self):
        # old tests moved from text_file.__main__
        # so they are really called by the buildbots

        # result 1: no fancy options
        result1 = [
            '# test file\n',
            '\n',
            'line 3 \\\n',
            '# intervening comment\n',
            '  continues on next line\n',
        ]

        # result 2: just strip comments
        result2 = ["\n", "line 3 \\\n", "  continues on next line\n"]

        # result 3: just strip blank lines
        result3 = [
            "# test file\n",
            "line 3 \\\n",
            "# intervening comment\n",
            "  continues on next line\n",
        ]

        # result 4: default, strip comments, blank lines,
        # and trailing whitespace
        result4 = ["line 3 \\", "  continues on next line"]

        # result 5: strip comments and blanks, plus join lines (but don't
        # "collapse" joined lines
        result5 = ["line 3   continues on next line"]

        # result 6: strip comments and blanks, plus join lines (and
        # "collapse" joined lines
        result6 = ["line 3 continues on next line"]

        def test_input(count, description, file, expected_result):
            result = file.readlines()
            assert result == expected_result

        tmp_path = path.Path(self.mkdtemp())
        filename = tmp_path / 'test.txt'
        jaraco.path.build({filename.name: TEST_DATA}, tmp_path)

        in_file = TextFile(
            filename,
            strip_comments=False,
            skip_blanks=False,
            lstrip_ws=False,
            rstrip_ws=False,
        )
        try:
            test_input(1, "no processing", in_file, result1)
        finally:
            in_file.close()

        in_file = TextFile(
            filename,
            strip_comments=True,
            skip_blanks=False,
            lstrip_ws=False,
            rstrip_ws=False,
        )
        try:
            test_input(2, "strip comments", in_file, result2)
        finally:
            in_file.close()

        in_file = TextFile(
            filename,
            strip_comments=False,
            skip_blanks=True,
            lstrip_ws=False,
            rstrip_ws=False,
        )
        try:
            test_input(3, "strip blanks", in_file, result3)
        finally:
            in_file.close()

        in_file = TextFile(filename)
        try:
            test_input(4, "default processing", in_file, result4)
        finally:
            in_file.close()

        in_file = TextFile(
            filename,
            strip_comments=True,
            skip_blanks=True,
            join_lines=True,
            rstrip_ws=True,
        )
        try:
            test_input(5, "join lines without collapsing", in_file, result5)
        finally:
            in_file.close()

        in_file = TextFile(
            filename,
            strip_comments=True,
            skip_blanks=True,
            join_lines=True,
            rstrip_ws=True,
            collapse_join=True,
        )
        try:
            test_input(6, "join lines with collapsing", in_file, result6)
        finally:
            in_file.close()
