"""Shell ``eval`` hands its arguments to the shell, so a hardline pattern must
see the command *inside* the quoted string — the same way it already sees the
payload of ``sh -c``.

``_SHELL_CARRIER_NAMES`` contains ``eval``, so the segment IS walked, but
``_SHELL_NAMES`` (the set that gates payload extraction) does not, so
``_bash_exec_payload`` was never asked for it. The result: ``eval "rm -rf /"``
matched nothing while the identical ``sh -c "rm -rf /"`` hit the hardline
floor. See issue #102317.
"""

import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command


class TestShellEvalPayloadReachesHardlinePatterns:
    @pytest.mark.parametrize(
        "cmd, expected_detail",
        [
            # The shell -c path already works; eval must reach the same verdict.
            ('eval "rm -rf /"', "recursive delete of root filesystem"),
            ('eval "rm -rf --no-preserve-root /"', "recursive delete of root filesystem"),
            # Command position must not shield the carrier.
            ('sudo eval "rm -rf /"', "recursive delete of root filesystem"),
            # bash strips ONE leading `--` before concatenating (`eval -- echo hi`
            # runs `echo hi`; `eval -- -- echo hi` fails on the second `--`), so
            # leaving it in the payload would hide the command behind it.
            ('eval -- "rm -rf /"', "recursive delete of root filesystem"),
            ('eval -- rm -rf /', "recursive delete of root filesystem"),
        ],
    )
    def test_eval_payload_is_matched_like_shell_c(self, cmd, expected_detail):
        dangerous, detail = detect_hardline_command(cmd)
        assert dangerous is True, f"hardline floor missed {cmd!r}"
        if expected_detail is not None:
            assert detail == expected_detail, cmd

    def test_only_one_leading_double_dash_is_stripped(self):
        """`eval -- -- rm -rf /` leaves the second `--` as a command word.

        bash rejects it (`--: command not found`), so the payload must keep it
        rather than stripping every separator.
        """
        dangerous, _ = detect_hardline_command('eval -- -- "rm -rf /"')
        assert dangerous is False

    @pytest.mark.parametrize(
        "cmd",
        [
            # Everyday eval stays unflagged — the fix must not blanket-match.
            'eval "ls -la"',
            'eval "git status"',
            'eval 1+1',
            # Bare `eval` opens an interactive REPL: nothing is executed.
            "eval",
            # `eval` as data, not as a command position.
            "echo 'use eval for that'",
            "echo \"eval\"",
        ],
    )
    def test_benign_eval_stays_unflagged(self, cmd):
        assert detect_hardline_command(cmd) == (False, None), cmd

    @pytest.mark.parametrize(
        "payload",
        [
            "rm -rf /",
            "rm -rf --no-preserve-root /",
            "dd if=/dev/zero of=/dev/sda",
        ],
    )
    def test_eval_matches_its_shell_c_equivalent(self, payload):
        """The invariant: eval must classify identically to the same payload via -c.

        Whatever the hardline floor decides about a payload, it must decide the
        same thing when the payload arrives through `eval` instead of `sh -c` —
        that equivalence is the whole bug.
        """
        via_c, detail_c = detect_hardline_command(f'sh -c "{payload}"')
        via_eval, detail_eval = detect_hardline_command(f'eval "{payload}"')
        assert via_eval is via_c, payload
        assert detail_eval == detail_c, payload
        assert via_eval is True, f"{payload!r} should be a hardline payload"

    @pytest.mark.parametrize(
        "cmd",
        [
            'eval "ls -la"',
            'eval "make -j4"',
            'eval "$(git rev-parse HEAD)"',
            "eval 1+1",
        ],
    )
    def test_eval_prompts_for_approval_exactly_like_shell_c(self, cmd):
        """`eval` also reaches the approval layer, and that is deliberate.

        `_execution_flag_findings` feeds both consumers: the un-bypassable hardline
        floor and the approval prompt. `sh -c "ls -la"` already prompts, so
        routing the same payload through `eval` to the same prompt is the parity
        this fix is about — NOT a new block on ordinary work. Pin it so a later
        change cannot silently narrow one path and not the other.
        """
        payload = cmd[len("eval "):]
        via_c, via_eval = detect_dangerous_command(f'sh -c "{payload}"'), detect_dangerous_command(cmd)
        # The DESCRIPTION differs by carrier ("via eval" vs "via -c/-lc flag") and
        # should; the VERDICT must not.
        assert via_c[0] is via_eval[0] is True, f"{cmd!r} must prompt like its sh -c equivalent"

    def test_bare_eval_is_not_an_execution_mechanism(self):
        """No arguments means an interactive REPL, not a payload."""
        assert detect_dangerous_command("eval") == (False, None, None)
