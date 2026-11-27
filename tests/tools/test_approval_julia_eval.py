"""Julia evaluates inline code through ``-e/--eval`` and ``-E/--print`` (src/jloptions.c), but it was not a
known interpreter family, so ``julia -e 'using Pkg; Pkg.add(...)'`` passed every detector. It must reach the
same approval classification as ``node -e`` / ``python -c``.
"""

import pytest

from tools.approval import detect_dangerous_command


class TestJuliaInlineScriptExecution:
    @pytest.mark.parametrize(
        "cmd, expected_key",
        [
            ("julia -e 'using Pkg; Pkg.add(\"Example\")'", "script execution via -e/-c flag"),
            ("julia --eval 'println(1)'", "script execution via -e/-c flag"),
            ("julia --eval='println(1)'", "script execution via -e/-c flag"),
            ("julia -E '1 + 1'", "script execution via -e/-c flag"),
            ("julia --print '1 + 1'", "script execution via -e/-c flag"),
            ("julia -e'println(1)'", "script execution via -e/-c flag"),
            ("julia -qe 'println(1)'", "script execution via -e/-c flag"),
            # getopt_long expands unique long-option prefixes.
            ("julia --ev 'println(1)'", "script execution via -e/-c flag"),
            ("julia --pri '1 + 1'", "script execution via -e/-c flag"),
            # Options that take a separate value must not end the scan before the eval flag.
            ("julia -t 4 -e 'println(1)'", "script execution via -e/-c flag"),
            ("julia --threads 4 --eval 'println(1)'", "script execution via -e/-c flag"),
            ("julia --thr 4 -e 'println(1)'", "script execution via -e/-c flag"),
            ("julia -L helper.jl -e 'main()'", "script execution via -e/-c flag"),
            ("julia -O 2 -e 'println(1)'", "script execution via -e/-c flag"),
            ("julia -O3 -e 'println(1)'", "script execution via -e/-c flag"),
            # A short-option bundle ending in a value-taking option consumes the next token as that value.
            ("julia -qt 4 -e 'println(1)'", "script execution via -e/-c flag"),
            ("julia -qL helper.jl -e 'main()'", "script execution via -e/-c flag"),
            # `--project` takes its value only with `=`, so a bare `--project` leaves `-e` next.
            ("julia --project -e 'using Pkg; Pkg.instantiate()'", "script execution via -e/-c flag"),
            ("julia --proj=. -e 'using Pkg; Pkg.instantiate()'", "script execution via -e/-c flag"),
            # juliaup's launcher consumes a leading `+<channel>` before julia parses its options.
            ("julia +1.10 -e 'println(1)'", "script execution via -e/-c flag"),
            ("julia +release --eval 'println(1)'", "script execution via -e/-c flag"),
            ("sudo julia -e 'println(1)'", "script execution via -e/-c flag"),
            ("JULIA.EXE -e 'println(1)'", "script execution via -e/-c flag"),
            ("julia << 'EOF'\nprintln(\"pwned\")\nEOF", "script execution via heredoc"),
            ('julia -e "println(1)', "command parser limit or malformed executable payload"),
        ],
    )
    def test_inline_eval_classified_like_node_and_python(self, cmd, expected_key):
        dangerous, key, _ = detect_dangerous_command(cmd)
        assert dangerous is True, cmd
        assert key == expected_key, cmd

    @pytest.mark.parametrize(
        "cmd",
        [
            "julia script.jl",
            "julia --version",
            "julia --project=. scripts/run_inversion.jl --time-idx=130",
            "julia -t 4 -L helper.jl script.jl",
            "julia -O 3 script.jl",
            "julia +1.10 script.jl",
            # Arguments after the program file belong to the script's ARGS.
            "julia script.jl -e foo",
            # `-m` ends option parsing; the remaining arguments go to the package's entry point.
            "julia -m Example run",
            "echo 'use julia -e for that'",
        ],
    )
    def test_everyday_julia_invocations_stay_unflagged(self, cmd):
        assert detect_dangerous_command(cmd) == (False, None, None), cmd
