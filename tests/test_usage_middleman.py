from agent.usage_middleman import build_compact_usage_table


def test_build_compact_usage_table_is_fixed_width_and_includes_sections():
    lines = build_compact_usage_table(
        model="anthropic/claude-sonnet-4.6",
        provider="openrouter",
        input_tokens=113_992,
        output_tokens=24_160,
        cache_read_tokens=3_573_909,
        cache_write_tokens=334_317,
        total_tokens=4_046_378,
        cost_usd=3.0302,
        cost_status="estimated",
        duration_str="26m",
        context_tokens=105_471,
        context_length=1_000_000,
        api_calls=50,
        balance_rows=[
            ("openrouter", "Credits balance: $44.48"),
            ("maritaca", "Saldo: R$ 118,96"),
        ],
        quota_sections=[
            ("claude code", [("Current session", 55.0, "in 3h 10m")]),
            (
                "codex / openai",
                [
                    ("5h limit", 0.0, "in 5h 26m"),
                    ("weekly", 100.0, "in 1d 21h"),
                ],
            ),
        ],
    )

    assert lines[0] == "#" * 79
    assert lines[-1] == "#" * 79
    assert all(len(line) == 79 for line in lines)
    assert any("Usage  openrouter / anthropic/claude-sonnet-4.6" in line for line in lines)
    assert any("session" in line for line in lines)
    assert any("BALANCES" in line for line in lines)
    assert any("claude code" in line for line in lines)
    assert any("codex / openai" in line for line in lines)
    assert any("[███████" in line for line in lines)


def test_build_compact_usage_table_renders_balance_rows_as_centered_pyramid_blocks():
    lines = build_compact_usage_table(
        model="anthropic/claude-sonnet-4.6",
        provider="openrouter",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        total_tokens=15,
        cost_usd=None,
        cost_status="unknown",
        duration_str="5m",
        context_tokens=200,
        context_length=400,
        api_calls=1,
        balance_rows=[
            (
                "openrouter",
                "Credits balance: $27.08 • API key usage: $38.63 total • $18.37 today • $38.63 this week • $38.63 this month",
            ),
            ("maritaca", "Saldo: R$ 118,29"),
        ],
        quota_sections=[],
    )

    assert all(len(line) == 79 for line in lines)
    assert all(line.startswith("#") and line.endswith("#") for line in lines)

    section_divider = "#" + "-" * 77 + "#"
    balances_title = next(line for line in lines if "BALANCES" in line)
    balances_index = lines.index(balances_title)
    assert lines[balances_index - 1] == section_divider
    assert balances_title.strip("# ") == "BALANCES"
    assert lines[balances_index + 1] == section_divider

    openrouter_title = next(line for line in lines if line.strip("# ") == "openrouter")
    openrouter_index = lines.index(openrouter_title)
    top_metrics = next(line for line in lines if "| d $18.37 |" in line)
    bottom_metrics = next(line for line in lines if "| bal $27.08 |" in line)
    top_index = lines.index(top_metrics)
    bottom_index = lines.index(bottom_metrics)

    assert lines[openrouter_index - 1] == section_divider
    assert openrouter_index < top_index < bottom_index
    assert lines[top_index - 1].strip("# ") == "+----------+----------+----------+"
    assert top_metrics.strip("# ") == "| d $18.37 | w $38.63 | m $38.63 |"
    assert lines[top_index + 1].strip("# ") == "+----------+----------+----------+"
    assert lines[bottom_index - 1].strip("# ") == "+------------+----------+"
    assert bottom_metrics.strip("# ") == "| bal $27.08 | Σ $38.63 |"
    assert lines[bottom_index + 1].strip("# ") == "+------------+----------+"
    assert bottom_metrics.index("|") > top_metrics.index("|")

    maritaca_title = next(line for line in lines if line.strip("# ") == "maritaca")
    maritaca_index = lines.index(maritaca_title)
    maritaca_saldo = next(line for line in lines if "saldo R$ 118,29" in line)
    assert lines[maritaca_index - 1] == section_divider
    assert maritaca_index < lines.index(maritaca_saldo)
    assert maritaca_saldo.strip("# ") == "saldo R$ 118,29"
