"""Generic tests for hermes_t TradingStateStore and shared CLI builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ── TradingStateStore tests ──────────────────────────────────────────────


def test_store_default_base_dir_is_home_based():
    """TradingStateStore should default to ~/.hermes_t_runtime/<profile_id>."""
    with pytest.MonkeyPatch.context() as mp:
        fake_home = Path("/tmp/fake_home")
        mp.setattr(Path, "home", lambda: fake_home)
        from hermes_t.store import TradingStateStore

        store = TradingStateStore(profile_id="test_profile")
        assert store.base_dir == fake_home / ".hermes_t_runtime"
        assert store.state_dir == fake_home / ".hermes_t_runtime" / "test_profile"


def test_store_explicit_base_dir(tmp_path: Path):
    """Explicit base_dir should be used directly."""
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="p1")
    assert store.base_dir == tmp_path
    assert store.state_dir == tmp_path / "p1"


def test_store_rejects_profile_id_path_traversal(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    with pytest.raises(ValueError, match="profile_id"):
        TradingStateStore(base_dir=str(tmp_path), profile_id="../escaped")


def test_store_creates_state_dir_on_first_write(tmp_path: Path):
    """Saving state should create the state directory if it doesn't exist."""
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="created_on_write")
    assert not store.state_dir.exists()
    store.save_execution_state({"trade_date": "2026-05-03"})
    assert store.state_dir.exists()


def test_store_load_execution_state_returns_empty_dict_when_missing(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="missing")
    state = store.load_execution_state()
    assert state == {}


def test_store_save_and_load_pending_signal_roundtrip(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="pending_test")
    signal = {"signal_id": "sell_1_20260503", "status": "pending", "action": "sell"}
    store.save_pending_signal(signal)
    loaded = store.load_pending_signal()
    assert loaded == signal


def test_store_save_pending_signal_overwrites_previous(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="overwrite")
    store.save_pending_signal({"status": "pending"})
    store.save_pending_signal({"status": "sent"})
    loaded = store.load_pending_signal()
    assert loaded == {"status": "sent"}


def test_store_clear_pending_signal_writes_empty_dict(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="clear_test")
    store.save_pending_signal({"status": "pending"})
    store.clear_pending_signal()
    loaded = store.load_pending_signal()
    assert loaded == {}


def test_store_load_position_returns_none_when_missing(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="no_pos")
    assert store.load_position() is None


def test_store_save_and_load_position(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="pos_test")
    pos = {"symbol": "688319", "total_shares": 220000}
    store.save_position(pos)
    loaded = store.load_position()
    assert loaded == pos


def test_store_append_and_read_dispatch_ledger(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="ledger_test")
    store.append_dispatch_ledger({"trade_date": "2026-05-03", "profit": 1000.0})
    store.append_dispatch_ledger({"trade_date": "2026-05-04", "profit": 500.0})
    rows = store.read_dispatch_ledger()
    assert len(rows) == 2
    assert rows[0]["profit"] == 1000.0
    assert rows[1]["profit"] == 500.0


def test_store_read_ledger_returns_empty_list_when_not_exists(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="no_ledger")
    assert store.read_dispatch_ledger() == []


def test_store_profile_scoped_paths_are_isolated(tmp_path: Path):
    from hermes_t.store import TradingStateStore

    s1 = TradingStateStore(base_dir=str(tmp_path), profile_id="alice")
    s2 = TradingStateStore(base_dir=str(tmp_path), profile_id="bob")
    s1.save_execution_state({"sell_count": 1})
    s2.save_execution_state({"sell_count": 2})
    assert s1.load_execution_state()["sell_count"] == 1
    assert s2.load_execution_state()["sell_count"] == 2


def test_store_default_base_dir_not_affected_by_cwd(tmp_path: Path):
    """The default base_dir should not change when cwd changes."""
    from hermes_t.store import TradingStateStore

    fake_home = Path("/tmp/fake_home_stable")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "home", lambda: fake_home)
        store = TradingStateStore(profile_id="stable")
        assert store.base_dir == fake_home / ".hermes_t_runtime"


# ── Shared CLI builder tests ─────────────────────────────────────────────


def test_build_runtime_parser_defaults_to_home_based_base_dir():
    from hermes_t.cli_shared import build_runtime_parser

    fake_home = Path("/tmp/cli_home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "home", lambda: fake_home)
        parser = build_runtime_parser()
        args = parser.parse_args(["--symbol", "688319", "--trade-unit", "10000"])

    assert args.base_dir == fake_home / ".hermes_t_runtime"
    assert args.profile_id == "default"
    assert args.max_trades == 4
    assert args.signal == "hold"
    assert args.score == 50


def test_build_runtime_parser_prefers_hermes_t_env_over_default():
    from hermes_t.cli_shared import build_runtime_parser

    fake_dir = "/tmp/hermes_t_env_runtime"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HERMES_T_BASE_DIR", fake_dir)
        parser = build_runtime_parser()
        args = parser.parse_args(["--symbol", "688319", "--trade-unit", "10000"])

    assert args.base_dir == Path(fake_dir)


def test_build_runtime_parser_falls_back_to_hermes_olin_env():
    from hermes_t.cli_shared import build_runtime_parser

    fake_dir = "/tmp/hermes_olin_env_runtime"
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("HERMES_T_BASE_DIR", raising=False)
        mp.setenv("HERMES_OLIN_BASE_DIR", fake_dir)
        parser = build_runtime_parser()
        args = parser.parse_args(["--symbol", "688319", "--trade-unit", "10000"])

    assert args.base_dir == Path(fake_dir)


def test_build_runtime_profile_from_args_builds_valid_profile():
    from hermes_t.cli_shared import build_runtime_parser, build_runtime_profile_from_args

    parser = build_runtime_parser()
    args = parser.parse_args([
        "--profile-id",
        "olin",
        "--symbol",
        "688319",
        "--trade-unit",
        "10000",
        "--max-trades",
        "3",
    ])
    profile = build_runtime_profile_from_args(args)

    assert profile.profile_id == "olin"
    assert profile.symbol == "688319"
    assert profile.trade_unit == 10000
    assert profile.max_trades == 3


@pytest.mark.parametrize("field_name, argv", [
    ("profile_id", ["--profile-id", "   ", "--symbol", "688319", "--trade-unit", "10000"]),
    ("symbol", ["--symbol", "   ", "--trade-unit", "10000"]),
])
def test_build_runtime_profile_from_args_rejects_blank_required_strings(field_name: str, argv: list[str]):
    from hermes_t.cli_shared import build_runtime_parser, build_runtime_profile_from_args

    parser = build_runtime_parser()
    args = parser.parse_args(argv)

    with pytest.raises(ValueError, match=field_name):
        build_runtime_profile_from_args(args)


@pytest.mark.parametrize("argv, field_name", [
    (["--symbol", "688319", "--trade-unit", "0"], "trade_unit"),
    (["--symbol", "688319", "--trade-unit", "10000", "--max-trades", "0"], "max_trades"),
    (["--symbol", "688319", "--trade-unit", "-1"], "trade_unit"),
    (["--symbol", "688319", "--trade-unit", "10000", "--max-trades", "-2"], "max_trades"),
])
def test_build_runtime_profile_from_args_rejects_non_positive_ints(argv: list[str], field_name: str):
    from hermes_t.cli_shared import build_runtime_parser, build_runtime_profile_from_args

    parser = build_runtime_parser()
    args = parser.parse_args(argv)

    with pytest.raises(ValueError, match=field_name):
        build_runtime_profile_from_args(args)


def test_build_runtime_profile_from_args_rejects_bool_trade_unit():
    from argparse import Namespace

    from hermes_t.cli_shared import build_runtime_profile_from_args

    args = Namespace(
        profile_id="default",
        symbol="688319",
        trade_unit=True,
        max_trades=4,
        base_dir=Path("/tmp/x"),
        signal="hold",
        score=50,
        dispatch=False,
        trade_date=None,
    )

    with pytest.raises(ValueError, match="trade_unit"):
        build_runtime_profile_from_args(args)


def test_build_runtime_profile_from_args_rejects_bool_max_trades():
    from argparse import Namespace

    from hermes_t.cli_shared import build_runtime_profile_from_args

    args = Namespace(
        profile_id="default",
        symbol="688319",
        trade_unit=10000,
        max_trades=True,
        base_dir=Path("/tmp/x"),
        signal="hold",
        score=50,
        dispatch=False,
        trade_date=None,
    )

    with pytest.raises(ValueError, match="max_trades"):
        build_runtime_profile_from_args(args)


def test_build_runtime_store_returns_trading_state_store(tmp_path: Path):
    from hermes_t.cli_shared import RuntimeProfile, build_runtime_store
    from hermes_t.store import TradingStateStore

    profile = RuntimeProfile(profile_id="olin", symbol="688319", trade_unit=10000, max_trades=4)
    store = build_runtime_store(tmp_path, profile, prefer_legacy_olin_store=False)

    assert isinstance(store, TradingStateStore)
    assert store.base_dir == tmp_path
    assert store.profile_id == "olin"


# ── hermes_t CLI (__main__) tests ────────────────────────────────────────


def test_hermes_t_cli_outputs_runtime_cycle_json(tmp_path: Path):
    """CLI should execute run_runtime_cycle() and output runtime result JSON."""
    profile_dir = tmp_path / "test_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "position.json").write_text(
        json.dumps({"symbol": "688319", "total_shares": 220000}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profile-id",
            "test_profile",
            "--symbol",
            "688319",
            "--trade-unit",
            "20000",
            "--max-trades",
            "5",
            "--signal",
            "sell",
            "--score",
            "90",
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["pending"]["status"] == "pending"
    assert payload["pending"]["action"] == "sell"
    assert payload["suggestion"]["action"] == "sell"
    assert payload["suggestion"]["unit"] == 20000
    assert "summary" in payload


def test_hermes_t_cli_defaults_to_home_based_dir_and_persists_state():
    """CLI should default base_dir to ~/.hermes_t_runtime/<profile_id> and persist runtime state."""
    import uuid

    fake_home = Path("/tmp/fake_cli_home")
    profile_id = f"def_{uuid.uuid4().hex[:8]}"

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--profile-id",
            profile_id,
            "--symbol",
            "688319",
            "--trade-unit",
            "10000",
            "--signal",
            "buy",
            "--score",
            "20",
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, "HOME": str(fake_home)},
    )
    payload = json.loads(result.strip())

    assert payload["pending"]["action"] == "buy"
    state_path = fake_home / ".hermes_t_runtime" / profile_id / "execution_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["profile_id"] == profile_id
    assert state["symbol"] == "688319"


def test_cli_run_buy_signal_persists_sequence_counter(tmp_path: Path):
    """Runtime state written by first run should affect second run summary."""
    profile_dir = tmp_path / "persist_test"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "execution_state.json").write_text(
        json.dumps(
            {
                "trade_date": "2026-05-03",
                "buy_count": 1,
                "sell_count": 0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profile-id",
            "persist_test",
            "--symbol",
            "688319",
            "--trade-unit",
            "10000",
            "--signal",
            "buy",
            "--score",
            "15",
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["pending"]["seq"] == 2
    assert payload["summary"]["buy_count"] == 2


def test_hermes_t_cli_profiles_config_runs_multiple_profiles_sequentially(tmp_path: Path):
    """--profiles-config should execute each configured profile and return per-profile results."""
    profiles_config = tmp_path / "profiles.json"
    profiles_config.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile_id": "olin",
                        "symbol": "688319",
                        "trade_unit": 10000,
                        "max_trades": 4,
                    },
                    {
                        "profile_id": "test_alt",
                        "symbol": "000001",
                        "trade_unit": 5000,
                        "max_trades": 2,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    test_alt_dir = tmp_path / "test_alt"
    test_alt_dir.mkdir(parents=True, exist_ok=True)
    (test_alt_dir / "position.json").write_text(
        json.dumps({"symbol": "000001", "total_shares": 5000}, ensure_ascii=False),
        encoding="utf-8",
    )
    quote_path = tmp_path / "quotes.json"
    quote_path.write_text(
        json.dumps(
            {
                "688319": {"tech_data": {"signal": "buy", "score": 10}},
                "000001": {"tech_data": {"signal": "sell", "score": 90}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profiles-config",
            str(profiles_config),
            "--quote-data-config",
            str(quote_path),
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["profile_count"] == 2
    assert [item["profile_id"] for item in payload["results"]] == ["olin", "test_alt"]
    assert payload["results"][0]["payload"]["suggestion"]["action"] == "buy"
    assert payload["results"][1]["payload"]["suggestion"]["action"] == "sell"


def test_hermes_t_cli_dispatch_flag_returns_dispatch_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hermes_t import __main__ as cli_main

    def fake_dispatch_pending_signal(*, store, profile_id: str, dispatch_target: str = "feishu"):
        pending = store.load_pending_signal()
        return {
            "status": "sent",
            "profile_id": profile_id,
            "dispatch_target": dispatch_target,
            "signal": {**pending, "status": "sent"},
        }

    monkeypatch.setattr(cli_main, "dispatch_pending_signal", fake_dispatch_pending_signal)

    payload = cli_main._build_payload(
        cli_main.build_runtime_parser().parse_args(
            [
                "--base-dir",
                str(tmp_path),
                "--profile-id",
                "dispatch_cli",
                "--symbol",
                "688319",
                "--trade-unit",
                "10000",
                "--signal",
                "buy",
                "--score",
                "15",
                "--dispatch",
            ]
        ),
        cli_main.build_runtime_profile_from_args(
            cli_main.build_runtime_parser().parse_args(
                [
                    "--base-dir",
                    str(tmp_path),
                    "--profile-id",
                    "dispatch_cli",
                    "--symbol",
                    "688319",
                    "--trade-unit",
                    "10000",
                    "--signal",
                    "buy",
                    "--score",
                    "15",
                    "--dispatch",
                ]
            )
        ),
    )

    assert payload["pending"]["status"] == "pending"
    assert payload["dispatch"]["status"] == "sent"
    assert payload["dispatch"]["profile_id"] == "dispatch_cli"
    assert payload["dispatch"]["dispatch_target"] == "feishu"
    assert payload["dispatch"]["signal"]["status"] == "sent"


def test_run_profiles_from_config_returns_summary_for_empty_profiles_list(tmp_path: Path):
    from hermes_t.orchestrator import run_profiles_from_config
    from hermes_t.tech_data import StaticTechDataProvider

    profiles_config = tmp_path / "profiles.json"
    profiles_config.write_text(json.dumps({"profiles": []}, ensure_ascii=False), encoding="utf-8")

    payload = run_profiles_from_config(
        base_dir=tmp_path,
        profiles_config_path=profiles_config,
        tech_data_provider=StaticTechDataProvider({"signal": "hold", "score": 50}),
    )

    assert payload == {"profile_count": 0, "results": []}


def test_hermes_t_cli_profiles_config_falls_back_to_inline_signal_when_symbol_missing(tmp_path: Path):
    profiles_config = tmp_path / "profiles.json"
    profiles_config.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile_id": "olin",
                        "symbol": "688319",
                        "trade_unit": 10000,
                        "max_trades": 4,
                    },
                    {
                        "profile_id": "test_alt",
                        "symbol": "000001",
                        "trade_unit": 5000,
                        "max_trades": 2,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    olin_dir = tmp_path / "olin"
    olin_dir.mkdir(parents=True, exist_ok=True)
    (olin_dir / "position.json").write_text(
        json.dumps({"symbol": "688319", "total_shares": 220000}, ensure_ascii=False),
        encoding="utf-8",
    )
    quote_path = tmp_path / "quotes.json"
    quote_path.write_text(
        json.dumps(
            {
                "688319": {"tech_data": {"signal": "sell", "score": 90}}
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profiles-config",
            str(profiles_config),
            "--quote-data-config",
            str(quote_path),
            "--signal",
            "buy",
            "--score",
            "15",
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["results"][0]["payload"]["suggestion"]["action"] == "sell"
    assert payload["results"][1]["payload"]["suggestion"]["action"] == "buy"


def test_hermes_t_cli_single_profile_quote_data_config_falls_back_to_inline_signal_when_symbol_missing(tmp_path: Path):
    quote_path = tmp_path / "quotes.json"
    quote_path.write_text(
        json.dumps(
            {
                "000001": {"tech_data": {"signal": "sell", "score": 90}}
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profile-id",
            "single_missing_symbol",
            "--symbol",
            "688319",
            "--trade-unit",
            "10000",
            "--quote-data-config",
            str(quote_path),
            "--signal",
            "buy",
            "--score",
            "15",
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["suggestion"]["action"] == "buy"
    assert payload["pending"]["action"] == "buy"
    assert payload["pending"]["seq"] == 1


def test_hermes_t_cli_profiles_config_prefers_tech_data_config_over_quote_data_config(tmp_path: Path):
    profiles_config = tmp_path / "profiles.json"
    profiles_config.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile_id": "olin",
                        "symbol": "688319",
                        "trade_unit": 10000,
                        "max_trades": 4,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tech_data_path = tmp_path / "tech_data.json"
    tech_data_path.write_text(
        json.dumps({"688319": {"signal": "buy", "score": 12}}, ensure_ascii=False),
        encoding="utf-8",
    )
    quote_path = tmp_path / "quotes.json"
    quote_path.write_text(
        json.dumps({"688319": {"tech_data": {"signal": "sell", "score": 90}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profiles-config",
            str(profiles_config),
            "--tech-data-config",
            str(tech_data_path),
            "--quote-data-config",
            str(quote_path),
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["results"][0]["payload"]["suggestion"]["action"] == "buy"


# ── Quote provider tests ──────────────────────────────────────────────────


def test_json_quote_data_provider_returns_raw_payload_for_symbol(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteDataProvider

    quote_map = {
        "688319": {
            "symbol": "688319",
            "last_price": 18.23,
            "tech_data": {"signal": "sell", "score": 88},
        }
    }
    config_path = tmp_path / "quotes.json"
    config_path.write_text(json.dumps(quote_map, ensure_ascii=False), encoding="utf-8")

    provider = JsonQuoteDataProvider(config_path)

    assert provider.get("688319") == quote_map["688319"]


def test_json_quote_data_provider_returns_empty_dict_for_unknown_symbol(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteDataProvider

    config_path = tmp_path / "quotes.json"
    config_path.write_text(json.dumps({"688319": {"last_price": 18.23}}), encoding="utf-8")

    provider = JsonQuoteDataProvider(config_path)

    assert provider.get("000001") == {}


def test_quote_tech_data_adapter_extracts_tech_data(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteDataProvider, QuoteTechDataAdapter

    quote_map = {
        "688319": {
            "symbol": "688319",
            "last_price": 18.23,
            "tech_data": {"signal": "sell", "score": 88},
        }
    }
    config_path = tmp_path / "quotes.json"
    config_path.write_text(json.dumps(quote_map, ensure_ascii=False), encoding="utf-8")

    adapter = QuoteTechDataAdapter(
        JsonQuoteDataProvider(config_path),
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert adapter.get("688319") == {"signal": "sell", "score": 88}


def test_quote_tech_data_adapter_returns_default_when_tech_data_missing(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteDataProvider, QuoteTechDataAdapter

    config_path = tmp_path / "quotes.json"
    config_path.write_text(json.dumps({"688319": {"last_price": 18.23}}), encoding="utf-8")

    adapter = QuoteTechDataAdapter(
        JsonQuoteDataProvider(config_path),
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert adapter.get("688319") == {"signal": "hold", "score": 50}


def test_build_tech_data_provider_from_quote_data_config_returns_adapter(tmp_path: Path):
    from hermes_t.tech_data import QuoteTechDataAdapter, build_tech_data_provider

    config_path = tmp_path / "quotes.json"
    config_path.write_text(
        json.dumps({"688319": {"tech_data": {"signal": "buy", "score": 20}}}),
        encoding="utf-8",
    )

    provider = build_tech_data_provider(
        quote_data_config_path=config_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert isinstance(provider, QuoteTechDataAdapter)
    assert provider.get("688319") == {"signal": "buy", "score": 20}


def test_build_tech_data_provider_without_config_returns_default_provider():
    from hermes_t.tech_data import build_tech_data_provider

    provider = build_tech_data_provider(default_tech_data={"signal": "hold", "score": 50})

    assert provider.get("688319") == {"signal": "hold", "score": 50}


def test_hermes_t_cli_with_quote_data_config_uses_file_provider(tmp_path: Path):
    """--quote-data-config should load tech_data from file instead of inline args."""
    import subprocess
    import sys
    from pathlib import Path

    profile_dir = tmp_path / "quote_cli_test"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "position.json").write_text(
        json.dumps({"symbol": "688319", "total_shares": 220000}, ensure_ascii=False),
        encoding="utf-8",
    )

    quote_path = tmp_path / "quotes.json"
    quote_path.write_text(
        json.dumps(
            {"688319": {"tech_data": {"signal": "sell", "score": 85}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profile-id",
            "quote_cli_test",
            "--symbol",
            "688319",
            "--trade-unit",
            "10000",
            "--quote-data-config",
            str(quote_path),
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["suggestion"]["action"] == "sell"
    assert payload["suggestion"]["unit"] == 10000


def test_hermes_t_cli_with_quote_snapshot_config_uses_snapshot_provider(tmp_path: Path):
    """--quote-snapshot-config should load tech_data from snapshot factory config."""
    import subprocess
    import sys
    from pathlib import Path

    snapshot_path = tmp_path / "snapshots.json"
    snapshot_path.write_text(
        json.dumps(
            {"688319": {"tech_data": {"signal": "buy", "score": 15}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    factory_path = tmp_path / "quote_snapshot_config.json"
    factory_path.write_text(
        json.dumps({"source": "file", "snapshot_path": str(snapshot_path)}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "hermes_t",
            "--base-dir",
            str(tmp_path),
            "--profile-id",
            "snapshot_cli_test",
            "--symbol",
            "688319",
            "--trade-unit",
            "10000",
            "--quote-snapshot-config",
            str(factory_path),
        ],
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.strip())

    assert payload["suggestion"]["action"] == "buy"
    assert payload["pending"]["status"] == "pending"


# ── Quote snapshot source tests ────────────────────────────────────────────


def test_json_quote_snapshot_source_returns_snapshot_for_symbol(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteSnapshotSource

    snapshot_map = {
        "688319": {
            "symbol": "688319",
            "last_price": 18.88,
            "source": "file",
        }
    }
    snapshot_path = tmp_path / "snapshots.json"
    snapshot_path.write_text(json.dumps(snapshot_map, ensure_ascii=False), encoding="utf-8")

    source = JsonQuoteSnapshotSource(snapshot_path)

    assert source.get("688319") == snapshot_map["688319"]


def test_json_quote_snapshot_source_returns_empty_dict_for_unknown_symbol(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteSnapshotSource

    snapshot_path = tmp_path / "snapshots.json"
    snapshot_path.write_text(json.dumps({"688319": {"last_price": 18.88}}), encoding="utf-8")

    source = JsonQuoteSnapshotSource(snapshot_path)

    assert source.get("000001") == {}


def test_build_quote_snapshot_provider_with_unsupported_source_raises():
    from hermes_t.tech_data import build_quote_snapshot_provider

    with pytest.raises(ValueError, match="unsupported quote snapshot source"):
        build_quote_snapshot_provider({"source": "unknown"})


class _RaisingSnapshotSource:
    def get(self, symbol: str) -> dict[str, object]:
        raise RuntimeError(f"boom:{symbol}")


def test_quote_snapshot_tech_data_adapter_extracts_tech_data(tmp_path: Path):
    from hermes_t.tech_data import JsonQuoteSnapshotSource, QuoteSnapshotTechDataAdapter

    snapshot_map = {
        "688319": {
            "symbol": "688319",
            "last_price": 18.66,
            "source": "file",
            "tech_data": {"signal": "buy", "score": 22},
        }
    }
    snapshot_path = tmp_path / "snapshots.json"
    snapshot_path.write_text(json.dumps(snapshot_map, ensure_ascii=False), encoding="utf-8")

    adapter = QuoteSnapshotTechDataAdapter(
        JsonQuoteSnapshotSource(snapshot_path),
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert adapter.get("688319") == {"signal": "buy", "score": 22}


def test_quote_snapshot_tech_data_adapter_falls_back_to_default_on_source_error():
    from hermes_t.tech_data import QuoteSnapshotTechDataAdapter

    adapter = QuoteSnapshotTechDataAdapter(
        _RaisingSnapshotSource(),
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert adapter.get("688319") == {"signal": "hold", "score": 50}


def test_build_tech_data_provider_from_quote_snapshot_config_returns_adapter(tmp_path: Path):
    from hermes_t.tech_data import QuoteSnapshotTechDataAdapter, build_tech_data_provider

    factory_path = tmp_path / "quote_snapshot_config.json"
    snapshot_path = tmp_path / "snapshots.json"
    snapshot_path.write_text(
        json.dumps({"688319": {"tech_data": {"signal": "sell", "score": 77}}}),
        encoding="utf-8",
    )
    factory_path.write_text(
        json.dumps({"source": "file", "snapshot_path": str(snapshot_path)}, ensure_ascii=False),
        encoding="utf-8",
    )

    provider = build_tech_data_provider(
        quote_snapshot_config_path=factory_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert isinstance(provider, QuoteSnapshotTechDataAdapter)
    assert provider.get("688319") == {"signal": "sell", "score": 77}


def test_build_tech_data_provider_quote_data_config_takes_precedence_over_quote_snapshot_config(tmp_path: Path):
    from hermes_t.tech_data import QuoteTechDataAdapter, build_tech_data_provider

    quote_data_path = tmp_path / "quotes.json"
    quote_snapshot_factory_path = tmp_path / "quote_snapshot_config.json"
    snapshot_path = tmp_path / "snapshots.json"

    quote_data_path.write_text(
        json.dumps({"688319": {"tech_data": {"signal": "buy", "score": 11}}}),
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps({"688319": {"tech_data": {"signal": "sell", "score": 88}}}),
        encoding="utf-8",
    )
    quote_snapshot_factory_path.write_text(
        json.dumps({"source": "file", "snapshot_path": str(snapshot_path)}, ensure_ascii=False),
        encoding="utf-8",
    )

    provider = build_tech_data_provider(
        quote_data_config_path=quote_data_path,
        quote_snapshot_config_path=quote_snapshot_factory_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert isinstance(provider, QuoteTechDataAdapter)
    assert provider.get("688319") == {"signal": "buy", "score": 11}


def test_build_tech_data_provider_resolves_relative_snapshot_path_from_config_dir(tmp_path: Path):
    from hermes_t.tech_data import build_tech_data_provider

    config_dir = tmp_path / "configs"
    snapshots_dir = config_dir / "data"
    snapshots_dir.mkdir(parents=True)

    snapshot_path = snapshots_dir / "snapshots.json"
    snapshot_path.write_text(
        json.dumps({"688319": {"tech_data": {"signal": "sell", "score": 66}}}),
        encoding="utf-8",
    )

    factory_path = config_dir / "quote_snapshot_config.json"
    factory_path.write_text(
        json.dumps({"source": "file", "snapshot_path": "data/snapshots.json"}, ensure_ascii=False),
        encoding="utf-8",
    )

    provider = build_tech_data_provider(
        quote_snapshot_config_path=factory_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert provider.get("688319") == {"signal": "sell", "score": 66}


class _FakeTdxAPI:
    server_plans: dict[tuple[str, int], dict[str, object]] = {}
    instances: list["_FakeTdxAPI"] = []

    def __init__(self):
        self.connected_to: tuple[str, int] | None = None
        self.disconnected = False
        self.calls: list[tuple[int, str]] = []
        type(self).instances.append(self)

    def connect(self, host: str, port: int) -> bool:
        plan = type(self).server_plans[(host, port)]
        if plan.get("connect_error"):
            raise plan["connect_error"]  # type: ignore[misc]
        if not plan.get("connect", True):
            return False
        self.connected_to = (host, port)
        return True

    def get_security_quotes(self, codes: list[tuple[int, str]]):
        assert self.connected_to is not None
        plan = type(self).server_plans[self.connected_to]
        self.calls.extend(codes)
        error = plan.get("quote_error")
        if error:
            raise error  # type: ignore[misc]
        return plan.get("quotes", [])

    def disconnect(self):
        self.disconnected = True



def test_build_quote_snapshot_provider_with_tdx_source_returns_tdx_source():
    from hermes_t.tech_data import TdxQuoteSnapshotSource, build_quote_snapshot_provider

    provider = build_quote_snapshot_provider(
        {
            "source": "tdx",
            "api_cls": _FakeTdxAPI,
            "servers": [["127.0.0.1", 7709]],
        }
    )

    assert isinstance(provider, TdxQuoteSnapshotSource)



def test_tdx_quote_snapshot_source_market_by_symbol_overrides_default_market():
    from hermes_t.tech_data import TdxQuoteSnapshotSource

    _FakeTdxAPI.instances = []
    _FakeTdxAPI.server_plans = {
        ("good-host", 7710): {
            "quotes": [
                {
                    "price": 18.52,
                    "servertime": "14:55:01",
                }
            ]
        },
    }

    source = TdxQuoteSnapshotSource(
        api_cls=_FakeTdxAPI,
        servers=[("good-host", 7710)],
        market_by_symbol={"688319": 0},
    )

    snapshot = source.get("688319")

    assert snapshot["last_price"] == 18.52
    assert len(_FakeTdxAPI.instances) == 1
    assert _FakeTdxAPI.instances[0].calls == [(0, "688319")]



def test_tdx_quote_snapshot_source_fails_over_and_normalizes_snapshot():
    from hermes_t.tech_data import TdxQuoteSnapshotSource

    _FakeTdxAPI.instances = []
    _FakeTdxAPI.server_plans = {
        ("bad-host", 7709): {"connect_error": ConnectionError("first server down")},
        ("good-host", 7710): {
            "quotes": [
                {
                    "price": 18.52,
                    "servertime": "14:55:01",
                }
            ]
        },
    }

    source = TdxQuoteSnapshotSource(
        api_cls=_FakeTdxAPI,
        servers=[("bad-host", 7709), ("good-host", 7710)],
    )

    snapshot = source.get("688319")

    assert snapshot["symbol"] == "688319"
    assert snapshot["last_price"] == 18.52
    assert snapshot["source"] == "tdx_tcp"
    assert snapshot["as_of"] == "14:55:01"
    assert len(_FakeTdxAPI.instances) == 2
    assert _FakeTdxAPI.instances[1].calls == [(1, "688319")]
    assert _FakeTdxAPI.instances[1].disconnected is True


def test_tdx_quote_snapshot_source_raises_when_all_servers_fail():
    from hermes_t.tech_data import TdxQuoteSnapshotSource

    _FakeTdxAPI.instances = []
    _FakeTdxAPI.server_plans = {
        ("bad-host-1", 7709): {"connect_error": ConnectionError("down1")},
        ("bad-host-2", 7710): {"quote_error": RuntimeError("down2"), "quotes": []},
    }

    source = TdxQuoteSnapshotSource(
        api_cls=_FakeTdxAPI,
        servers=[("bad-host-1", 7709), ("bad-host-2", 7710)],
    )

    with pytest.raises(RuntimeError, match="all tdx servers failed"):
        source.get("688319")



def test_tdx_quote_snapshot_source_raises_on_empty_quote():
    from hermes_t.tech_data import TdxQuoteSnapshotSource

    _FakeTdxAPI.instances = []
    _FakeTdxAPI.server_plans = {
        ("good-host", 7710): {"quotes": []},
    }

    source = TdxQuoteSnapshotSource(
        api_cls=_FakeTdxAPI,
        servers=[("good-host", 7710)],
    )

    with pytest.raises(ValueError, match="empty tdx quote"):
        source.get("688319")



def test_tdx_quote_snapshot_source_raises_on_non_positive_price():
    from hermes_t.tech_data import TdxQuoteSnapshotSource

    _FakeTdxAPI.instances = []
    _FakeTdxAPI.server_plans = {
        ("good-host", 7710): {"quotes": [{"price": 0, "servertime": "14:55:01"}]},
    }

    source = TdxQuoteSnapshotSource(
        api_cls=_FakeTdxAPI,
        servers=[("good-host", 7710)],
    )

    with pytest.raises(ValueError, match="non-positive tdx price"):
        source.get("688319")



def test_build_tech_data_provider_with_tdx_snapshot_config_falls_back_to_default_on_source_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hermes_t.tech_data import QuoteSnapshotTechDataAdapter, build_tech_data_provider
    import hermes_t.tech_data as tech_data_module

    factory_path = tmp_path / "quote_snapshot_tdx.json"
    factory_path.write_text(
        json.dumps(
            {
                "source": "tdx",
                "servers": [["bad-host", 7709]],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tech_data_module, "_import_tdx_api_cls", lambda: _FakeTdxAPI)
    _FakeTdxAPI.instances = []
    _FakeTdxAPI.server_plans = {
        ("bad-host", 7709): {"connect_error": ConnectionError("down")},
    }

    provider = build_tech_data_provider(
        quote_snapshot_config_path=factory_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert isinstance(provider, QuoteSnapshotTechDataAdapter)
    assert provider.get("688319") == {"signal": "hold", "score": 50}



def test_build_quote_snapshot_provider_with_eastmoney_source_returns_placeholder_source():
    from hermes_t.tech_data import EastmoneyQuoteSnapshotSource, build_quote_snapshot_provider

    provider = build_quote_snapshot_provider({"source": "eastmoney"})

    assert isinstance(provider, EastmoneyQuoteSnapshotSource)



def test_eastmoney_quote_snapshot_source_uses_default_http_fetcher(monkeypatch: pytest.MonkeyPatch):
    from hermes_t.tech_data import EastmoneyQuoteSnapshotSource
    import hermes_t.tech_data as tech_data_module

    calls: list[str] = []

    def _fake_fetcher(secid: str):
        calls.append(secid)
        return {"data": {"f43": 185200, "f86": "20260503 14:55:01"}}

    monkeypatch.setattr(tech_data_module, "_import_eastmoney_fetcher", lambda: _fake_fetcher)

    source = EastmoneyQuoteSnapshotSource()

    assert source.get("688319") == {
        "symbol": "688319",
        "last_price": 18.52,
        "source": "eastmoney",
        "as_of": "20260503 14:55:01",
    }
    assert calls == ["1.688319"]



def test_build_tech_data_provider_with_eastmoney_snapshot_config_falls_back_to_default(tmp_path: Path):
    from hermes_t.tech_data import QuoteSnapshotTechDataAdapter, build_tech_data_provider

    factory_path = tmp_path / "quote_snapshot_eastmoney.json"
    factory_path.write_text(
        json.dumps(
            {
                "source": "eastmoney",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    provider = build_tech_data_provider(
        quote_snapshot_config_path=factory_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert isinstance(provider, QuoteSnapshotTechDataAdapter)
    assert provider.get("688319") == {"signal": "hold", "score": 50}


class _FakeEastmoneyFetcher:
    payloads_by_secid: dict[str, object] = {}
    calls: list[str] = []

    def __call__(self, secid: str):
        type(self).calls.append(secid)
        payload = type(self).payloads_by_secid[secid]
        if isinstance(payload, Exception):
            raise payload
        return payload



def test_eastmoney_quote_snapshot_source_normalizes_snapshot_and_market_override():
    from hermes_t.tech_data import EastmoneyQuoteSnapshotSource

    _FakeEastmoneyFetcher.payloads_by_secid = {
        "0.688319": {
            "data": {
                "f43": 185200,
                "f86": "20260503 14:55:01",
                "tech_data": {"signal": "sell", "score": 72},
            }
        }
    }
    _FakeEastmoneyFetcher.calls = []

    source = EastmoneyQuoteSnapshotSource(
        fetcher=_FakeEastmoneyFetcher(),
        market_by_symbol={"688319": 0},
    )

    snapshot = source.get("688319")

    assert snapshot == {
        "symbol": "688319",
        "last_price": 18.52,
        "source": "eastmoney",
        "as_of": "20260503 14:55:01",
        "tech_data": {"signal": "sell", "score": 72},
    }
    assert _FakeEastmoneyFetcher.calls == ["0.688319"]



def test_build_tech_data_provider_with_eastmoney_snapshot_config_returns_adapter_and_extracts_tech_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from hermes_t.tech_data import QuoteSnapshotTechDataAdapter, build_tech_data_provider
    import hermes_t.tech_data as tech_data_module

    factory_path = tmp_path / "quote_snapshot_eastmoney_live.json"
    factory_path.write_text(
        json.dumps(
            {
                "source": "eastmoney",
                "market_by_symbol": {"688319": 0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _FakeEastmoneyFetcher.payloads_by_secid = {
        "0.688319": {
            "data": {
                "f43": 185200,
                "f86": "20260503 14:55:01",
                "tech_data": {"signal": "sell", "score": 72},
            }
        }
    }
    _FakeEastmoneyFetcher.calls = []
    monkeypatch.setattr(tech_data_module, "_import_eastmoney_fetcher", lambda: _FakeEastmoneyFetcher())

    provider = build_tech_data_provider(
        quote_snapshot_config_path=factory_path,
        default_tech_data={"signal": "hold", "score": 50},
    )

    assert isinstance(provider, QuoteSnapshotTechDataAdapter)
    assert provider.get("688319") == {"signal": "sell", "score": 72}
    assert _FakeEastmoneyFetcher.calls == ["0.688319"]


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True

    def json(self):
        return self.payload



def test_import_eastmoney_fetcher_returns_http_fetcher_using_expected_request_contract(monkeypatch: pytest.MonkeyPatch):
    import hermes_t.tech_data as tech_data_module

    calls: list[dict[str, object]] = []
    response = _FakeResponse({"data": {"f43": 185200}})

    class _FakeRequestsModule:
        @staticmethod
        def get(url: str, *, params: dict[str, object], timeout: int):
            calls.append({"url": url, "params": params, "timeout": timeout})
            return response

    monkeypatch.setattr(tech_data_module, "_import_requests_module", lambda: _FakeRequestsModule)

    fetcher = tech_data_module._import_eastmoney_fetcher()
    payload = fetcher("0.688319")

    assert payload == {"data": {"f43": 185200}}
    assert calls == [
        {
            "url": "https://push2.eastmoney.com/api/qt/stock/get",
            "params": {
                "secid": "0.688319",
                "fields": "f43,f57,f58,f86",
                "invt": "2",
                "fltt": "1",
            },
            "timeout": 10,
        }
    ]
    assert response.raise_for_status_called is True
