"""
AI Scientist Runner — SakanaAI/AI-Scientist の Slim モード統合ラッパー。

LaTeX/PDF 生成はスキップし、アイデア生成と実験実行のみを行う。
Ollama 互換: ``ollama/<model>`` または ``openai-compatible/<model>`` で
``OLLAMA_BASE_URL`` 経由の OpenAI 互換 API を使う。

Redis キー:
  atlas:failures         (LIST, 読み取り) → リサーチテーマ自動設定
  ai_scientist:findings  (LIST, 書き込み, max 200) → 発見ログ
  ai_scientist:tasks     (LIST, 読み取り) → 外部タスク投入
  shinka:fitness_hints   (LIST, 書き込み) → ShinkaEvolve 改善ヒント

デーモンモード (このファイルを直接実行した場合):
  AI_SCIENTIST_INTERVAL_SEC (default: 7200) 秒毎に run_from_failures() を実行。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _ensure_hermes_repo_on_path() -> None:
    """Allow importing ``tools.ai_scientist_env`` from the Hermes checkout."""
    try:
        import tools.ai_scientist_env  # noqa: F401

        return
    except ImportError:
        pass
    candidate = Path(__file__).resolve()
    for _ in range(8):
        candidate = candidate.parent
        if (candidate / "tools" / "ai_scientist_env.py").is_file():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def _apply_hermes_credentials(model: Optional[str]) -> None:
    _ensure_hermes_repo_on_path()
    try:
        from tools.ai_scientist_env import apply_ai_scientist_run_config

        apply_ai_scientist_run_config(model)
    except Exception as exc:
        logger.debug("Hermes credential bridge unavailable: %s", exc)


# vendor/AI-Scientist を sys.path に追加
_AI_SCIENTIST_DIR = os.environ.get(
    "AI_SCIENTIST_DIR",
    str(Path(__file__).parent.parent.parent.parent / "AI-Scientist"),
)
if Path(_AI_SCIENTIST_DIR).exists() and _AI_SCIENTIST_DIR not in sys.path:
    sys.path.insert(0, _AI_SCIENTIST_DIR)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
_ollama_raw = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
OLLAMA_BASE_URL = _ollama_raw if _ollama_raw.endswith("/v1") else _ollama_raw + "/v1"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen-hakua-core:latest")
DEFAULT_CLOUD_MODEL = os.environ.get("AI_SCIENTIST_MODEL", "auto")
INTERVAL_SEC = int(os.environ.get("AI_SCIENTIST_INTERVAL_SEC", "7200"))
MAX_IDEAS = int(os.environ.get("AI_SCIENTIST_MAX_IDEAS", "5"))
MAX_FINDINGS = 200
NUM_REFLECTIONS = int(os.environ.get("AI_SCIENTIST_NUM_REFLECTIONS", "2"))


def resolve_ai_scientist_client(model: Optional[str]) -> Tuple[Any, str]:
    """Map Hermes model aliases to SakanaAI ``create_client`` or Ollama OpenAI shim."""
    raw = (model or DEFAULT_CLOUD_MODEL).strip()
    _apply_hermes_credentials(raw)

    if raw.startswith(("ollama/", "openai-compatible/")):
        import openai

        client_model = raw.split("/", 1)[1]
        base = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL")
            or OLLAMA_BASE_URL
        ).rstrip("/")
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            base_url=base,
        )
        return client, client_model

    bridge = os.environ.get("AI_SCIENTIST_HERMES_BRIDGE", "").strip() == "1"
    base = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip().rstrip("/")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    api_model = (os.environ.get("AI_SCIENTIST_API_MODEL") or raw).strip()
    force_shim = os.environ.get("AI_SCIENTIST_FORCE_OPENAI_SHIM", "").strip() == "1"
    gpt_like = "gpt" in raw or "o1" in raw or "o3" in raw

    if bridge and base and api_key and (gpt_like or force_shim):
        import openai

        return openai.OpenAI(api_key=api_key, base_url=base), api_model

    from ai_scientist.llm import create_client

    normalized = raw.removeprefix("ollama/")
    return create_client(normalized)


def template_dir(template: str) -> Path:
    return Path(_AI_SCIENTIST_DIR) / "templates" / template


# ── Redis ヘルパー ──────────────────────────────────────────────────────────

def _get_redis():
    try:
        import redis as redis_lib

        url = REDIS_URL.removeprefix("redis://")
        host, _, port = url.partition(":")
        r = redis_lib.Redis(host=host or "localhost", port=int(port or 6379), decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        logger.debug("Redis unavailable: %s", e)
        return None


def _push_finding(r, topic: str, idea: dict, result: dict) -> None:
    record = {
        "topic": topic,
        "idea": idea,
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    r.rpush("ai_scientist:findings", json.dumps(record, ensure_ascii=False))
    r.ltrim("ai_scientist:findings", -MAX_FINDINGS, -1)


def _push_fitness_hint(r, hint: str) -> None:
    if not hint:
        return
    r.rpush("shinka:fitness_hints", hint[:100])
    r.ltrim("shinka:fitness_hints", -100, -1)


# ── AI-Scientist ラッパー ─────────────────────────────────────────────────

class AiScientistRunner:
    """SakanaAI/AI-Scientist の Slim モード (アイデア生成 + 実験、LaTeX なし)。"""

    def __init__(self) -> None:
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import ai_scientist  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "ai_scientist package not found at %s. "
                "Run: py -3 scripts/sync_ai_scientist_vendor.py --execute",
                _AI_SCIENTIST_DIR,
            )
            return False

    # ── public API ────────────────────────────────────────────────────────

    def run_ideas(
        self,
        topic: str,
        template: str = "nanoGPT_lite",
        num_ideas: int = 3,
        model: Optional[str] = None,
    ) -> list[dict]:
        """AI-Scientist のアイデア生成を実行する。"""
        if not self._available:
            return self._fallback_ideas(topic, num_ideas, model)

        try:
            return self._run_ideas_sakana(topic, template, num_ideas, model)
        except Exception as e:
            logger.warning("SakanaAI idea generation failed (%s), falling back: %s", type(e).__name__, e)
            return self._fallback_ideas(topic, num_ideas, model)

    def run_experiment(self, idea: dict, template: str = "nanoGPT_lite", model: Optional[str] = None) -> dict:
        """AI-Scientist の実験実行 (perform_experiments + aider) を呼ぶ。"""
        if not self._available:
            return {"success": False, "error": "ai_scientist not available"}

        try:
            return self._run_experiment_sakana(idea, template, model)
        except Exception as e:
            logger.warning("SakanaAI experiment failed: %s", e)
            return {"success": False, "error": str(e)}

    def run_from_failures(self, model: Optional[str] = None) -> dict:
        """
        atlas:failures を読んでリサーチテーマを自動設定し、アイデアを生成する。
        発見を ai_scientist:findings + shinka:fitness_hints に保存する。
        """
        r = _get_redis()
        if r is None:
            return {"success": False, "error": "Redis unavailable"}

        raw_failures = r.lrange("atlas:failures", -20, -1)
        failures = []
        for raw in raw_failures:
            try:
                failures.append(json.loads(raw))
            except json.JSONDecodeError:
                pass

        if not failures:
            topic = "improve code generation quality"
        else:
            errors = [f.get("error", "")[:80] for f in failures if f.get("error")]
            stop_reasons = list({f.get("stop_reason", "") for f in failures})
            topic = (
                f"Fix common code generation failures: {', '.join(stop_reasons[:3])}. "
                f"Top errors: {'; '.join(errors[:3])}"
            )[:200]

        logger.info("AI-Scientist run_from_failures: topic=%s", topic[:80])
        ideas = self.run_ideas(topic, num_ideas=MAX_IDEAS, model=model)

        stored = 0
        for idea in ideas:
            _push_finding(r, topic, idea, {})
            hint = idea.get("fitness_hint") or idea.get("Interestingness") or ""
            if hint:
                _push_fitness_hint(r, str(hint)[:100])
                stored += 1

        return {
            "success": True,
            "topic": topic,
            "ideas_generated": len(ideas),
            "hints_stored": stored,
        }

    # ── internal: SakanaAI 実装 ──────────────────────────────────────────

    def _run_ideas_sakana(self, topic: str, template: str, num_ideas: int, model: Optional[str]) -> list[dict]:
        from ai_scientist.generate_ideas import generate_ideas  # type: ignore[import]

        base = template_dir(template)
        if not base.is_dir():
            raise FileNotFoundError(f"template not found: {base}")

        client, client_model = resolve_ai_scientist_client(model)
        ideas = generate_ideas(
            str(base),
            client=client,
            model=client_model,
            skip_generation=False,
            max_num_generations=num_ideas,
            num_reflections=NUM_REFLECTIONS,
        )
        if topic and isinstance(ideas, list):
            for idea in ideas:
                if isinstance(idea, dict):
                    idea.setdefault("Topic", topic)
        return ideas if isinstance(ideas, list) else []

    def _run_experiment_sakana(self, idea: dict, template: str, model: Optional[str]) -> dict:
        import os.path as osp
        import shutil
        from datetime import datetime as dt

        _ensure_hermes_repo_on_path()
        try:
            from tools.ai_scientist_deps import ensure_ai_scientist_deps

            ensure_ai_scientist_deps(prompt=False)
        except Exception as exc:
            return {
                "success": False,
                "error": f"AI-Scientist deps missing (uv sync --extra ai-scientist): {exc}",
            }

        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
        from ai_scientist.perform_experiments import perform_experiments  # type: ignore[import]

        base_dir = template_dir(template)
        if not base_dir.is_dir():
            return {"success": False, "error": f"template not found: {base_dir}"}

        _, client_model = resolve_ai_scientist_client(model)
        aider_model = (os.environ.get("AI_SCIENTIST_API_MODEL") or client_model).strip()
        baseline_path = base_dir / "run_0" / "final_info.json"
        if not baseline_path.is_file():
            return {"success": False, "error": f"baseline missing: {baseline_path}"}

        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline_results = json.load(handle)
        if isinstance(baseline_results, dict):
            baseline_results = {k: v["means"] for k, v in baseline_results.items() if isinstance(v, dict)}

        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        idea_name = f"{timestamp}_{idea.get('Name', 'idea')}"
        folder_name = Path(_AI_SCIENTIST_DIR) / "results" / template / idea_name
        if folder_name.exists():
            return {"success": False, "error": f"folder already exists: {folder_name}"}

        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)
        exp_file = folder_name / "experiment.py"
        vis_file = folder_name / "plot.py"
        notes = folder_name / "notes.txt"
        with notes.open("w", encoding="utf-8") as handle:
            handle.write(f"# Title: {idea.get('Title', '')}\n")
            handle.write(f"# Experiment description: {idea.get('Experiment', '')}\n")
            handle.write(f"## Run 0: Baseline\n")
            handle.write(f"Results: {baseline_results}\n")

        io = InputOutput(yes=True, chat_history_file=str(folder_name / f"{idea_name}_aider.txt"))
        main_model = Model(aider_model)
        coder = Coder.create(
            main_model=main_model,
            fnames=[str(exp_file), str(vis_file), str(notes)],
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
        success = perform_experiments(idea, str(folder_name), coder, baseline_results)
        return {
            "success": bool(success),
            "folder": str(folder_name),
            "message": "experiments completed" if success else "experiments failed",
        }

    # ── fallback: LLM 直接呼び出し (SakanaAI が import できない場合) ──────

    def _fallback_ideas(self, topic: str, num_ideas: int, model: Optional[str]) -> list[dict]:
        """ai_scientist パッケージが利用不可の場合、Ollama に直接問い合わせてアイデアを生成する。"""
        import requests

        m = (model or f"ollama/{OLLAMA_MODEL}").removeprefix("ollama/").removeprefix("openai-compatible/")
        system_prompt = (
            "You are an AI research scientist generating novel research ideas. "
            "Given a topic, output a JSON array of research ideas. "
            "Each idea must have: Name, Title, Experiment, Interestingness (1-10), "
            "Feasibility (1-10), Novelty (1-10), fitness_hint (actionable improvement directive, max 100 chars). "
            "Return ONLY valid JSON array."
        )
        user_msg = f"Generate {num_ideas} research ideas for: {topic}"

        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/chat/completions",
                json={
                    "model": m,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return self._parse_ideas_json(content)
        except Exception as e:
            logger.error("Fallback idea generation failed: %s", e)
            return []

    def _parse_ideas_json(self, content: str) -> list[dict]:
        import re

        for pattern in [r"```json\n?(.*?)```", r"```\n?(.*?)```", r"(\[.*?\])"]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    pass
        try:
            data = json.loads(content.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []


# ── デーモンエントリポイント ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    runner = AiScientistRunner()
    logger.info(
        "AI Scientist Runner starting (interval=%ds, model=%s, available=%s)",
        INTERVAL_SEC,
        DEFAULT_CLOUD_MODEL,
        runner._available,
    )
    while True:
        try:
            result = runner.run_from_failures()
            logger.info("Cycle done: %s", result)
        except Exception as e:
            logger.error("Cycle error: %s", e)
        logger.info("Next cycle in %d seconds", INTERVAL_SEC)
        time.sleep(INTERVAL_SEC)
