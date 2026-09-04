#!/usr/bin/env python3
"""HERMES Local — تغطية كاملة لباقي قدرات المنظومة المحلية.

يكمّل ``hermes_local_tool`` الذي يغطّي الاستدلال والمعرفة والسوق والحالة.
هنا الـ 17 قدرة الباقية، مجمّعة في خمس أدوات بمعامل ``action`` بدل سبع عشرة
أداة منفصلة تُثقل كل نداء:

    hermes_task      مهام متعددة الخطوات: تفكيك + توازي + استئناف بعد السقوط
    hermes_schedule  الجدولة المتكررة + الموجز الصباحي + أنبوب رسائل الوكلاء
    hermes_ops       المراقبة والإصلاح الذاتي + الموديلات + النسخ الاحتياطي
                     + واجهة المتصفح + مراقبة الملفات + ضغط السياق
    hermes_codex     Codex أوف‑لاين: الحالة، البروكسي، فهرسة وثائقه
    hermes_web       بحث متعدّد المصادر مع تحلّل رشيق + بحث وقراءة وتلخيص

كل أداة تتحلّل برشاقة: غياب المنظومة أو سقوط Ollama يُرجع خطأً واضحاً بلا
انهيار، ولا تُسجَّل أصلاً إن لم يكن مجلد المنظومة موجوداً.
"""

from __future__ import annotations

import json

from tools.hermes_local_tool import _engine, _in_dir, _ok, check_hermes_local
from tools.registry import registry, tool_error


def _eng():
    """يُرجع (engine, None) أو (None, رسالة خطأ)."""
    try:
        return _engine(), None
    except RuntimeError as exc:
        return None, tool_error(str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# 1) hermes_task — مهام متعددة الخطوات
# ═══════════════════════════════════════════════════════════════════════════════

def hermes_task_tool(task: str, steps: str = "", max_parallel: int = 3) -> str:
    task = (task or "").strip()
    if not task:
        return tool_error("task مطلوب.")
    eng, err = _eng()
    if err:
        return err
    try:
        step_list = None
        if steps:
            step_list = [s.strip() for s in steps.split("|") if s.strip()]
        r = _in_dir(eng.run_big_task, task, step_list,
                    max(1, min(int(max_parallel or 3), 8)))
        return _ok({"run_id": r["run_id"], "steps": r["steps"],
                    "results": {k: str(v)[:700] for k, v in r["results"].items()}})
    except Exception as exc:
        return tool_error(f"فشل تنفيذ المهمة: {type(exc).__name__}: {exc}")


HERMES_TASK_SCHEMA = {
    "name": "hermes_task",
    "description": (
        "Run a large multi-step task on the user's LOCAL stack. Independent steps "
        "execute in parallel, each step is checkpointed to SQLite, and a crashed "
        "run resumes from where it stopped instead of restarting. Use it for work "
        "that decomposes into several sub-answers; for a single question use "
        "hermes_local_ask instead. Runs offline on the local model."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The overall task."},
            "steps": {"type": "string",
                      "description": "Optional explicit steps separated by '|'. "
                                     "Omit to let the local model decompose."},
            "max_parallel": {"type": "integer",
                             "description": "Concurrent steps, 1-8 (default 3)."},
        },
        "required": ["task"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2) hermes_schedule — جدولة + موجز صباحي + أنبوب
# ═══════════════════════════════════════════════════════════════════════════════

def hermes_schedule_tool(action: str = "list", task: str = "",
                         minutes: int = 0, hour: int = -1, minute: int = 0,
                         label: str = "", topic: str = "default",
                         message: str = "") -> str:
    eng, err = _eng()
    if err:
        return err
    try:
        s = eng.scheduler
        if action == "add_interval":
            if not task or minutes <= 0:
                return tool_error("add_interval يحتاج task و minutes > 0.")
            _in_dir(s.every_minutes, int(minutes), task, "research",
                    label or f"كل {minutes}د")
            return _ok({"added": label or f"كل {minutes}د", "jobs": len(s._jobs)})
        if action == "add_daily":
            if not task or not (0 <= hour <= 23):
                return tool_error("add_daily يحتاج task و hour بين 0 و 23.")
            _in_dir(s.daily_at, int(hour), int(minute), task, "research",
                    label or f"يومي {hour:02d}:{minute:02d}")
            return _ok({"added": label or f"يومي {hour:02d}:{minute:02d}",
                        "jobs": len(s._jobs)})
        if action == "presets":
            _in_dir(lambda: s.add_preset_morning().add_preset_okx(30))
            return _ok({"added": ["الروتين الصباحي", "OKX كل 30د"],
                        "jobs": len(s._jobs)})
        if action == "run_due":
            return _ok({"executed": _in_dir(s.run_due_now)})
        if action == "clear":
            n = len(s._jobs)
            s._jobs.clear()
            return _ok({"cleared": n})
        if action == "morning":
            return _ok({"brief": _in_dir(eng.morning.build, True, False)[:4000]})
        if action == "pipe_send":
            if not message:
                return tool_error("pipe_send يحتاج message.")
            mid = _in_dir(eng.pipe.send, {"text": message}, topic, "app")
            return _ok({"sent": mid, "topic": topic,
                        "pending": _in_dir(eng.pipe.pending, topic)})
        if action == "pipe_recv":
            m = _in_dir(eng.pipe.recv, topic, 0.0)
            return _ok({"message": m, "pending": _in_dir(eng.pipe.pending, topic)})
        return _ok({"status": _in_dir(s.status), "jobs": len(s._jobs),
                    "pipe_pending": _in_dir(eng.pipe.pending)})
    except Exception as exc:
        return tool_error(f"فشل الجدولة: {type(exc).__name__}: {exc}")


HERMES_SCHEDULE_SCHEMA = {
    "name": "hermes_schedule",
    "description": (
        "Recurring jobs, the daily brief, and the agent message pipe on the "
        "user's local stack. 'add_interval'/'add_daily' register a repeating "
        "task, 'presets' adds the morning routine plus a 30-minute OKX funding "
        "check, 'run_due' executes anything due right now, 'morning' builds the "
        "brief immediately (live OKX + funding + news + health), and "
        "'pipe_send'/'pipe_recv' pass durable messages between agents (they "
        "survive a restart). Default action lists current jobs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["list", "add_interval", "add_daily", "presets",
                                "run_due", "clear", "morning", "pipe_send",
                                "pipe_recv"]},
            "task": {"type": "string", "description": "Task text for add_*."},
            "minutes": {"type": "integer", "description": "Interval for add_interval."},
            "hour": {"type": "integer", "description": "Hour 0-23 for add_daily."},
            "minute": {"type": "integer", "description": "Minute for add_daily."},
            "label": {"type": "string", "description": "Optional job label."},
            "topic": {"type": "string", "description": "Pipe topic."},
            "message": {"type": "string", "description": "Pipe message text."},
        },
        "required": [],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3) hermes_ops — تشغيل: مراقبة، إصلاح، موديلات، نسخ، واجهة، ملفات، ضغط
# ═══════════════════════════════════════════════════════════════════════════════

def hermes_ops_tool(action: str = "dashboard", path: str = "",
                    text: str = "", model: str = "") -> str:
    eng, err = _eng()
    if err:
        return err
    try:
        if action == "dashboard":
            return _ok({"dashboard": _in_dir(eng.monitor.dashboard)})
        if action == "health_check":
            return _ok({"report": _in_dir(eng.monitor.tick)})
        if action == "health_history":
            return _ok({"history": _in_dir(eng.monitor.history, 10)})
        if action == "repair":
            rep = _in_dir(eng.monitor.snapshot)
            fixed = _in_dir(eng.monitor.repair, rep)
            return _ok({"repaired": fixed or ["لا شيء يحتاج إصلاحاً"],
                        "after": _in_dir(eng.monitor.snapshot)})
        if action == "models":
            return _ok({"report": _in_dir(eng.model_selector.report),
                        "benchmarks": _in_dir(eng.model_selector.benchmarks),
                        "selected": eng.local_model})
        if action == "set_model":
            if not model:
                return tool_error("set_model يحتاج model.")
            installed = _in_dir(eng.ollama_sup.list_models)
            if model not in installed:
                return tool_error(f"غير مثبّت. المتاح: {installed}")
            if not eng.model_selector.is_chat_model(model):
                return tool_error(f"{model} نموذج تضمين — لا يولّد نصاً.")
            pf = _in_dir(eng.ollama_sup.preflight, model, 4096)
            if not pf["ok"]:
                return tool_error(f"الذاكرة لا تكفي: {pf['reason']}")
            eng.local_model = model
            return _ok({"model": model, "preflight": pf})
        if action == "restart_ollama":
            return _ok({"up": _in_dir(eng.ollama_sup.ensure_up),
                        "restarts": eng.ollama_sup.restarts})
        if action == "free_memory":
            freed = _in_dir(eng.monitor.unload_models, eng.ollama_sup)
            return _ok({"unloaded": freed,
                        "free_ram_gb": round(eng.ollama_sup.free_ram_gb(), 1)})
        if action == "backup":
            import sqlite3
            import time as _t
            dest = path or f"hermes_backup_{_t.strftime('%Y%m%d_%H%M%S')}.db"

            def _do():
                src = sqlite3.connect(eng.memory.db_path)
                dst = sqlite3.connect(dest)
                with dst:
                    src.backup(dst)
                src.close()
                dst.close()
                import os as _o
                return {"file": _o.path.abspath(dest),
                        "bytes": _o.path.getsize(dest)}
            return _ok(_in_dir(_do))
        if action == "ui_start":
            return _ok({"url": _in_dir(eng.webui.start, False),
                        "note": "5 أوضاع + محادثة محفوظة"})
        if action == "ui_stop":
            _in_dir(eng.webui.stop)
            return _ok({"stopped": True})
        if action == "watch_file":
            if not path:
                return tool_error("watch_file يحتاج path.")
            _in_dir(eng.watchdog.watch, path, None, path)
            return _ok({"watching": path,
                        "changed_now": _in_dir(eng.watchdog.scan_once)})
        if action == "compress":
            if not text:
                return tool_error("compress يحتاج text.")
            c = eng.compressor
            msgs = [{"role": "user", "content": text}]
            before = c.count_messages(msgs)
            _, out = c.compress("", msgs)
            return _ok({"tokens_before": before,
                        "tokens_after": c.count_messages(out),
                        "messages": len(out)})
        if action == "autostart":
            return _ok({"status": eng.autostart.status()})
        return tool_error(f"action غير معروف: {action}")
    except Exception as exc:
        return tool_error(f"فشل التشغيل: {type(exc).__name__}: {exc}")


HERMES_OPS_SCHEMA = {
    "name": "hermes_ops",
    "description": (
        "Operate and repair the user's local stack. 'dashboard' and "
        "'health_check' report server, memory and model health; 'repair' "
        "restarts a dead Ollama and frees memory automatically; 'models' shows "
        "measured tokens/sec per model and 'set_model' switches (refusing "
        "embedding models and any that memory cannot fit); 'free_memory' unloads "
        "loaded models; 'backup' snapshots the memory database safely; "
        "'ui_start'/'ui_stop' control the local browser interface; 'watch_file' "
        "monitors a path; 'compress' shrinks long context. Call 'repair' first "
        "whenever another local tool failed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["dashboard", "health_check", "health_history",
                                "repair", "models", "set_model",
                                "restart_ollama", "free_memory", "backup",
                                "ui_start", "ui_stop", "watch_file",
                                "compress", "autostart"]},
            "path": {"type": "string", "description": "File path for watch_file/backup."},
            "text": {"type": "string", "description": "Text for compress."},
            "model": {"type": "string", "description": "Model name for set_model."},
        },
        "required": ["action"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4) hermes_codex — Codex أوف‑لاين
# ═══════════════════════════════════════════════════════════════════════════════

_CODEX_PROXY = {"obj": None}


def hermes_codex_tool(action: str = "status") -> str:
    eng, err = _eng()
    if err:
        return err
    try:
        if action == "status":
            return _ok({"codex": eng.codex.status(),
                        "docs": _in_dir(eng.codex_docs.stats),
                        "proxy_running": _CODEX_PROXY["obj"] is not None})
        if action == "index_docs":
            return _ok({"indexed": _in_dir(eng.codex_docs.index_into, eng)})
        if action == "proxy_start":
            if _CODEX_PROXY["obj"] is not None:
                return _ok({"already_running": True})
            p = eng.codex_proxy()
            url = p.start()
            _CODEX_PROXY["obj"] = p
            return _ok({"url": url, "target": p.target,
                        "note": "وجّه Codex إلى هذا العنوان"})
        if action == "proxy_stop":
            p = _CODEX_PROXY["obj"]
            if p is None:
                return _ok({"was_running": False})
            translations = p.translations
            p.stop()
            _CODEX_PROXY["obj"] = None
            return _ok({"stopped": True, "translations": translations})
        return tool_error(f"action غير معروف: {action}")
    except Exception as exc:
        return tool_error(f"فشل Codex: {type(exc).__name__}: {exc}")


HERMES_CODEX_SCHEMA = {
    "name": "hermes_codex",
    "description": (
        "Codex CLI offline support on the user's machine. 'status' reports the "
        "installed Codex version, its config and the local documentation corpus; "
        "'index_docs' loads that corpus into the local semantic index so "
        "hermes_knowledge can answer Codex questions offline; "
        "'proxy_start'/'proxy_stop' run a shim that adds the 'models' key Codex "
        "expects but Ollama omits, letting Codex talk to the local model with no "
        "internet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["status", "index_docs", "proxy_start",
                                "proxy_stop"]},
        },
        "required": [],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5) hermes_web — بحث متعدّد المصادر + بحث وقراءة وتلخيص
# ═══════════════════════════════════════════════════════════════════════════════

def hermes_web_tool(query: str, action: str = "search", n: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return tool_error("query مطلوب.")
    eng, err = _eng()
    if err:
        return err
    try:
        k = max(1, min(int(n or 5), 10))
        if action == "research":
            return _ok({"answer": _in_dir(eng.research, query, min(k, 3))})
        if action == "fetch":
            return _ok({"page": _in_dir(eng.web.fetch, query, 5000)})
        hits = _in_dir(eng.web.search, query, k)
        return _ok({"count": len(hits),
                    "source": (hits[0].get("source") if hits else None),
                    "results": hits})
    except Exception as exc:
        return tool_error(f"فشل البحث: {type(exc).__name__}: {exc}")


HERMES_WEB_SCHEMA = {
    "name": "hermes_web",
    "description": (
        "Web search that survives a blocked provider. Falls through Tavily → "
        "DuckDuckGo (with retry) → DuckDuckGo API → Wikipedia (Arabic and "
        "English), so it still returns results when one source rate-limits — "
        "measured behaviour, not a guess. 'research' also fetches the top pages "
        "and summarises them with citations on the local model; 'fetch' pulls one "
        "URL as clean text. Prefer this over guessing when a single search "
        "backend has already failed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Search terms, or a URL when action=fetch."},
            "action": {"type": "string", "enum": ["search", "research", "fetch"]},
            "n": {"type": "integer", "description": "Results 1-10 (default 5)."},
        },
        "required": ["query"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# التسجيل
# ═══════════════════════════════════════════════════════════════════════════════

# المُكتشِف في tools/registry.py يقبل نداء registry.register(...) مباشراً في
# جسم الوحدة فقط (فحص AST لعناصر tree.body). تسجيل داخل حلقة يمرّ صامتاً —
# الوحدة تُستورد يدوياً بنجاح لكن لا تُكتشَف تلقائياً. لذلك نداءات صريحة:

registry.register(
    name="hermes_task",
    toolset="hermes_local",
    schema=HERMES_TASK_SCHEMA,
    handler=lambda a, **k: hermes_task_tool(
        a.get("task", ""), a.get("steps", ""), a.get("max_parallel", 3)),
    check_fn=check_hermes_local,
    description="Multi-step local task with parallelism and resume.",
    emoji="🧩",
)

registry.register(
    name="hermes_schedule",
    toolset="hermes_local",
    schema=HERMES_SCHEDULE_SCHEMA,
    handler=lambda a, **k: hermes_schedule_tool(
        a.get("action", "list"), a.get("task", ""), a.get("minutes", 0),
        a.get("hour", -1), a.get("minute", 0), a.get("label", ""),
        a.get("topic", "default"), a.get("message", "")),
    check_fn=check_hermes_local,
    description="Recurring jobs, daily brief and agent pipe.",
    emoji="⏰",
)

registry.register(
    name="hermes_ops",
    toolset="hermes_local",
    schema=HERMES_OPS_SCHEMA,
    handler=lambda a, **k: hermes_ops_tool(
        a.get("action", "dashboard"), a.get("path", ""),
        a.get("text", ""), a.get("model", "")),
    check_fn=check_hermes_local,
    description="Monitor, self-repair, models, backup, local UI.",
    emoji="🛠",
)

registry.register(
    name="hermes_codex",
    toolset="hermes_local",
    schema=HERMES_CODEX_SCHEMA,
    handler=lambda a, **k: hermes_codex_tool(a.get("action", "status")),
    check_fn=check_hermes_local,
    description="Offline Codex support: docs and compatibility proxy.",
    emoji="📘",
)

registry.register(
    name="hermes_web",
    toolset="hermes_local",
    schema=HERMES_WEB_SCHEMA,
    handler=lambda a, **k: hermes_web_tool(
        a.get("query", ""), a.get("action", "search"), a.get("n", 5)),
    check_fn=check_hermes_local,
    description="Multi-backend web search with graceful fallback.",
    emoji="🌍",
)


__all__ = ["hermes_task_tool", "hermes_schedule_tool", "hermes_ops_tool",
           "hermes_codex_tool", "hermes_web_tool"]
