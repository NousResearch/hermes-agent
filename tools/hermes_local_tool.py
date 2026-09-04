#!/usr/bin/env python3
"""HERMES Local — جسر إلى منظومة هيرمس المحلية على جهاز المستخدم.

يضيف إلى أي جلسة محادثة قدرات لا يوفّرها التطبيق أصلاً:

    hermes_local_ask   استدلال محلي عبر Ollama — أوف‑لاين، مجاني، خاص تماماً.
                       لا يغادر أي حرف الجهاز. مع كاش يجعل التكرار فورياً.
    hermes_knowledge   بحث دلالي (بالمعنى لا بالكلمة) في ملفاتك المفهرسة محلياً.
    hermes_market      أسعار ومعدلات تمويل حية من OKX (بيانات عامة، بلا مفاتيح).
    hermes_status      صحة المنظومة المحلية: الخادم، الموديل، الذاكرة الحرة.

المنظومة تعيش في ``HERMES_LOCAL_DIR`` (افتراضياً ~/HERMES). كل الأدوات
تتحلّل برشاقة: إن غابت المنظومة أو سقط Ollama تُرجع خطأً واضحاً بلا انهيار،
ولا تُسجَّل أصلاً إن لم يكن المجلد موجوداً.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from tools.registry import registry, tool_error

# ── مكان المنظومة المحلية ────────────────────────────────────────────────────


def _local_dir() -> Path:
    return Path(os.getenv("HERMES_LOCAL_DIR")
                or (Path.home() / "HERMES")).expanduser()


def _available() -> bool:
    """هل منظومة هيرمس المحلية مثبّتة؟ (فحص رخيص، بلا استيراد)"""
    d = _local_dir()
    return (d / "hermes_boot.py").is_file()


# ── إقلاع كسول مُخزَّن (الإقلاع يكلّف ثوانٍ — مرة واحدة لكل عملية) ──────────

_ENGINE = None
_ENGINE_ERR: str | None = None
_LOCK = threading.Lock()
# os.chdir عالمي على مستوى العملية، وهذا التطبيق متعدد الخيوط: نداءان
# متوازيان كانا يفسدان مجلد بعضهما وقد يتركان التطبيق كله في مجلد خاطئ.
# كل تبديل مجلد يمرّ عبر هذا القفل، ولا يُستخدم إلا حيث لا بديل عنه.
_CWD_LOCK = threading.RLock()


def _engine():
    """يُقلع منظومة هيرمس مرة واحدة ويعيدها. يرفع RuntimeError عند الفشل."""
    global _ENGINE, _ENGINE_ERR
    if _ENGINE is not None:
        return _ENGINE
    with _LOCK:
        if _ENGINE is not None:
            return _ENGINE
        if _ENGINE_ERR:
            raise RuntimeError(_ENGINE_ERR)
        d = _local_dir()
        if not (d / "hermes_boot.py").is_file():
            _ENGINE_ERR = f"منظومة هيرمس المحلية غير موجودة في {d}"
            raise RuntimeError(_ENGINE_ERR)
        added = False
        try:
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
                added = True
            # تثبيت المسارات مطلقة قبل الإقلاع: يزيل اعتماد قاعدة الذاكرة
            # و.env على مجلد العمل، فلا نحتاج chdir في أي نداء لاحق.
            os.environ.setdefault("DB_PATH", str(d / "hermes_memory.db"))
            os.environ.setdefault("HERMES_CODEX_DOCS", str(d / "codex_docs"))
            with _CWD_LOCK:
                cwd = os.getcwd()
                os.chdir(d)      # الإقلاع وحده يحتاجه (تحميل .env النسبي)
                try:
                    from hermes_boot import boot  # type: ignore
                    _ENGINE = boot()
                finally:
                    os.chdir(cwd)
        except Exception as exc:
            if added:
                try:
                    sys.path.remove(str(d))
                except ValueError:
                    pass
            _ENGINE_ERR = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(_ENGINE_ERR) from exc
        return _ENGINE


def _in_dir(fn, *a, **kw):
    """
    ينفّذ داخل مجلد هيرمس، محروساً بقفل.

    بعد تثبيت DB_PATH مطلقاً عند الإقلاع لم يعد أغلب الكود يحتاج مجلد العمل،
    لكن يبقى مسار احتياطي لأي جزء يقرأ مساراً نسبياً. القفل إلزامي: chdir
    يغيّر حالة العملية كلها، فبدونه ينهار أي نداءين متوازيين معاً.
    """
    with _CWD_LOCK:
        cwd = os.getcwd()
        try:
            os.chdir(_local_dir())
            return fn(*a, **kw)
        finally:
            try:
                os.chdir(cwd)
            except Exception:
                pass


def _ok(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ── 1) استدلال محلي أوف‑لاين ─────────────────────────────────────────────────

def hermes_local_ask_tool(prompt: str, domain: str = "",
                          mode: str = "ask") -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return tool_error("prompt مطلوب.")
    try:
        eng = _engine()
    except RuntimeError as exc:
        return tool_error(str(exc))
    try:
        if mode == "agent":
            r = _in_dir(eng.run_agent, prompt)
            return _ok({"answer": r.get("answer", ""),
                        "tools_used": [t.get("tool") for t in r.get("trace", [])],
                        "steps": r.get("steps", 0), "engine": "local/ollama"})
        if mode == "specialist":
            r = _in_dir(eng.supervisor.run, prompt, domain or None)
            return _ok({"answer": r.get("answer", ""),
                        "domain": r.get("domain"),
                        "seconds": r.get("seconds"), "engine": "local/ollama"})
        dom = domain or eng._auto_detect_domain(prompt)
        answer = _in_dir(eng.turbo_call, prompt, dom)
        return _ok({"answer": answer, "domain": dom, "engine": "local/ollama",
                    "model": getattr(eng, "local_model", "?")})
    except Exception as exc:
        return tool_error(f"فشل الاستدلال المحلي: {type(exc).__name__}: {exc}")


HERMES_LOCAL_ASK_SCHEMA = {
    "name": "hermes_local_ask",
    "description": (
        "Ask the user's LOCAL Hermes stack (Ollama on this machine). Runs fully "
        "offline: nothing leaves the device, no API cost, and repeated questions "
        "return instantly from cache. Use it when the user asks for a private/"
        "offline/local answer, when you must not send data to a cloud model, or "
        "when the user explicitly says 'اسأل هيرمس المحلي'. Note it runs on CPU "
        "(~14 tokens/s) so keep prompts short; prefer your own reasoning for long "
        "or accuracy-critical work. mode='agent' lets the local model use its own "
        "tools (calculator, web search, knowledge base); mode='specialist' routes "
        "to one of 9 domain experts (trading, engineering, programming, …)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string",
                       "description": "The question or task, in Arabic or English."},
            "mode": {"type": "string", "enum": ["ask", "agent", "specialist"],
                     "description": "ask = direct answer (default); agent = the "
                                    "local model may call its own tools; "
                                    "specialist = route to a domain expert."},
            "domain": {"type": "string",
                       "description": "Optional domain hint: trading, engineering, "
                                      "programming, research, marketing, design, "
                                      "automation, bot, freelancing."},
        },
        "required": ["prompt"],
    },
}


# ── 2) المعرفة الشخصية بالمعنى ───────────────────────────────────────────────

def hermes_knowledge_tool(query: str, top_k: int = 5,
                          domain: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return tool_error("query مطلوب.")
    try:
        eng = _engine()
    except RuntimeError as exc:
        return tool_error(str(exc))
    try:
        k = max(1, min(int(top_k or 5), 20))
        sem = getattr(eng, "semantic_rag", None)
        hits, how = [], "keyword"
        if sem is not None and _in_dir(sem.available):
            hits = _in_dir(sem.query, query, domain or None, k)
            how = "semantic"
        if not hits:
            hits = _in_dir(eng.rag.query, query, domain or None, k)
            how = "keyword"
        if not hits:
            return _ok({"found": 0, "method": how, "results": [],
                        "hint": "لا نتائج. فهرِس ملفات أولاً: "
                                "hermes_boot.py rag <مجلد>"})
        return _ok({"found": len(hits), "method": how,
                    "results": [h[:900] for h in hits]})
    except Exception as exc:
        return tool_error(f"فشل البحث في المعرفة: {type(exc).__name__}: {exc}")


HERMES_KNOWLEDGE_SCHEMA = {
    "name": "hermes_knowledge",
    "description": (
        "Search the user's OWN indexed documents on this machine — their personal "
        "knowledge base, not the internet. Uses semantic (meaning-based) matching "
        "via local embeddings, so it finds relevant passages even when they share "
        "no keywords with the query; falls back to full-text search. Use it before "
        "answering anything about the user's own files, notes, specs, or projects. "
        "Returns nothing if the user has not indexed any files yet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "top_k": {"type": "integer",
                      "description": "How many passages to return (1-20, default 5)."},
            "domain": {"type": "string",
                       "description": "Optional domain filter used at index time."},
        },
        "required": ["query"],
    },
}


# ── 3) سوق OKX الحي ──────────────────────────────────────────────────────────

def hermes_market_tool(symbols: str = "", what: str = "prices") -> str:
    try:
        eng = _engine()
    except RuntimeError as exc:
        return tool_error(str(exc))
    try:
        syms = [s.strip() for s in (symbols or "").split(",") if s.strip()] or None
        if what == "funding":
            return _ok({"funding": _in_dir(eng.okx.get_funding_rates, syms)})
        if what == "arbitrage":
            return _ok({"opportunities": _in_dir(eng.okx.scan_arbitrage, 0.01)})
        if what == "report":
            return _ok({"report": _in_dir(eng.okx.market_report)})
        return _ok({"prices": _in_dir(eng.okx.get_prices, syms)})
    except Exception as exc:
        return tool_error(f"فشل جلب بيانات السوق: {type(exc).__name__}: {exc}")


HERMES_MARKET_SCHEMA = {
    "name": "hermes_market",
    "description": (
        "Live crypto market data from OKX — spot/swap prices, 24h change, funding "
        "rates, and funding-arbitrage opportunities. Public data, no API keys "
        "needed. Use it whenever the user asks about current crypto prices or "
        "funding rather than guessing or using stale training data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "what": {"type": "string",
                     "enum": ["prices", "funding", "arbitrage", "report"],
                     "description": "prices (default), funding, arbitrage, or a "
                                    "combined human-readable report."},
            "symbols": {"type": "string",
                        "description": "Comma-separated OKX instrument ids, e.g. "
                                       "'BTC-USDT-SWAP,ETH-USDT-SWAP'. Defaults to "
                                       "BTC, ETH and SOL."},
        },
        "required": [],
    },
}


# ── 4) صحة المنظومة المحلية ──────────────────────────────────────────────────

def hermes_status_tool() -> str:
    d = _local_dir()
    if not _available():
        return _ok({"installed": False, "dir": str(d),
                    "hint": "ثبّت منظومة هيرمس المحلية أو اضبط HERMES_LOCAL_DIR"})
    try:
        eng = _engine()
    except RuntimeError as exc:
        return _ok({"installed": True, "dir": str(d), "booted": False,
                    "error": str(exc)})
    try:
        sup = eng.ollama_sup
        return _ok({
            "installed": True, "booted": True, "dir": str(d),
            "ollama_up": sup.is_up(),
            "model": getattr(eng, "local_model", "?"),
            "models": sup.list_models(),
            "free_ram_gb": round(sup.free_ram_gb(), 1),
            "indexed_files": len(_in_dir(eng.rag.list_indexed)),
            "semantic_vectors": _in_dir(eng.semantic_rag.count),
            "cache": _in_dir(lambda: eng.hyperdrive.cache.stats()),
        })
    except Exception as exc:
        return tool_error(f"فشل قراءة الحالة: {type(exc).__name__}: {exc}")


HERMES_STATUS_SCHEMA = {
    "name": "hermes_status",
    "description": (
        "Health of the user's local Hermes stack: is the Ollama server up, which "
        "model is selected, free RAM, how many personal files are indexed, and "
        "cache statistics. Call it before relying on hermes_local_ask or "
        "hermes_knowledge if a previous local call failed."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ── التسجيل ──────────────────────────────────────────────────────────────────

def check_hermes_local() -> tuple:
    """يخبر التطبيق هل الأدوات صالحة للاستخدام في هذه البيئة."""
    if not _available():
        return False, (f"منظومة هيرمس المحلية غير موجودة في {_local_dir()} "
                       "(اضبط HERMES_LOCAL_DIR)")
    return True, ""


registry.register(
    name="hermes_local_ask",
    toolset="hermes_local",
    schema=HERMES_LOCAL_ASK_SCHEMA,
    handler=lambda args, **kw: hermes_local_ask_tool(
        prompt=args.get("prompt", ""),
        domain=args.get("domain", ""),
        mode=args.get("mode", "ask")),
    check_fn=check_hermes_local,
    description="Offline local inference on the user's own machine (Ollama).",
    emoji="🏠",
)

registry.register(
    name="hermes_knowledge",
    toolset="hermes_local",
    schema=HERMES_KNOWLEDGE_SCHEMA,
    handler=lambda args, **kw: hermes_knowledge_tool(
        query=args.get("query", ""),
        top_k=args.get("top_k", 5),
        domain=args.get("domain", "")),
    check_fn=check_hermes_local,
    description="Semantic search over the user's own indexed documents.",
    emoji="📚",
)

registry.register(
    name="hermes_market",
    toolset="hermes_local",
    schema=HERMES_MARKET_SCHEMA,
    handler=lambda args, **kw: hermes_market_tool(
        symbols=args.get("symbols", ""),
        what=args.get("what", "prices")),
    check_fn=check_hermes_local,
    description="Live OKX crypto prices and funding rates.",
    emoji="📈",
)

registry.register(
    name="hermes_status",
    toolset="hermes_local",
    schema=HERMES_STATUS_SCHEMA,
    handler=lambda args, **kw: hermes_status_tool(),
    check_fn=check_hermes_local,
    description="Health of the local Hermes stack.",
    emoji="🩺",
)


__all__ = [
    "hermes_local_ask_tool",
    "hermes_knowledge_tool",
    "hermes_market_tool",
    "hermes_status_tool",
    "check_hermes_local",
]
