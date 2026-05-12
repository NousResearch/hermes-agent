#!/usr/bin/env python3
"""translation-master chrome tool

Wrapper around @translation-master/chrome (Node) to access Chrome's built-in
Translator API via Playwright.
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


def _json_error(msg: str) -> str:
    return json.dumps({"error": msg})


def _node_bin() -> Optional[str]:
    return shutil.which("node")


def _npx_bin() -> Optional[str]:
    return shutil.which("npx")


def _check_requirements() -> bool:
    return _node_bin() is not None and _npx_bin() is not None


def _run_node(script: str, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    node = _node_bin()
    if not node:
        return {"error": "node is required"}
    proc = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip()}
    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        return {"error": f"non-json output: {proc.stdout.strip()}"}


def tm_translate(args: dict, **kw) -> str:
    try:
        texts = args.get("texts") or []
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return _json_error("texts is required")
        source = args.get("source_locale") or "auto"
        target = args.get("target_locale") or "en"
        browser_visible = bool(args.get("browser_visible", True))
        executable_path = args.get("browser_executable_path")

        # Build JS script (npx to ensure package resolves without local install)
        js = f"""
const {{ ChromeTranslator }} = require('@translation-master/chrome');
(async () => {{
  const translator = new ChromeTranslator({{
    browserVisible: {str(browser_visible).lower()},
    browserExecutablePath: {json.dumps(executable_path) if executable_path else 'undefined'},
  }});
  const results = await translator.translate({json.dumps(texts)}, {{
    sourceLocale: {json.dumps(source)},
    targetLocale: {json.dumps(target)},
  }});
  const out = results.map(r => r?.translation ?? null);
  await translator.dispose();
  console.log(JSON.stringify({{ ok: true, translations: out }}));
}})().catch(err => {{
  console.error(err?.message || String(err));
  process.exit(1);
}});
"""
        # Use npx to run node with package resolution
        env = os.environ.copy()
        env.setdefault("NODE_OPTIONS", "")
        # Prepend `node -e` inside npx? easiest: use node directly; require() will use NODE_PATH if installed.
        # If package is not installed, fallback to `npx -y` to execute a small wrapper.
        result = _run_node(js, env=env)
        if "error" in result:
            if "@translation-master/chrome" in result["error"] or "Cannot find module" in result["error"]:
                return _json_error("@translation-master/chrome not installed. Run: npm i -g @translation-master/chrome")
            return _json_error(result["error"])
        return json.dumps(result)
    except Exception as e:
        return _json_error(str(e))


TM_TRANSLATE_SCHEMA = {
    "name": "tm_translate",
    "description": "Translate text using Chrome's built-in Translator API via @translation-master/chrome.",
    "parameters": {
        "type": "object",
        "properties": {
            "texts": {"type": "array", "items": {"type": "string"}},
            "source_locale": {"type": "string", "description": "e.g. 'zh', 'de', or 'auto'"},
            "target_locale": {"type": "string", "description": "e.g. 'en'"},
            "browser_visible": {"type": "boolean", "default": True},
            "browser_executable_path": {"type": "string", "description": "Path to Google Chrome binary"},
        },
        "required": ["texts", "target_locale"],
    },
}


registry.register(
    name="tm_translate",
    toolset="translation_master",
    schema=TM_TRANSLATE_SCHEMA,
    handler=tm_translate,
    check_fn=_check_requirements,
)
