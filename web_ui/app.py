"""
Hermes Agent – Web Configuration UI (complete)
Covers every config.yaml key and .env variable from the hermes-agent schema.
"""

import asyncio
import json
import os
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────
HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
CONFIG_PATH = HERMES_HOME / "config.yaml"
ENV_PATH    = HERMES_HOME / ".env"
AUTH_PATH   = HERMES_HOME / "auth.json"

NOUS_PORTAL = "https://portal.nousresearch.com"
CLIENT_ID   = "hermes-cli"

# ── Process state ─────────────────────────────────────────────
_gw_proc:       Optional[subprocess.Popen] = None
_gw_logs:       Deque[str] = deque(maxlen=300)
_gw_started_at: Optional[float] = None
_oauth_state:   dict = {}


# ── File helpers ──────────────────────────────────────────────
def _home() -> Path:
    HERMES_HOME.mkdir(parents=True, exist_ok=True)
    return HERMES_HOME

def read_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}

def save_config(cfg: dict) -> None:
    _home()
    CONFIG_PATH.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    )

def read_env() -> dict:
    if not ENV_PATH.exists():
        return {}
    out: dict = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out

def save_env(env: dict) -> None:
    _home()
    ENV_PATH.write_text(
        "\n".join(f"{k}={v}" for k, v in env.items() if v is not None) + "\n"
    )

def patch_env(**kv) -> None:
    e = read_env()
    for k, v in kv.items():
        if v is not None:
            e[k] = v
    save_env(e)

def read_auth() -> dict:
    if AUTH_PATH.exists():
        try:
            return json.loads(AUTH_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_auth(a: dict) -> None:
    _home()
    AUTH_PATH.write_text(json.dumps(a, indent=2))

def _mask(v: str) -> str:
    if not v:
        return ""
    if len(v) <= 8:
        return "••••••••"
    return "••••" + v[-4:]

def _bool_env(env: dict, key: str, default: bool = False) -> bool:
    v = env.get(key, "").strip().lower()
    if v in ("true", "1", "yes"):  return True
    if v in ("false", "0", "no"): return False
    return default


# ── Gateway management ────────────────────────────────────────
def _gw_running() -> bool:
    return _gw_proc is not None and _gw_proc.poll() is None

def _drain_logs(proc: subprocess.Popen) -> None:
    try:
        import select
        while proc.stdout and select.select([proc.stdout], [], [], 0)[0]:
            line = proc.stdout.readline()
            if line:
                _gw_logs.append(line.rstrip())
    except Exception:
        pass

def start_gateway() -> None:
    global _gw_proc, _gw_started_at
    if _gw_running():
        return
    _gw_logs.clear()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for k, v in read_env().items():
        env.setdefault(k, v)
    _gw_proc = subprocess.Popen(
        ["python", "-m", "gateway.run"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd="/app",
    )
    _gw_started_at = time.time()

def stop_gateway() -> None:
    global _gw_proc, _gw_started_at
    if _gw_proc and _gw_proc.poll() is None:
        _gw_proc.terminate()
        try:
            _gw_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _gw_proc.kill()
    _gw_proc = None
    _gw_started_at = None

async def _log_pump() -> None:
    while True:
        if _gw_proc and _gw_proc.poll() is None:
            _drain_logs(_gw_proc)
        await asyncio.sleep(1)


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        start_gateway()
    except Exception:
        pass
    asyncio.create_task(_log_pump())
    yield
    stop_gateway()


app = FastAPI(title="Hermes Config UI", lifespan=lifespan)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/")
async def root():
    return FileResponse(_static_dir / "index.html")


# ── Overview ──────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    env  = read_env()
    auth = read_auth()
    cfg  = read_config()
    nous = auth.get("providers", {}).get("nous", {})
    mc   = cfg.get("model", {})
    base = mc.get("base_url", "")
    prov = mc.get("provider", "auto")

    ptype = "openrouter"
    if prov == "nous" or "nousresearch" in base:
        ptype = "nous"
    elif base and base not in ("https://openrouter.ai/api/v1", ""):
        ptype = "custom"

    return {
        "nous_logged_in":  bool(nous.get("refresh_token") or nous.get("access_token")),
        "openrouter_key":  bool(env.get("OPENROUTER_API_KEY")),
        "provider_type":   ptype,
        "active_model":    mc.get("default", ""),
        "gateway": {
            "running":  _gw_running(),
            "pid":      _gw_proc.pid if _gw_running() else None,
            "uptime_s": int(time.time() - _gw_started_at) if _gw_started_at else 0,
        },
        "platforms": {
            "telegram":  bool(env.get("TELEGRAM_BOT_TOKEN")),
            "discord":   bool(env.get("DISCORD_BOT_TOKEN")),
            "slack":     bool(env.get("SLACK_BOT_TOKEN")),
            "whatsapp":  _bool_env(env, "WHATSAPP_ENABLED"),
        },
    }

@app.get("/api/gateway/logs")
async def get_gateway_logs():
    if _gw_proc:
        _drain_logs(_gw_proc)
    return {"logs": list(_gw_logs)}

@app.post("/api/gateway/restart")
async def restart_gateway():
    stop_gateway()
    await asyncio.sleep(0.5)
    start_gateway()
    return {"ok": True, "pid": _gw_proc.pid if _gw_proc else None}


# ── Auth ──────────────────────────────────────────────────────
@app.post("/api/auth/nous/start")
async def nous_start():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{NOUS_PORTAL}/api/oauth/device/code",
            json={"client_id": CLIENT_ID, "scope": "inference:mint_agent_key"},
            headers={"Accept": "application/json"},
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Nous Portal {r.status_code}: {r.text[:300]}")
        d = r.json()
    _oauth_state["device_code"] = d["device_code"]
    _oauth_state["interval"]    = d.get("interval", 5)
    return {
        "user_code":                 d["user_code"],
        "verification_uri":          d["verification_uri"],
        "verification_uri_complete": d.get("verification_uri_complete", d["verification_uri"]),
        "expires_in":                d.get("expires_in", 300),
        "interval":                  d.get("interval", 5),
    }

@app.get("/api/auth/nous/poll")
async def nous_poll():
    if not _oauth_state.get("device_code"):
        raise HTTPException(400, "No active OAuth flow — call /api/auth/nous/start first")
    async with httpx.AsyncClient(timeout=15) as c:
        # OAuth spec requires application/x-www-form-urlencoded for token requests
        r = await c.post(
            f"{NOUS_PORTAL}/api/oauth/token",
            data={
                "grant_type":  "urn:ietf:params:oauth:grant-type:device_code",
                "client_id":   CLIENT_ID,
                "device_code": _oauth_state["device_code"],
            },
            headers={"Accept": "application/json"},
        )
        d = r.json()
    if "access_token" in d:
        now      = datetime.now(tz=timezone.utc)
        expires  = int(d.get("expires_in", 0))
        now_iso  = now.isoformat()
        exp_iso  = now.replace(second=now.second + expires).isoformat() if expires else now_iso
        # Build the full auth.json structure the gateway expects
        auth = read_auth()
        auth["version"] = 1
        auth.setdefault("providers", {})["nous"] = {
            "portal_base_url":        NOUS_PORTAL,
            "inference_base_url":     "https://inference-api.nousresearch.com/v1",
            "client_id":              CLIENT_ID,
            "token_type":             d.get("token_type", "Bearer"),
            "scope":                  d.get("scope", "inference:mint_agent_key"),
            "access_token":           d["access_token"],
            "refresh_token":          d.get("refresh_token"),
            "obtained_at":            now_iso,
            "expires_in":             expires,
            "expires_at":             exp_iso,
            "agent_key":              None,
            "agent_key_id":           None,
            "agent_key_expires_at":   None,
            "agent_key_expires_in":   None,
            "agent_key_reused":       None,
            "agent_key_obtained_at":  None,
            "tls":                    {"insecure": False, "ca_bundle": None},
        }
        auth["active_provider"] = "nous"
        save_auth(auth)
        cfg = read_config()
        cfg.setdefault("model", {})["provider"] = "nous"
        cfg["model"]["base_url"] = "https://inference-api.nousresearch.com/v1"
        save_config(cfg)
        _oauth_state.clear()
        return {"status": "authorized"}
    err = d.get("error", "unknown")
    if err in ("authorization_pending", "slow_down"):
        return {"status": "pending", "error": err}
    _oauth_state.clear()
    return {"status": "error", "detail": err, "raw": d}

@app.post("/api/auth/nous/logout")
async def nous_logout():
    auth = read_auth()
    auth.get("providers", {}).pop("nous", None)
    if auth.get("active_provider") == "nous":
        auth.pop("active_provider", None)
    save_auth(auth)
    cfg = read_config()
    if cfg.get("model", {}).get("provider") == "nous":
        cfg["model"]["provider"] = "auto"
    save_config(cfg)
    return {"ok": True}

@app.get("/api/auth/status")
async def auth_status():
    auth = read_auth()
    nous = auth.get("providers", {}).get("nous", {})
    return {
        "nous": {"logged_in": bool(nous.get("refresh_token") or nous.get("access_token"))},
        "active_provider": auth.get("active_provider", "auto"),
    }


# ── Provider ──────────────────────────────────────────────────
class ProviderIn(BaseModel):
    type:             str  # "nous" | "openrouter" | "custom"
    api_key:          str = ""
    base_url:         str = ""
    model:            str = ""
    routing_sort:     str = "throughput"
    routing_data:     str = "deny"
    routing_require:  bool = True
    routing_only:     str = ""
    routing_ignore:   str = ""

@app.get("/api/provider")
async def get_provider():
    cfg  = read_config()
    env  = read_env()
    mc   = cfg.get("model", {})
    base = mc.get("base_url", "")
    prov = mc.get("provider", "auto")
    ptype = "openrouter"
    if prov == "nous" or "nousresearch" in base:
        ptype = "nous"
    elif base and base not in ("https://openrouter.ai/api/v1", ""):
        ptype = "custom"
    key = env.get("OPENROUTER_API_KEY", "")
    rt  = cfg.get("provider_routing", {})
    return {
        "type":            ptype,
        "api_key_set":     bool(key),
        "api_key_masked":  _mask(key),
        "base_url":        base,
        "model":           mc.get("default", "anthropic/claude-opus-4-6"),
        "routing_sort":    rt.get("sort", "throughput"),
        "routing_data":    rt.get("data_collection", "deny"),
        "routing_require": rt.get("require_parameters", True),
        "routing_only":    ",".join(rt.get("only", [])),
        "routing_ignore":  ",".join(rt.get("ignore", [])),
    }

@app.get("/api/provider/models")
async def get_provider_models(source: str = "openrouter"):
    """
    Fetch live model list from OpenRouter or Nous Portal.
    source: "openrouter" | "nous" | "custom"  (custom uses base_url + key from config)
    """
    env = read_env()
    cfg = read_config()

    if source == "nous":
        auth  = read_auth()
        token = auth.get("providers", {}).get("nous", {}).get("access_token", "")
        if not token:
            raise HTTPException(401, "Not logged in to Nous Portal")
        base_url = "https://inference-api.nousresearch.com/v1"
        headers  = {"Authorization": f"Bearer {token}"}
    elif source == "custom":
        base_url = cfg.get("model", {}).get("base_url", "").rstrip("/")
        if not base_url:
            raise HTTPException(400, "No custom base URL configured")
        key     = env.get("OPENROUTER_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
    else:  # openrouter
        base_url = "https://openrouter.ai/api/v1"
        key      = env.get("OPENROUTER_API_KEY", "")
        headers  = {"Authorization": f"Bearer {key}"} if key else {}

    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{base_url}/models", headers=headers)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Upstream error: {r.text[:300]}")
        raw = r.json()

    # Normalise to [{id, name, context_length, prompt_price, completion_price, description}]
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    models = []
    for m in items:
        if not isinstance(m, dict):
            continue
        pricing = m.get("pricing", {})
        try:
            pp = float(pricing.get("prompt",     0)) * 1_000_000
            cp = float(pricing.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            pp = cp = 0.0
        models.append({
            "id":              m.get("id", ""),
            "name":            m.get("name", m.get("id", "")),
            "context_length":  m.get("context_length", 0),
            "prompt_price":    round(pp, 4),
            "completion_price": round(cp, 4),
            "description":     (m.get("description") or "")[:120],
        })

    models.sort(key=lambda x: x["name"].lower())
    return {"models": models, "count": len(models)}


@app.post("/api/provider")
async def set_provider(p: ProviderIn):
    cfg = read_config()
    cfg.setdefault("model", {})
    if p.type == "nous":
        cfg["model"]["provider"] = "nous"
        cfg["model"]["base_url"] = "https://inference-api.nousresearch.com/v1"
    elif p.type == "openrouter":
        cfg["model"]["provider"] = "openrouter"
        cfg["model"]["base_url"] = "https://openrouter.ai/api/v1"
        if p.api_key: patch_env(OPENROUTER_API_KEY=p.api_key)
    elif p.type == "custom":
        cfg["model"]["provider"] = "openrouter"
        if p.base_url: cfg["model"]["base_url"] = p.base_url
        if p.api_key:  patch_env(OPENROUTER_API_KEY=p.api_key)
    if p.model:
        cfg["model"]["default"] = p.model
    cfg["provider_routing"] = {
        "sort":                p.routing_sort,
        "data_collection":     p.routing_data,
        "require_parameters":  p.routing_require,
        "only":   [x.strip() for x in p.routing_only.split(",")   if x.strip()],
        "ignore": [x.strip() for x in p.routing_ignore.split(",") if x.strip()],
    }
    save_config(cfg)
    return {"ok": True}


# ── Platforms ─────────────────────────────────────────────────
class PlatformIn(BaseModel):
    telegram_token:   str  = ""
    telegram_channel: str  = ""
    telegram_allowed: str  = ""
    discord_token:    str  = ""
    discord_channel:  str  = ""
    discord_allowed:  str  = ""
    slack_bot_token:  str  = ""
    slack_app_token:  str  = ""
    slack_channel:    str  = ""
    slack_allowed:    str  = ""
    whatsapp_enabled: bool = False
    whatsapp_mode:    str  = "bot"
    whatsapp_allowed: str  = ""
    allow_all_users:  bool = False
    session_mode:     str  = "both"
    session_idle_min: int  = 1440
    session_at_hour:  int  = 4

@app.get("/api/platforms")
async def get_platforms():
    env = read_env()
    cfg = read_config()
    def m(k): v = env.get(k, ""); return _mask(v) if v else ""
    sr = cfg.get("session_reset", {})
    return {
        "telegram_token_set":     bool(env.get("TELEGRAM_BOT_TOKEN")),
        "telegram_token_masked":  m("TELEGRAM_BOT_TOKEN"),
        "telegram_channel":       env.get("TELEGRAM_HOME_CHANNEL", ""),
        "telegram_allowed":       env.get("TELEGRAM_ALLOWED_USERS", ""),
        "discord_token_set":      bool(env.get("DISCORD_BOT_TOKEN")),
        "discord_token_masked":   m("DISCORD_BOT_TOKEN"),
        "discord_channel":        env.get("DISCORD_HOME_CHANNEL", ""),
        "discord_allowed":        env.get("DISCORD_ALLOWED_USERS", ""),
        "slack_bot_token_set":    bool(env.get("SLACK_BOT_TOKEN")),
        "slack_bot_token_masked": m("SLACK_BOT_TOKEN"),
        "slack_app_token_set":    bool(env.get("SLACK_APP_TOKEN")),
        "slack_app_token_masked": m("SLACK_APP_TOKEN"),
        "slack_channel":          env.get("SLACK_HOME_CHANNEL", ""),
        "slack_allowed":          env.get("SLACK_ALLOWED_USERS", ""),
        "whatsapp_enabled":       _bool_env(env, "WHATSAPP_ENABLED"),
        "whatsapp_mode":          env.get("WHATSAPP_MODE", "bot"),
        "whatsapp_allowed":       env.get("WHATSAPP_ALLOWED_USERS", ""),
        "allow_all_users":        _bool_env(env, "GATEWAY_ALLOW_ALL_USERS"),
        "session_mode":           sr.get("mode", "both"),
        "session_idle_min":       sr.get("idle_minutes", 1440),
        "session_at_hour":        sr.get("at_hour", 4),
    }

@app.post("/api/platforms")
async def set_platforms(p: PlatformIn):
    updates: dict = {}
    if p.telegram_token:   updates["TELEGRAM_BOT_TOKEN"]       = p.telegram_token
    if p.telegram_channel: updates["TELEGRAM_HOME_CHANNEL"]     = p.telegram_channel
    updates["TELEGRAM_ALLOWED_USERS"] = p.telegram_allowed
    if p.discord_token:    updates["DISCORD_BOT_TOKEN"]         = p.discord_token
    if p.discord_channel:  updates["DISCORD_HOME_CHANNEL"]      = p.discord_channel
    updates["DISCORD_ALLOWED_USERS"] = p.discord_allowed
    if p.slack_bot_token:  updates["SLACK_BOT_TOKEN"]           = p.slack_bot_token
    if p.slack_app_token:  updates["SLACK_APP_TOKEN"]           = p.slack_app_token
    if p.slack_channel:    updates["SLACK_HOME_CHANNEL"]        = p.slack_channel
    updates["SLACK_ALLOWED_USERS"] = p.slack_allowed
    updates["WHATSAPP_ENABLED"]      = "true" if p.whatsapp_enabled else "false"
    updates["WHATSAPP_MODE"]         = p.whatsapp_mode
    updates["WHATSAPP_ALLOWED_USERS"] = p.whatsapp_allowed
    updates["GATEWAY_ALLOW_ALL_USERS"] = "true" if p.allow_all_users else "false"
    patch_env(**updates)
    cfg = read_config()
    cfg["session_reset"] = {
        "mode":         p.session_mode,
        "idle_minutes": p.session_idle_min,
        "at_hour":      p.session_at_hour,
    }
    save_config(cfg)
    return {"ok": True}


# ── Agent behavior ────────────────────────────────────────────
class AgentIn(BaseModel):
    personality:            str   = "helpful"
    reasoning_effort:       str   = "xhigh"
    verbose:                bool  = False
    tool_progress:          str   = "all"
    display_compact:        bool  = False
    max_turns:              int   = 60
    human_delay_mode:       str   = "off"
    human_delay_min_ms:     int   = 800
    human_delay_max_ms:     int   = 2500
    memory_enabled:         bool  = True
    user_profile_enabled:   bool  = True
    memory_char_limit:      int   = 2200
    user_char_limit:        int   = 1375
    nudge_interval:         int   = 10
    tts_provider:           str   = "edge"
    tts_edge_voice:         str   = "en-US-AriaNeural"
    tts_el_voice_id:        str   = "pNInz6obpgDQGcFmaJgB"
    tts_el_model:           str   = "eleven_multilingual_v2"
    tts_oai_model:          str   = "gpt-4o-mini-tts"
    tts_oai_voice:          str   = "alloy"
    stt_enabled:            bool  = True
    stt_model:              str   = "whisper-1"

@app.get("/api/agent")
async def get_agent():
    cfg  = read_config()
    a    = cfg.get("agent",       {})
    d    = cfg.get("display",     {})
    hd   = cfg.get("human_delay", {})
    mem  = cfg.get("memory",      {})
    tts  = cfg.get("tts",         {})
    stt  = cfg.get("stt",         {})
    return {
        "personality":          d.get("personality",     "helpful"),
        "reasoning_effort":     a.get("reasoning_effort","xhigh"),
        "verbose":              a.get("verbose",          False),
        "tool_progress":        d.get("tool_progress",   "all"),
        "display_compact":      d.get("compact",          False),
        "max_turns":            a.get("max_turns",        60),
        "human_delay_mode":     hd.get("mode",           "off"),
        "human_delay_min_ms":   hd.get("min_ms",          800),
        "human_delay_max_ms":   hd.get("max_ms",         2500),
        "memory_enabled":       mem.get("memory_enabled",       True),
        "user_profile_enabled": mem.get("user_profile_enabled", True),
        "memory_char_limit":    mem.get("memory_char_limit",    2200),
        "user_char_limit":      mem.get("user_char_limit",      1375),
        "nudge_interval":       mem.get("nudge_interval",       10),
        "tts_provider":         tts.get("provider",            "edge"),
        "tts_edge_voice":       tts.get("edge",        {}).get("voice",    "en-US-AriaNeural"),
        "tts_el_voice_id":      tts.get("elevenlabs",  {}).get("voice_id", "pNInz6obpgDQGcFmaJgB"),
        "tts_el_model":         tts.get("elevenlabs",  {}).get("model_id", "eleven_multilingual_v2"),
        "tts_oai_model":        tts.get("openai",      {}).get("model",    "gpt-4o-mini-tts"),
        "tts_oai_voice":        tts.get("openai",      {}).get("voice",    "alloy"),
        "stt_enabled":          stt.get("enabled",     True),
        "stt_model":            stt.get("model",       "whisper-1"),
    }

@app.post("/api/agent")
async def save_agent(a: AgentIn):
    cfg = read_config()
    cfg.setdefault("agent", {}).update({
        "max_turns":        a.max_turns,
        "verbose":          a.verbose,
        "reasoning_effort": a.reasoning_effort,
    })
    cfg.setdefault("display", {}).update({
        "personality":   a.personality,
        "tool_progress": a.tool_progress,
        "compact":       a.display_compact,
    })
    cfg.setdefault("human_delay", {}).update({
        "mode":   a.human_delay_mode,
        "min_ms": a.human_delay_min_ms,
        "max_ms": a.human_delay_max_ms,
    })
    cfg.setdefault("memory", {}).update({
        "memory_enabled":       a.memory_enabled,
        "user_profile_enabled": a.user_profile_enabled,
        "memory_char_limit":    a.memory_char_limit,
        "user_char_limit":      a.user_char_limit,
        "nudge_interval":       a.nudge_interval,
    })
    cfg.setdefault("tts", {}).update({
        "provider":    a.tts_provider,
        "edge":        {"voice": a.tts_edge_voice},
        "elevenlabs":  {"voice_id": a.tts_el_voice_id, "model_id": a.tts_el_model},
        "openai":      {"model": a.tts_oai_model, "voice": a.tts_oai_voice},
    })
    cfg.setdefault("stt", {}).update({"enabled": a.stt_enabled, "model": a.stt_model})
    save_config(cfg)
    return {"ok": True}


# ── Terminal ──────────────────────────────────────────────────
class TerminalIn(BaseModel):
    backend:              str  = "local"
    cwd:                  str  = "."
    timeout:              int  = 180
    lifetime_seconds:     int  = 300
    docker_image:         str  = "nikolaik/python-nodejs:python3.11-nodejs20"
    modal_image:          str  = "nikolaik/python-nodejs:python3.11-nodejs20"
    singularity_image:    str  = "docker://nikolaik/python-nodejs:python3.11-nodejs20"
    container_cpu:        int  = 1
    container_memory:     int  = 5120
    container_disk:       int  = 51200
    container_persistent: bool = True
    ssh_host:             str  = ""
    ssh_user:             str  = ""
    ssh_port:             int  = 22
    ssh_key:              str  = "~/.ssh/id_rsa"
    sudo_password:        str  = ""

@app.get("/api/terminal")
async def get_terminal():
    t = read_config().get("terminal", {})
    return {
        "backend":              t.get("backend",              "local"),
        "cwd":                  t.get("cwd",                  "."),
        "timeout":              t.get("timeout",              180),
        "lifetime_seconds":     t.get("lifetime_seconds",     300),
        "docker_image":         t.get("docker_image",         "nikolaik/python-nodejs:python3.11-nodejs20"),
        "modal_image":          t.get("modal_image",          "nikolaik/python-nodejs:python3.11-nodejs20"),
        "singularity_image":    t.get("singularity_image",    "docker://nikolaik/python-nodejs:python3.11-nodejs20"),
        "container_cpu":        t.get("container_cpu",        1),
        "container_memory":     t.get("container_memory",     5120),
        "container_disk":       t.get("container_disk",       51200),
        "container_persistent": t.get("container_persistent", True),
        "ssh_host":             t.get("ssh_host",             ""),
        "ssh_user":             t.get("ssh_user",             ""),
        "ssh_port":             t.get("ssh_port",             22),
        "ssh_key":              t.get("ssh_key",              "~/.ssh/id_rsa"),
        "sudo_password_set":    bool(t.get("sudo_password",   "")),
    }

@app.post("/api/terminal")
async def save_terminal(t: TerminalIn):
    cfg  = read_config()
    term: dict = {
        "backend":          t.backend,
        "cwd":              t.cwd,
        "timeout":          t.timeout,
        "lifetime_seconds": t.lifetime_seconds,
    }
    if t.backend == "docker":
        term.update({
            "docker_image":         t.docker_image,
            "container_cpu":        t.container_cpu,
            "container_memory":     t.container_memory,
            "container_disk":       t.container_disk,
            "container_persistent": t.container_persistent,
        })
    elif t.backend == "modal":
        term["modal_image"] = t.modal_image
    elif t.backend == "singularity":
        term["singularity_image"] = t.singularity_image
    elif t.backend == "ssh":
        term.update({
            "ssh_host": t.ssh_host,
            "ssh_user": t.ssh_user,
            "ssh_port": t.ssh_port,
            "ssh_key":  t.ssh_key,
        })
    if t.sudo_password:
        term["sudo_password"] = t.sudo_password
    cfg["terminal"] = term
    save_config(cfg)
    return {"ok": True}


# ── Tools ─────────────────────────────────────────────────────
class ToolsIn(BaseModel):
    firecrawl_key:            str  = ""
    fal_key:                  str  = ""
    browserbase_key:          str  = ""
    browserbase_project:      str  = ""
    browserbase_proxies:      bool = True
    browserbase_stealth:      bool = False
    browser_session_timeout:  int  = 300
    browser_inactivity:       int  = 120
    voice_openai_key:         str  = ""
    elevenlabs_key:           str  = ""
    github_token:             str  = ""
    github_app_id:            str  = ""
    github_app_key_path:      str  = ""
    github_app_install_id:    str  = ""
    honcho_key:               str  = ""
    tinker_key:               str  = ""
    wandb_key:                str  = ""
    rl_api_url:               str  = "http://localhost:8080"

@app.get("/api/tools")
async def get_tools():
    env = read_env()
    def isset(k): return bool(env.get(k))
    return {
        "firecrawl_key_set":           isset("FIRECRAWL_API_KEY"),
        "fal_key_set":                 isset("FAL_KEY"),
        "browserbase_key_set":         isset("BROWSERBASE_API_KEY"),
        "browserbase_project_set":     isset("BROWSERBASE_PROJECT_ID"),
        "browserbase_proxies":         _bool_env(env, "BROWSERBASE_PROXIES", True),
        "browserbase_stealth":         _bool_env(env, "BROWSERBASE_ADVANCED_STEALTH"),
        "browser_session_timeout":     int(env.get("BROWSER_SESSION_TIMEOUT", 300)),
        "browser_inactivity":          int(env.get("BROWSER_INACTIVITY_TIMEOUT", 120)),
        "voice_openai_key_set":        isset("VOICE_TOOLS_OPENAI_KEY"),
        "elevenlabs_key_set":          isset("ELEVENLABS_API_KEY"),
        "github_token_set":            isset("GITHUB_TOKEN"),
        "github_app_id":               env.get("GITHUB_APP_ID", ""),
        "github_app_key_path":         env.get("GITHUB_APP_PRIVATE_KEY_PATH", ""),
        "github_app_install_id":       env.get("GITHUB_APP_INSTALLATION_ID", ""),
        "honcho_key_set":              isset("HONCHO_API_KEY"),
        "tinker_key_set":              isset("TINKER_API_KEY"),
        "wandb_key_set":               isset("WANDB_API_KEY"),
        "rl_api_url":                  env.get("RL_API_URL", "http://localhost:8080"),
    }

@app.post("/api/tools")
async def save_tools(t: ToolsIn):
    updates: dict = {}
    if t.firecrawl_key:         updates["FIRECRAWL_API_KEY"]           = t.firecrawl_key
    if t.fal_key:               updates["FAL_KEY"]                     = t.fal_key
    if t.browserbase_key:       updates["BROWSERBASE_API_KEY"]         = t.browserbase_key
    if t.browserbase_project:   updates["BROWSERBASE_PROJECT_ID"]      = t.browserbase_project
    updates["BROWSERBASE_PROXIES"]          = "true" if t.browserbase_proxies else "false"
    updates["BROWSERBASE_ADVANCED_STEALTH"] = "true" if t.browserbase_stealth else "false"
    updates["BROWSER_SESSION_TIMEOUT"]      = str(t.browser_session_timeout)
    updates["BROWSER_INACTIVITY_TIMEOUT"]   = str(t.browser_inactivity)
    if t.voice_openai_key:      updates["VOICE_TOOLS_OPENAI_KEY"]      = t.voice_openai_key
    if t.elevenlabs_key:        updates["ELEVENLABS_API_KEY"]          = t.elevenlabs_key
    if t.github_token:          updates["GITHUB_TOKEN"]                = t.github_token
    if t.github_app_id:         updates["GITHUB_APP_ID"]               = t.github_app_id
    if t.github_app_key_path:   updates["GITHUB_APP_PRIVATE_KEY_PATH"] = t.github_app_key_path
    if t.github_app_install_id: updates["GITHUB_APP_INSTALLATION_ID"]  = t.github_app_install_id
    if t.honcho_key:            updates["HONCHO_API_KEY"]              = t.honcho_key
    if t.tinker_key:            updates["TINKER_API_KEY"]              = t.tinker_key
    if t.wandb_key:             updates["WANDB_API_KEY"]               = t.wandb_key
    updates["RL_API_URL"] = t.rl_api_url
    patch_env(**updates)
    return {"ok": True}


# ── Advanced ──────────────────────────────────────────────────
class AdvancedIn(BaseModel):
    compression_enabled:   bool              = True
    compression_threshold: float             = 0.85
    compression_model:     str               = "google/gemini-flash-1.5"
    toolsets:              str               = "all"
    web_debug:             bool              = False
    vision_debug:          bool              = False
    moa_debug:             bool              = False
    image_debug:           bool              = False
    oauth_trace:           bool              = False
    mcp_servers:           Dict[str, Any]    = {}

@app.get("/api/advanced")
async def get_advanced():
    cfg  = read_config()
    env  = read_env()
    comp = cfg.get("compression", {})
    ts   = cfg.get("toolsets", ["all"])
    return {
        "compression_enabled":   comp.get("enabled",       True),
        "compression_threshold": comp.get("threshold",     0.85),
        "compression_model":     comp.get("summary_model", "google/gemini-flash-1.5"),
        "toolsets":              ",".join(ts) if isinstance(ts, list) else str(ts),
        "web_debug":             _bool_env(env, "WEB_TOOLS_DEBUG"),
        "vision_debug":          _bool_env(env, "VISION_TOOLS_DEBUG"),
        "moa_debug":             _bool_env(env, "MOA_TOOLS_DEBUG"),
        "image_debug":           _bool_env(env, "IMAGE_TOOLS_DEBUG"),
        "oauth_trace":           _bool_env(env, "HERMES_OAUTH_TRACE"),
        "mcp_servers":           cfg.get("mcp_servers", {}),
    }

@app.post("/api/advanced")
async def save_advanced(a: AdvancedIn):
    cfg = read_config()
    cfg.setdefault("compression", {}).update({
        "enabled":       a.compression_enabled,
        "threshold":     a.compression_threshold,
        "summary_model": a.compression_model,
    })
    ts_raw = [x.strip() for x in a.toolsets.split(",") if x.strip()]
    cfg["toolsets"]    = ts_raw if ts_raw else ["all"]
    cfg["mcp_servers"] = a.mcp_servers
    save_config(cfg)
    patch_env(
        WEB_TOOLS_DEBUG    = "true" if a.web_debug    else "false",
        VISION_TOOLS_DEBUG = "true" if a.vision_debug else "false",
        MOA_TOOLS_DEBUG    = "true" if a.moa_debug    else "false",
        IMAGE_TOOLS_DEBUG  = "true" if a.image_debug  else "false",
        HERMES_OAUTH_TRACE = "true" if a.oauth_trace  else "false",
    )
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), log_level="info")
