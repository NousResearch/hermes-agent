#!/usr/bin/env python3
"""
Phone Call Tool Module

Make outbound phone calls using AI voice agents on the user's behalf.

Supports two providers:
- Bland.ai (simple, all-in-one, built-in voices): needs BLAND_API_KEY
- Vapi (flexible, better voice quality via ElevenLabs/Deepgram/PlayHT etc.):
    needs VAPI_API_KEY and a phone number imported into Vapi (e.g. from Twilio)

Configuration is loaded from ~/.hermes/config.yaml under the 'phone:' key.
The user chooses the provider; the model sends the task.

Config example (~/.hermes/config.yaml):

    phone:
        provider: bland          # or "vapi"
        bland:
            api_key: org_xxx
            default_voice: mason
        vapi:
            api_key: xxx-xxx
            phone_number_id: xxx-xxx
            default_voice_provider: 11labs
            default_voice_id: cjVigY5qzO86Huf0OWal
            model: gpt-4o

Usage:
    from tools.phone_tool import phone_call_tool, phone_call_result_tool

    result = phone_call_tool(
        phone_number="+15551234567",
        task="Schedule a dental cleaning for John, Tuesday afternoon.",
    )
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "bland"
DEFAULT_MAX_DURATION = 3  # minutes

# Bland defaults
BLAND_API_BASE = "https://api.bland.ai/v1"
BLAND_DEFAULT_VOICE = "mason"
BLAND_DEFAULT_MODEL = "enhanced"
BLAND_VOICES = {
    "mason": "Male, natural, friendly (recommended)",
    "josh": "Male, conversational",
    "ryan": "Male, professional",
    "matt": "Male, casual",
    "evelyn": "Female, natural, warm (recommended)",
    "tina": "Female, warm, friendly",
    "june": "Female, conversational",
}

# Vapi defaults
VAPI_API_BASE = "https://api.vapi.ai"
VAPI_DEFAULT_VOICE_PROVIDER = "11labs"
VAPI_DEFAULT_VOICE_ID = "cjVigY5qzO86Huf0OWal"  # ElevenLabs "Eric"
VAPI_DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_phone_config() -> Dict[str, Any]:
    """Load phone config from ~/.hermes/config.yaml under the 'phone:' key."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        return config.get("phone", {})
    except ImportError:
        logger.debug("hermes_cli.config not available, using default phone config")
        return {}
    except Exception as e:
        logger.warning("Failed to load phone config: %s", e, exc_info=True)
        return {}


def _get_provider(phone_config: Dict[str, Any]) -> str:
    """Get the configured phone provider name."""
    return phone_config.get("provider", DEFAULT_PROVIDER).lower().strip()


# ---------------------------------------------------------------------------
# Bland.ai provider
# ---------------------------------------------------------------------------
def _bland_get_api_key(phone_config: Dict[str, Any]) -> Optional[str]:
    key = os.getenv("BLAND_API_KEY", "")
    if key:
        return key
    return phone_config.get("bland", {}).get("api_key", "")


def _bland_make_call(
    api_key: str,
    phone_number: str,
    task: str,
    voice: str,
    first_sentence: Optional[str] = None,
    max_duration: int = DEFAULT_MAX_DURATION,
) -> Dict[str, Any]:
    payload = {
        "phone_number": phone_number,
        "task": task,
        "voice": voice,
        "model": BLAND_DEFAULT_MODEL,
        "max_duration": max_duration,
        "record": True,
        "wait_for_greeting": True,
    }
    if first_sentence:
        payload["first_sentence"] = first_sentence

    resp = requests.post(
        f"{BLAND_API_BASE}/calls",
        headers={"Content-Type": "application/json", "authorization": api_key},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _bland_get_call(api_key: str, call_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{BLAND_API_BASE}/calls/{call_id}",
        headers={"authorization": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _bland_analyze_call(
    api_key: str, call_id: str, questions: list
) -> Dict[str, Any]:
    resp = requests.post(
        f"{BLAND_API_BASE}/calls/{call_id}/analyze",
        headers={"Content-Type": "application/json", "authorization": api_key},
        json={"questions": questions},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Vapi provider
# ---------------------------------------------------------------------------
def _vapi_get_api_key(phone_config: Dict[str, Any]) -> Optional[str]:
    key = os.getenv("VAPI_API_KEY", "")
    if key:
        return key
    return phone_config.get("vapi", {}).get("api_key", "")


def _vapi_get_phone_number_id(phone_config: Dict[str, Any]) -> Optional[str]:
    pid = os.getenv("VAPI_PHONE_NUMBER_ID", "")
    if pid:
        return pid
    return phone_config.get("vapi", {}).get("phone_number_id", "")


def _vapi_make_call(
    api_key: str,
    phone_number_id: str,
    phone_number: str,
    task: str,
    first_sentence: Optional[str] = None,
    max_duration: int = DEFAULT_MAX_DURATION,
    voice_provider: str = VAPI_DEFAULT_VOICE_PROVIDER,
    voice_id: str = VAPI_DEFAULT_VOICE_ID,
    model: str = VAPI_DEFAULT_MODEL,
) -> Dict[str, Any]:
    assistant = {
        "model": {
            "provider": "openai",
            "model": model,
            "messages": [{"role": "system", "content": task}],
        },
        "voice": {
            "provider": voice_provider,
            "voiceId": voice_id,
        },
        "maxDurationSeconds": max_duration * 60,
    }
    if first_sentence:
        assistant["firstMessage"] = first_sentence

    payload = {
        "phoneNumberId": phone_number_id,
        "customer": {"number": phone_number},
        "assistant": assistant,
    }

    resp = requests.post(
        f"{VAPI_API_BASE}/call",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _vapi_get_call(api_key: str, call_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{VAPI_API_BASE}/call/{call_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ===========================================================================
# Tool: phone_call — initiate an outbound call
# ===========================================================================
def phone_call_tool(
    phone_number: str,
    task: str,
    voice: Optional[str] = None,
    first_sentence: Optional[str] = None,
    max_duration: Optional[int] = None,
) -> str:
    """
    Initiate an outbound phone call via AI voice agent.

    The provider (Bland or Vapi) is determined by config. The model
    doesn't need to know which provider is active — just provide the
    phone number and task.

    Args:
        phone_number: Phone number in E.164 format (e.g. +15551234567).
        task: Instructions for the AI voice agent.
        voice: Voice to use (provider-specific, optional).
        first_sentence: Opening line the AI speaks first (optional).
        max_duration: Max call duration in minutes (default: 3).

    Returns:
        JSON with call_id on success, or error on failure.
    """
    if not phone_number or not phone_number.strip():
        return json.dumps({"success": False, "error": "phone_number is required"})
    if not task or not task.strip():
        return json.dumps({"success": False, "error": "task is required"})

    phone_number = phone_number.strip()
    if not phone_number.startswith("+"):
        return json.dumps(
            {
                "success": False,
                "error": f"Phone number must be in E.164 format (e.g. +15551234567), got: {phone_number}",
            }
        )

    phone_config = _load_phone_config()
    provider = _get_provider(phone_config)

    if max_duration is None:
        max_duration = DEFAULT_MAX_DURATION

    try:
        if provider == "vapi":
            return _make_call_vapi(phone_config, phone_number, task, voice, first_sentence, max_duration)
        else:
            return _make_call_bland(phone_config, phone_number, task, voice, first_sentence, max_duration)
    except Exception as e:
        error_msg = f"Failed to initiate call ({provider}): {type(e).__name__}: {e}"
        logger.error("%s", error_msg, exc_info=True)
        return json.dumps({"success": False, "error": error_msg})


def _make_call_bland(phone_config, phone_number, task, voice, first_sentence, max_duration) -> str:
    api_key = _bland_get_api_key(phone_config)
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "No Bland.ai API key found. Set BLAND_API_KEY env var or add to config.yaml under phone.bland.api_key. Sign up free at https://app.bland.ai",
            }
        )

    bland_config = phone_config.get("bland", {})
    if voice is None:
        voice = bland_config.get("default_voice", BLAND_DEFAULT_VOICE)

    result = _bland_make_call(
        api_key=api_key,
        phone_number=phone_number,
        task=task,
        voice=voice,
        first_sentence=first_sentence,
        max_duration=max_duration,
    )

    call_id = result.get("call_id")
    if not call_id:
        return json.dumps({"success": False, "error": f"Bland.ai returned no call_id: {result}"})

    logger.info("Phone call initiated (bland): %s -> %s (call_id: %s)", voice, phone_number, call_id)
    return json.dumps(
        {
            "success": True,
            "provider": "bland",
            "call_id": call_id,
            "phone_number": phone_number,
            "voice": voice,
            "max_duration": max_duration,
            "message": "Call queued. Use phone_call_result to check status and get transcript.",
        }
    )


def _make_call_vapi(phone_config, phone_number, task, voice, first_sentence, max_duration) -> str:
    api_key = _vapi_get_api_key(phone_config)
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "No Vapi API key found. Set VAPI_API_KEY env var or add to config.yaml under phone.vapi.api_key. Sign up at https://dashboard.vapi.ai",
            }
        )

    phone_number_id = _vapi_get_phone_number_id(phone_config)
    if not phone_number_id:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "No Vapi phone number ID found. Vapi requires a Twilio number for outbound calls. "
                    "Setup: 1) Sign up at twilio.com 2) Buy a phone number 3) Import it into Vapi with your "
                    "Twilio Account SID + Auth Token via POST https://api.vapi.ai/phone-number "
                    "4) Set the returned ID as VAPI_PHONE_NUMBER_ID env var or phone.vapi.phone_number_id in config."
                ),
            }
        )

    vapi_config = phone_config.get("vapi", {})
    voice_provider = vapi_config.get("default_voice_provider", VAPI_DEFAULT_VOICE_PROVIDER)
    voice_id = voice if voice else vapi_config.get("default_voice_id", VAPI_DEFAULT_VOICE_ID)
    model = vapi_config.get("model", VAPI_DEFAULT_MODEL)

    result = _vapi_make_call(
        api_key=api_key,
        phone_number_id=phone_number_id,
        phone_number=phone_number,
        task=task,
        first_sentence=first_sentence,
        max_duration=max_duration,
        voice_provider=voice_provider,
        voice_id=voice_id,
        model=model,
    )

    call_id = result.get("id")
    if not call_id:
        return json.dumps({"success": False, "error": f"Vapi returned no call id: {result}"})

    logger.info("Phone call initiated (vapi): %s -> %s (call_id: %s)", voice_id, phone_number, call_id)
    return json.dumps(
        {
            "success": True,
            "provider": "vapi",
            "call_id": call_id,
            "phone_number": phone_number,
            "voice_provider": voice_provider,
            "voice_id": voice_id,
            "max_duration": max_duration,
            "message": "Call queued. Use phone_call_result to check status and get transcript.",
        }
    )


# ===========================================================================
# Tool: phone_call_result — get call status/transcript
# ===========================================================================
def phone_call_result_tool(
    call_id: str,
    analyze: Optional[str] = None,
) -> str:
    """
    Get the status, transcript, and recording of a phone call.

    Automatically detects the provider from config.

    Args:
        call_id: The call ID returned by phone_call.
        analyze: Optional comma-separated questions to analyze the call
                 (Bland.ai only, e.g. "Was the appointment confirmed?").

    Returns:
        JSON with call status, transcript, recording URL, and optional analysis.
    """
    if not call_id or not call_id.strip():
        return json.dumps({"success": False, "error": "call_id is required"})

    phone_config = _load_phone_config()
    provider = _get_provider(phone_config)

    try:
        if provider == "vapi":
            return _get_result_vapi(phone_config, call_id.strip(), analyze)
        else:
            return _get_result_bland(phone_config, call_id.strip(), analyze)
    except Exception as e:
        error_msg = f"Failed to get call result ({provider}): {type(e).__name__}: {e}"
        logger.error("%s", error_msg, exc_info=True)
        return json.dumps({"success": False, "error": error_msg})


def _get_result_bland(phone_config, call_id, analyze) -> str:
    api_key = _bland_get_api_key(phone_config)
    if not api_key:
        return json.dumps({"success": False, "error": "No Bland.ai API key found."})

    data = _bland_get_call(api_key, call_id)
    result = {
        "success": True,
        "provider": "bland",
        "status": data.get("status"),
        "duration_minutes": data.get("call_length"),
        "answered_by": data.get("answered_by"),
        "transcript": data.get("concatenated_transcript", ""),
        "recording_url": data.get("recording_url"),
    }

    if analyze and data.get("status") == "completed":
        questions = [[q.strip(), "string"] for q in analyze.split(",") if q.strip()]
        if questions:
            try:
                analysis = _bland_analyze_call(api_key, call_id, questions)
                result["analysis"] = analysis
            except Exception as e:
                result["analysis_error"] = str(e)

    return json.dumps(result, ensure_ascii=False)


def _get_result_vapi(phone_config, call_id, analyze) -> str:
    api_key = _vapi_get_api_key(phone_config)
    if not api_key:
        return json.dumps({"success": False, "error": "No Vapi API key found."})

    data = _vapi_get_call(api_key, call_id)
    result = {
        "success": True,
        "provider": "vapi",
        "status": data.get("status"),
        "duration_seconds": data.get("duration"),
        "ended_reason": data.get("endedReason"),
        "transcript": data.get("transcript", ""),
        "recording_url": data.get("recordingUrl"),
        "summary": data.get("summary"),
        "cost": data.get("cost"),
    }

    return json.dumps(result, ensure_ascii=False)


# ===========================================================================
# Requirements check
# ===========================================================================
def check_phone_requirements() -> bool:
    """Check if phone calling is available (requests + at least one provider configured)."""
    if not _HAS_REQUESTS:
        return False
    phone_config = _load_phone_config()
    # Check Bland
    if _bland_get_api_key(phone_config):
        return True
    # Check Vapi
    if _vapi_get_api_key(phone_config) and _vapi_get_phone_number_id(phone_config):
        return True
    return False


# ===========================================================================
# Main -- diagnostics
# ===========================================================================
if __name__ == "__main__":
    print("📞 Phone Call Tool Module")
    print("=" * 50)

    print(f"\n  requests: {'✅ installed' if _HAS_REQUESTS else '❌ not installed (pip install requests)'}")

    phone_config = _load_phone_config()
    provider = _get_provider(phone_config)
    print(f"  Provider: {provider}")

    # Bland
    bland_key = _bland_get_api_key(phone_config)
    print(f"\n  Bland.ai:")
    print(f"    API key:  {'✅ set' if bland_key else '❌ not set (BLAND_API_KEY)'}")
    bland_config = phone_config.get("bland", {})
    print(f"    Voice:    {bland_config.get('default_voice', BLAND_DEFAULT_VOICE)}")

    # Vapi
    vapi_key = _vapi_get_api_key(phone_config)
    vapi_phone = _vapi_get_phone_number_id(phone_config)
    vapi_config = phone_config.get("vapi", {})
    print(f"\n  Vapi:")
    print(f"    API key:      {'✅ set' if vapi_key else '❌ not set (VAPI_API_KEY)'}")
    print(f"    Phone number: {'✅ set' if vapi_phone else '❌ not set (VAPI_PHONE_NUMBER_ID)'}")
    print(f"    Voice:        {vapi_config.get('default_voice_provider', VAPI_DEFAULT_VOICE_PROVIDER)}:{vapi_config.get('default_voice_id', VAPI_DEFAULT_VOICE_ID)}")
    print(f"    Model:        {vapi_config.get('model', VAPI_DEFAULT_MODEL)}")

    print(f"\n  Bland.ai voices:")
    for name, desc in BLAND_VOICES.items():
        print(f"    {name:10s} — {desc}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

PHONE_CALL_SCHEMA = {
    "name": "phone_call",
    "description": (
        "Make an outbound phone call using an AI voice agent. The agent dials "
        "the number, has a voice conversation to accomplish the given task "
        "(e.g. schedule appointment, make reservation), and records the call. "
        "Returns a call_id to check results later with phone_call_result. "
        "Supports Bland.ai and Vapi providers (configured in config.yaml). "
        "IMPORTANT: Always confirm with the user before calling — show them the "
        "number, purpose, and voice."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Phone number to call in E.164 format (e.g. '+15551234567')",
            },
            "task": {
                "type": "string",
                "description": (
                    "Detailed instructions for the AI voice agent — what to say, "
                    "what to accomplish, what info to share. Write it like you're "
                    "briefing a human assistant."
                ),
            },
            "voice": {
                "type": "string",
                "description": (
                    "Voice to use. For Bland: mason, josh, ryan, matt (male); "
                    "evelyn, tina, june (female). For Vapi: a voice ID from your "
                    "configured voice provider (e.g. ElevenLabs). Leave empty for default."
                ),
            },
            "first_sentence": {
                "type": "string",
                "description": "Optional opening line the AI speaks first.",
            },
            "max_duration": {
                "type": "integer",
                "description": "Maximum call duration in minutes (default: 3).",
            },
        },
        "required": ["phone_number", "task"],
    },
}

PHONE_CALL_RESULT_SCHEMA = {
    "name": "phone_call_result",
    "description": (
        "Get the status, transcript, and recording of a phone call "
        "initiated with phone_call. Poll this after initiating a call "
        "to see the conversation and outcome."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "call_id": {
                "type": "string",
                "description": "The call_id returned by phone_call.",
            },
            "analyze": {
                "type": "string",
                "description": (
                    "Optional comma-separated questions to extract structured "
                    "info from the call (Bland.ai only, e.g. "
                    "'Was the appointment confirmed?,What time was booked?')."
                ),
            },
        },
        "required": ["call_id"],
    },
}

registry.register(
    name="phone_call",
    toolset="phone",
    schema=PHONE_CALL_SCHEMA,
    handler=lambda args, **kw: phone_call_tool(
        phone_number=args.get("phone_number", ""),
        task=args.get("task", ""),
        voice=args.get("voice"),
        first_sentence=args.get("first_sentence"),
        max_duration=args.get("max_duration"),
    ),
    check_fn=check_phone_requirements,
)

registry.register(
    name="phone_call_result",
    toolset="phone",
    schema=PHONE_CALL_RESULT_SCHEMA,
    handler=lambda args, **kw: phone_call_result_tool(
        call_id=args.get("call_id", ""),
        analyze=args.get("analyze"),
    ),
    check_fn=check_phone_requirements,
)
