"""ChatGPT/Codex subscription speech-to-text backend."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

from agent.codex_headers import codex_cloudflare_headers
from tools.transcription_common import _error_result, _ok_result

logger = logging.getLogger("tools.transcription_tools")
OPENAI_CODEX_TRANSCRIBE_URL = "https://chatgpt.com/backend-api/transcribe"


def _codex_stt_credentials_from_pool_entry(entry: Any) -> Dict[str, Any]:
    from hermes_cli.auth import codex_account_id_from_access_token

    token = str(entry.runtime_api_key or "").strip()
    if not token:
        raise ValueError("OpenAI Codex OAuth credentials are unavailable.")
    return {
        "provider": "openai-codex",
        "api_key": token,
        "source": "credential_pool",
        "credential_id": entry.id,
        "account_id": codex_account_id_from_access_token(token),
    }


def _resolve_codex_stt_credentials() -> Dict[str, Any]:
    from agent.credential_pool import load_pool

    entry = load_pool("openai-codex").select()
    if entry is None:
        raise ValueError("OpenAI Codex OAuth credentials are unavailable.")
    return _codex_stt_credentials_from_pool_entry(entry)


def _retry_codex_stt_credentials(credentials: Dict[str, Any], status_code: int) -> Optional[Dict[str, Any]]:
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    credential_id = str(credentials.get("credential_id") or "").strip() or None
    failed_token = str(credentials.get("api_key") or "").strip()
    if status_code == 401:
        refreshed = pool.try_refresh_matching(api_key_hint=failed_token or None, credential_id=credential_id)
        if refreshed is not None and refreshed.runtime_api_key != failed_token:
            return _codex_stt_credentials_from_pool_entry(refreshed)
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code, api_key_hint=failed_token or None, credential_id=credential_id)
    else:
        next_entry = pool.select_excluding(
            credential_id=credential_id, api_key_hint=failed_token or None)
    if next_entry is None or next_entry.runtime_api_key == failed_token:
        return None
    return _codex_stt_credentials_from_pool_entry(next_entry)


def _mark_codex_stt_credentials_failed(credentials: Dict[str, Any], status_code: int) -> None:
    from agent.credential_pool import load_pool

    load_pool("openai-codex").mark_exhausted_and_rotate(
        status_code=status_code,
        api_key_hint=str(credentials.get("api_key") or "").strip() or None,
        credential_id=str(credentials.get("credential_id") or "").strip() or None,
    )


def _has_codex_stt_backend() -> bool:
    from hermes_cli.auth import has_codex_runtime_credentials

    return has_codex_runtime_credentials()


def _transcribe_openai_codex(
    file_path: str,
    model_name: str = "",
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Transcribe through ChatGPT's subscription-backed private dictation endpoint."""
    del model_name, prompt
    import requests
    from tools.transcription_tools import (
        _mark_codex_stt_credentials_failed as mark_failed,
        _resolve_codex_stt_credentials as resolve_credentials,
        _retry_codex_stt_credentials as retry_credentials,
    )

    language = str(language or "").strip()
    timeout = max(1, min(int(timeout), 600))

    def _request(creds: Dict[str, Any]):
        token = str(creds.get("api_key") or "").strip()
        if not token:
            raise ValueError("ChatGPT/Codex OAuth login is missing")
        suffix = Path(file_path).suffix.lower()
        mime_type = ({
            ".webm": "audio/webm", ".ogg": "audio/ogg", ".oga": "audio/ogg",
            ".opus": "audio/ogg", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
        }.get(suffix) or mimetypes.guess_type(file_path)[0] or "application/octet-stream")
        headers = codex_cloudflare_headers(token, base_url=OPENAI_CODEX_TRANSCRIBE_URL)
        headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})
        account_id = str(creds.get("account_id") or "").strip()
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        with open(file_path, "rb") as audio_file:
            return requests.post(
                OPENAI_CODEX_TRANSCRIBE_URL, headers=headers,
                files={"file": (Path(file_path).name, audio_file, mime_type)},
                data={"language": language} if language else {}, timeout=timeout,
                allow_redirects=False)

    try:
        credentials = resolve_credentials()
        response = _request(credentials)
        if response.status_code in {401, 429}:
            replacement = retry_credentials(credentials, response.status_code)
            if replacement is not None:
                credentials = replacement
                response = _request(credentials)
                if response.status_code == 401:
                    mark_failed(credentials, response.status_code)
        if 300 <= response.status_code < 400:
            return _error_result("Codex OAuth transcription refused an unexpected redirect.")
        if response.status_code == 403 and response.headers.get("cf-mitigated") == "challenge":
            return _error_result("ChatGPT blocked Codex OAuth transcription at its edge; retry later.")
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            return _error_result("Codex OAuth transcription returned invalid JSON.")
        if not isinstance(payload, dict):
            return _error_result("Codex OAuth transcription returned an invalid response.")
        transcript_value = payload.get("text")
        if not isinstance(transcript_value, str) or not transcript_value.strip():
            return _error_result("Codex OAuth transcription returned no text.")
        transcript = transcript_value.strip()
        logger.info("Transcribed %s via Codex OAuth (lang=%s, %d chars)",
                    Path(file_path).name, language or "auto", len(transcript))
        return _ok_result(transcript, "openai-codex")
    except PermissionError:
        return _error_result(f"Permission denied: {file_path}")
    except requests.Timeout:
        return _error_result("Codex OAuth transcription request timed out.")
    except requests.ConnectionError:
        return _error_result("Could not connect to ChatGPT for Codex OAuth transcription.")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return _error_result(f"Codex OAuth transcription failed with HTTP {status}.")
    except (ValueError, KeyError) as exc:
        return _error_result(str(exc))
    except Exception as exc:
        logger.error("Codex OAuth transcription failed: %s", exc, exc_info=True)
        return _error_result("Codex OAuth transcription failed unexpectedly.")
