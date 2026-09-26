"""One fail-closed policy boundary for sync and chunked-streaming speech."""


def enforce_pre_synthesis(text: str, provider: str) -> str:
    """Return the approved script or raise ValueError before contacting a TTS backend.

    A rewrite must be checked against every hook again. A second rewrite that changes the
    script is rejected rather than synthesizing text a prior guard never inspected.
    """
    from hermes_cli.plugins import invoke_hook

    def apply(script: str) -> str:
        result = script
        for decision in invoke_hook("pre_tts_synthesis", text=script, provider=provider):
            if not isinstance(decision, dict):
                raise ValueError("pre_tts_synthesis returned an invalid directive")
            if decision.get("action") == "block" and set(decision) <= {"action", "message"}:
                raise ValueError(str(decision.get("message") or "TTS synthesis blocked"))
            if set(decision) != {"text"} or not isinstance(decision["text"], str) or not decision["text"].strip():
                raise ValueError("pre_tts_synthesis returned an invalid script or directive")
            result = decision["text"]
        return result

    try:
        approved = apply(text)
        if approved != text and apply(approved) != approved:
            raise ValueError("pre_tts_synthesis returned an unstable script")
        return approved
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"pre_tts_synthesis dispatch failed: {exc}") from exc
