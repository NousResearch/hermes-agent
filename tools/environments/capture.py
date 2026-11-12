"""Output integrity is independent of the command's exit status.

Diagnostics deliberately contain no exception text: backend errors can embed
commands, credentials or captured output. Never retry a command on this signal.
"""

from dataclasses import dataclass


@dataclass(kw_only=True)
class CaptureStatus:
    capture_error: str = ""

    def capture_failed(self, stage: str) -> None:
        if not self.capture_error:
            self.capture_error = (
                f"Output capture failed during {stage}; output may be incomplete. "
                "Inspect the command's effects before considering a retry."
            )

    def capture_fields(self) -> dict:
        return capture_fields(self.capture_error)


def capture_fields(error: str, incomplete: bool = False) -> dict:
    if not error and not incomplete:
        return {}
    if not error:
        error = "Output capture is incomplete. Inspect the command's effects before considering a retry."
    return {"capture_error": error, "output_incomplete": True}


class OutputCaptureError(RuntimeError):
    """A consumer cannot safely use captured output as a complete payload."""


def require_complete_output(result: dict) -> None:
    if result.get("output_incomplete") or result.get("capture_error"):
        raise OutputCaptureError(
            "Output capture is incomplete; refusing to use a partial payload. "
            "The command may already have taken effect; inspect its effects before retrying."
        )
