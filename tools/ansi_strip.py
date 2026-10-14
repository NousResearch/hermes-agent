"""Strip ANSI escape sequences from subprocess output so they never reach the
model's context (models otherwise copy them into file writes). Covers full
ECMA-48: CSI, OSC (BEL/ST), DCS/SOS/PM/APC, nF, Fp/Fe/Fs and 8-bit C1 controls."""

import re

_ANSI_ESCAPE_RE = re.compile(
    r"\x1b"
    r"(?:"
        r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # CSI sequence
        r"|\][\s\S]*?(?:\x07|\x1b\\)"                  # OSC (BEL or ST terminator)
        r"|[PX^_][\s\S]*?(?:\x1b\\)"                   # DCS/SOS/PM/APC strings
        r"|[\x20-\x2f]+[\x30-\x7e]"                    # nF escape sequences
        r"|[\x30-\x7e]"                                 # Fp/Fe/Fs single-byte
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"       # 8-bit CSI
    r"|\x9d[\s\S]*?(?:\x07|\x9c)"                       # 8-bit OSC
    r"|[\x80-\x9f]",                                    # Other 8-bit C1 controls
    re.DOTALL,
)
# Fast-path checks: skip the full regex when no candidate bytes are present.
_HAS_ESCAPE = re.compile(r"[\x1b\x80-\x9f]")
_HAS_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_HAS_UNICODE_TAG = re.compile(r"[\U000E0000-\U000E007F]")

# C0 controls (minus tab/newline/CR) plus DEL: they survive strip_ansi() (which only
# removes *sequences*) but are dangerous echoed to a terminal (BEL, backspace, NUL).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Unicode TAG chars (U+E0000–U+E007F) render as nothing but LLM tokenizers see them:
# the "ASCII smuggling" injection channel. Emoji tag sequences (TR51: U+1F3F4 base +
# tag spec + U+E007F CANCEL TAG, e.g. Scotland/Wales flags) are the only legit use.
# Deprecated as language tags, these render as nothing in every terminal and chat UI but are perfectly
# visible to an LLM tokenizer — the classic "ASCII smuggling" prompt-injection channel (hide
# `\u{E0069}\u{E0067}\u{E006E}...` = invisible instructions inside otherwise benign tool output). Ported
# from block/goose#10746. The ONLY legitimate modern use is emoji tag sequences (Unicode TR51): a U+1F3F4
# black-flag base followed by tag spec characters and the U+E007F CANCEL TAG terminator (e.g. the flags of
# Scotland/Wales/England). goose strips those too; we preserve them — same rationale as keeping ZWJ inside
# emoji sequences.
_UNICODE_TAG_SUB_RE = re.compile(
    r"(\U0001F3F4[\U000E0020-\U000E007E]+\U000E007F)"  # valid emoji tag seq (kept)
    r"|[\U000E0000-\U000E007F]"                        # any other tag char (stripped)
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences; clean text passes through unchanged (fast path)."""
    if not text or not _HAS_ESCAPE.search(text):
        return text
    return _ANSI_ESCAPE_RE.sub("", text)


def sanitize_display_text(text: str) -> str:
    """Sanitize stored/untrusted text before echoing it to a terminal: strips ANSI
    sequences AND bare control chars, keeping only newlines/tabs (CRs become newlines
    so ``\\r``-overwrite spoofing can't hide content). Rich's ``Text()`` does NOT
    neutralize raw escape bytes, so a replayed ``/resume`` message must not be able
    to clear the screen, retitle the window, or restyle UI.

    Use this when re-rendering conversation history or other persisted text in a terminal UI (e.g. the
    ``/resume`` recap): a message that arrived with embedded escapes — pasted content, gateway-origin text,
    or model output echoing injected tool results — must not be able to clear the screen, retitle the
    window, move the cursor, or restyle adjacent UI when replayed. Mirrors openai/codex#31494
    (``sanitize_user_text``).
    """
    if not text or not _HAS_CONTROL.search(text):
        return text
    text = strip_ansi(text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS_RE.sub("", text)


def strip_unicode_tags(text: str) -> str:
    """Remove invisible Unicode TAG chars (a prompt-injection smuggling channel in
    untrusted tool output); valid emoji tag sequences are preserved.

    Returns the input unchanged (fast path) when no plane-14 tag characters are present. Ported from
    block/goose#10746.
    """
    if not text or not _HAS_UNICODE_TAG.search(text):
        return text
    return _UNICODE_TAG_SUB_RE.sub(lambda m: m.group(1) or "", text)
