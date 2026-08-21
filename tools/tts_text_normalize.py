"""Utilities for preparing assistant text for speech synthesis.

The TTS provider should receive a spoken script, not raw chat Markdown.  This
module centralises the lightweight, deterministic cleanup used by explicit TTS
calls and gateway auto-TTS replies.

Non-ASCII characters are written as escapes on purpose so the file stays free of
invisible/look-alike glyphs.
"""

from __future__ import annotations

import html
import re

# Sentinel appended to former heading lines so smooth_whitespace_for_tts can
# fold a heading into the sentence that follows it ("Weather, it will be sunny")
# rather than leaving a bare "Weather." label that reads abruptly aloud.
_HEAD = "\x00"

_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_MD_UNDERSCORE_BOLD_RE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", flags=re.DOTALL)
_MD_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", flags=re.DOTALL)
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~", flags=re.DOTALL)
_MD_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", flags=re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", flags=re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", flags=re.MULTILINE)
_MD_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
_URL_RE = re.compile(r"https?://\S+")
# Local file links ("MEDIA:/Users/me/file.xlsx") are click targets on screen,
# not speech. The client renders them as "Open <slug>"; nothing a voice can
# say gracefully, and ElevenLabs-style models stutter/loop on the hyphenated
# slug ("pt-es", "xlsx"). Keep the audio clean: the assistant's own prose says
# "the files are below"; the token itself is silence.
_MEDIA_PATH_RE = re.compile(r"MEDIA:\S+")
# Bare filesystem paths in prose ("~/.config/himalaya/config.toml", "/etc/hosts",
# "src/lib/app.ts") have no MEDIA: marker, so a shape match is the only guard.
# A path is an address for the screen, not speech: ElevenLabs loops on "slash
# dot config slash himalaya slash config dot toml". Tilde paths are unambiguous;
# absolute paths and dot-slash paths need at least one letter (so "/06/02" is
# left alone); bare relative paths must end in a file extension, which keeps
# "and/or", "N/A", "2026/06/02", "1.5/2.5" and "5/month" intact. The replacement
# is "the path": the surrounding prose carries the meaning, the path is on screen.
_PATH_TOKEN_RE = re.compile(
    r"~/(?:[\w.-]+/)*[\w.-]+(?:\.[A-Za-z]{1,8})?"
    r"|(?:\./|/)(?=[\w.-]*[A-Za-z])(?:[\w.-]+/)+[\w.-]+(?:\.[A-Za-z]{1,8})?"
    r"|(?:\./|/)[\w.-]+\.[A-Za-z]{1,8}"
    r"|(?<![\w/])[\w.-]+(?:/[\w.-]+)+\.[A-Za-z]{1,8}",
)

# Broad emoji / pictograph cleanup.  Voice providers vary a lot here; most read
# emojis as awkward labels, so keep the speech script calm and literal.
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "☀-➿"
    "]+",
    flags=re.UNICODE,
)
_VARIATION_SELECTOR_RE = re.compile("[︎️]")


def strip_markdown_for_tts(text: str) -> str:
    """Strip Markdown/Telegram formatting while preserving readable words."""
    if not text:
        return ""

    text = html.unescape(str(text))
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: f" {m.group(1)} " if m.group(1) else " ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _MEDIA_PATH_RE.sub(" ", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_ITALIC_RE.sub(r"\1", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    # Mark headings (do not just delete the marker): the whitespace pass folds a
    # heading into the sentence after it so speech says "Weather, it will be
    # sunny" instead of a clipped "Weather." then a separate sentence.
    text = _MD_HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip() + _HEAD, text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_ITEM_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)

    # Pipe tables are terrible read aloud.  Turn any leftover pipes into pauses
    # instead of letting a provider speak "vertical bar".
    text = _MD_TABLE_PIPE_RE.sub("; ", text)
    return text


def _normalize_temperature_ranges(text: str) -> str:
    # 11-17 degrees C -> "11 to 17 degrees Celsius" (en/em dash or hyphen).
    text = re.sub(
        r"(?<!\w)([-+\u2212]?\d+(?:\.\d+)?)\s*[\u2013\u2014-]\s*([-+\u2212]?\d+(?:\.\d+)?)\s*°\s*C\b",
        lambda m: f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees Celsius",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)([-+\u2212]?\d+(?:\.\d+)?)\s*[\u2013\u2014-]\s*([-+\u2212]?\d+(?:\.\d+)?)\s*°\s*F\b",
        lambda m: f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees Fahrenheit",
        text,
        flags=re.IGNORECASE,
    )
    return text


def normalize_symbols_for_tts(text: str) -> str:
    """Expand common symbols/shorthand into words a TTS engine reads well."""
    if not text:
        return ""

    text = str(text)
    text = re.sub("[   ]", " ", text)  # non-breaking / thin spaces
    text = text.replace("\u2212", "-")  # minus sign
    text = text.replace("…", "...")  # ellipsis
    text = _normalize_temperature_ranges(text)

    # Temperatures with a number.  Do this before generic degree handling.
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*C\b", r"\1 degrees Celsius", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*F\b", r"\1 degrees Fahrenheit", text, flags=re.IGNORECASE)
    # Bare units with no leading number ("measured in degrees C").
    text = re.sub(r"°\s*C\b", "degrees Celsius", text, flags=re.IGNORECASE)
    text = re.sub(r"°\s*F\b", "degrees Fahrenheit", text, flags=re.IGNORECASE)
    # Any remaining degree symbol (angles, stray cases).
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°", r"\1 degrees", text)
    text = text.replace("°", " degrees")

    # Common weather/travel units.
    text = re.sub(r"(?<=\d)\s*km\s*/\s*h\b", " kilometres per hour", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*km/h\b", " kilometres per hour", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*mm\b", " millimetres", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*cm\b", " centimetres", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*m\b", " metres", text, flags=re.IGNORECASE)

    # Numeric rates only ("5/month" -> "5 per month").  Requiring digit-then-letter
    # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " per ", text)

    # Money and percentages.  The integer part must END in a digit so a trailing
    # comma ("A$50, ...") is not swallowed into the spoken amount.
    text = re.sub(r"NZ\$\s*([\d,]*\d(?:\.\d+)?)", r"\1 New Zealand dollars", text, flags=re.IGNORECASE)
    text = re.sub(r"A\$\s*([\d,]*\d(?:\.\d+)?)", r"\1 Australian dollars", text, flags=re.IGNORECASE)
    text = re.sub(r"US\$\s*([\d,]*\d(?:\.\d+)?)", r"\1 US dollars", text, flags=re.IGNORECASE)
    text = re.sub(r"€\s*([\d,]*\d(?:\.\d+)?)", r"\1 euros", text)
    # PT/ES writes the sign after the amount ("1.499,90 €"); cover both
    # placements, then any bare remainder ("prices in €").
    text = re.sub(r"([\d,]*\d(?:\.\d+)?)\s*€", r"\1 euros", text)
    text = text.replace("€", " euros")
    text = re.sub(r"£\s*([\d,]*\d(?:\.\d+)?)", r"\1 pounds", text)
    text = re.sub(r"\$\s*([\d,]*\d(?:\.\d+)?)", r"\1 dollars", text)
    text = re.sub(r"(?<=\d)\s*%", " percent", text)

    # Operators and separators that commonly leak from formatted answers.
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\d)\s*×\s*", " times ", text)  # "2×5090" -> "2 times 5090"
    text = re.sub("[•◦▪▫]", " ", text)  # bullet glyphs
    text = text.replace("→", " to ")  # ->
    text = text.replace("⇒", " to ")  # =>

    # Bare paths first, before the tilde/units rules can eat bits of them.
    text = _PATH_TOKEN_RE.sub(" the path ", text)

    # Em/en dashes: a pause the voice cannot read, same family as the colon
    # trigger. A digit-to-digit range becomes "to" ("pages 5–10" reads "pages
    # 5 to 10"); a dash before an uppercase letter opens a new sentence; every
    # other dash becomes a comma pause. The range pass runs first so the later
    # rules can consume every remaining dash, including ones next to digits
    # ("Step 1 — open the file") that a naive digit guard would leave raw.
    text = re.sub(r"(\d)\s*[—–]\s*(\d)", r"\1 to \2", text)
    text = re.sub(r"\s*[—–]\s*(?=[A-Z])", ". ", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)

    text = text.replace("≈", " about ")  # almost equal
    text = text.replace("~", " about ")

    text = _VARIATION_SELECTOR_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text


def smooth_whitespace_for_tts(text: str) -> str:
    """Collapse visual formatting into calm spoken paragraphs.

    A former heading line (marked with the _HEAD sentinel) folds into the next
    content line as a spoken lead-in: "Weather" + "It will be sunny" becomes
    "Weather, It will be sunny."  A heading with no content after it becomes its
    own short sentence.
    """
    if not text:
        return ""

    raw_lines = text.splitlines()
    add_sentence_pauses = sum(1 for raw_line in raw_lines if raw_line.replace(_HEAD, "").strip()) > 1
    lines: list[str] = []
    pending_heading: str | None = None

    def flush_pending() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            lines.append(pending_heading.rstrip(".:;,") + ".")
            pending_heading = None

    for raw_line in raw_lines:
        is_heading = raw_line.rstrip().endswith(_HEAD)
        line = raw_line.replace(_HEAD, "").strip()
        if not line:
            # Hold a pending heading across blank lines so it still folds into
            # the next real content line; otherwise just collapse the blank.
            if pending_heading is None and lines and lines[-1] != "":
                lines.append("")
            continue
        if is_heading:
            flush_pending()
            pending_heading = line.rstrip(".:;,")
            continue
        if pending_heading is not None:
            line = f"{pending_heading.rstrip('.:;,')}, {line}"
            pending_heading = None
        if add_sentence_pauses:
            if line[-1] in ":;":
                # A trailing colon promises a continuation ("the list:"); the
                # next raw token (file link, code fence) is gone from speech by
                # now, so the voice would hang on the open pause. Close it.
                line = line.rstrip(":;").rstrip() + "."
            elif line[-1] not in ".!?":
                line += "."
        lines.append(line)

    flush_pending()

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\.{4,}", "...", text)
    # Single-line text ("Here is the list:") skips the per-line pause pass, so
    # close a final colon here too. Only ever touches the very end of the text,
    # and never a digit-preceded colon ("final score 3:2" stays a ratio).
    text = re.sub(r"([^0-9])(?:[:;]\s*)+$", r"\1.", text)
    return text.strip()


# Reasoning blocks: models with ``/reasoning show`` enabled emit
# ``<think>...</think>`` blocks in the final assistant message.  Users want to
# SEE reasoning, not hear it read aloud (#34213).
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL | re.IGNORECASE)
# An unterminated block (streaming cut-off) should still not be spoken.
_THINK_BLOCK_OPEN_RE = re.compile(r"<think[\s>].*\Z", flags=re.DOTALL | re.IGNORECASE)

# Turn-end file-mutation verifier footer appended by run_agent.py
# (``_format_file_mutation_failure_footer``).  It's a UI affordance — reading
# "warning file mutation verifier, 2 files were NOT modified..." aloud is
# noise (#40772).  The footer is a ``⚠️ File-mutation verifier:`` header line
# followed by indented ``•`` bullet lines; strip the whole block.
_VERIFIER_FOOTER_RE = re.compile(
    r"^\s*⚠️?\s*File-mutation verifier:.*(?:\n[ \t]+•.*)*",
    flags=re.MULTILINE,
)


def strip_nonspoken_blocks(text: str) -> str:
    """Remove blocks that must never reach a speech provider.

    Currently: ``<think>`` reasoning blocks and the end-of-turn
    file-mutation verifier footer.
    """
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _THINK_BLOCK_OPEN_RE.sub(" ", text)
    text = _VERIFIER_FOOTER_RE.sub(" ", text)
    return text


def flatten_newlines_for_payload(text: str) -> str:
    """Collapse newlines into sentence breaks for single-line TTS payloads.

    Some OpenAI-compatible backends (e.g. Kokoro) truncate synthesis at the
    first newline (#9004).  The smoothing pass already terminates each line
    with punctuation, so newlines can safely become plain spaces.
    """
    if not text:
        return ""
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"(?<=[.!?;:,])\n", " ", text)
    text = text.replace("\n", ". ")
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def prepare_spoken_text(text: str, max_chars: int | None = 4000) -> str:
    """Return a TTS-friendly script from assistant text.

    Deterministic cleanup, not a semantic rewrite: it removes ``<think>``
    reasoning blocks and the file-mutation verifier footer, removes Markdown,
    expands common symbols such as a degree-Celsius sign to "degrees Celsius",
    turns visual line formatting into speakable sentence pauses, and flattens
    the result to a single line so newline-sensitive providers (Kokoro) speak
    the whole script.
    """
    spoken = strip_nonspoken_blocks(text)
    spoken = strip_markdown_for_tts(spoken)
    spoken = normalize_symbols_for_tts(spoken)
    spoken = smooth_whitespace_for_tts(spoken)
    spoken = flatten_newlines_for_payload(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken
