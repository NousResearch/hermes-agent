"""Deterministic cleanup turning assistant Markdown into a spoken script.

Shared by explicit TTS calls, gateway auto-TTS, voice-mode streaming and the web
dashboard. Non-ASCII characters are written as escapes on purpose so the file
stays free of invisible/look-alike glyphs.
"""

from __future__ import annotations

import html
import re

# Sentinel appended to former heading lines so smooth_whitespace_for_tts folds the
# heading into the sentence after it ("Weather, it will be sunny") instead of a bare
# "Weather." label.
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
# Local file links ("MEDIA:/Users/me/file.xlsx") are click targets on screen, not
# speech: voices loop on the hyphenated slug ("eeeeee"). The token is silence; the
# assistant's prose already says "the files are below". Trailing sentence
# punctuation is left in place so "see MEDIA:/x.py. Then" keeps its full stop.
_MEDIA_PATH_RE = re.compile(r"MEDIA:\S+?(?=[.,;:!?)\]]*(?:\s|$))")

_DEGREE_UNITS = (("C", "Celsius"), ("F", "Fahrenheit"))
# Unit suffix (regex, after a digit) -> spoken word; km/h variants before the bare "m".
_UNIT_WORDS = (
    (r"km\s*/\s*h", "kilometres per hour"), (r"km/h", "kilometres per hour"),
    (r"mm", "millimetres"), (r"cm", "centimetres"), (r"m", "metres"))
# Currency prefix (regex) -> spoken word; order matters (NZ$/A$/US$ before bare $).
_CURRENCY_WORDS = (
    (r"NZ\$", "New Zealand dollars", re.IGNORECASE), (r"A\$", "Australian dollars", re.IGNORECASE),
    (r"US\$", "US dollars", re.IGNORECASE), ("€", "euros", 0), ("£", "pounds", 0), (r"\$", "dollars", 0),
)

# Broad emoji / pictograph cleanup: most voice providers read emojis as awkward labels.
_EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF☀-➿]+",
    flags=re.UNICODE)
_VARIATION_SELECTOR_RE = re.compile("[︎️]")


# Opt-in command-provider fallback, not an SSML synthesizer. Match complete
# XML-style tags, including quoted '>' in attributes, without eating comparisons.
_SPEECH_CUE_TAG_RE = re.compile(
    r"</?(?P<name>[A-Za-z][\w:.-]*)(?=[\s/>])"
    r"(?:[^<>\"']|\"[^\"]*\"|'[^']*')*/?>"
)
_SPEECH_AUDIO_CUE_RE = re.compile(
    r"\[(?:/?(?:pause|break|emphasis|whisper|whispers|laugh|laughs|sigh|sighs|"
    r"excited|excitedly|slow|very slow|fast))\]", re.IGNORECASE
)


def prepare_command_speech_cues(text: str) -> str:
    """Plain-text fallback for opted-in command engines: breaks become commas,
    emphasis keeps its words, complete XML/audio tags are not sent to speech.
    No duration or prosody guarantee: those belong to the configured engine.
    Run before generic cleanup, which otherwise mangles closing tags as paths.
    """
    text = html.unescape(text)

    def replace_tag(match: re.Match[str]) -> str:
        opening = not match.group(0).startswith("</")
        return ", " if opening and match.group("name").lower() == "break" else ""

    text = _SPEECH_CUE_TAG_RE.sub(replace_tag, text)
    return _SPEECH_AUDIO_CUE_RE.sub(
        lambda m: ", " if m.group(0).lower() in {"[pause]", "[break]"} else "", text
    )


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
    # Mark headings (do not just delete the marker): see _HEAD.
    text = _MD_HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip() + _HEAD, text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_ITEM_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)
    # Leftover table pipes become pauses instead of a spoken "vertical bar".
    return _MD_TABLE_PIPE_RE.sub("; ", text)


def _normalize_temperature_ranges(text: str) -> str:
    """``11-17°C`` -> ``11 to 17 degrees Celsius`` (en/em dash or hyphen; unicode minus normalized)."""
    number = r"([-+\u2212]?\d+(?:\.\d+)?)"
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)" + number + r"\s*[\u2013\u2014-]\s*" + number + r"\s*°\s*" + unit + r"\b",
            lambda m, w=word: (
                f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees {w}"
            ),
            text, flags=re.IGNORECASE)
    return text


def normalize_symbols_for_tts(text: str) -> str:
    """Expand common symbols/shorthand into words a TTS engine reads well."""
    if not text:
        return ""
    text = re.sub("[   ]", " ", str(text))  # non-breaking / thin spaces
    text = text.replace("\u2212", "-").replace("…", "...")  # minus sign, ellipsis
    text = _normalize_temperature_ranges(text)
    # Temperatures with a number first, then bare units ("measured in degrees C"),
    # then any remaining degree symbol (angles, stray cases).
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*" + unit + r"\b", r"\1 degrees " + word, text, flags=re.IGNORECASE,
        )
    for unit, word in _DEGREE_UNITS:
        text = re.sub(r"°\s*" + unit + r"\b", "degrees " + word, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°", r"\1 degrees", text).replace("°", " degrees")
    # Common weather/travel units.
    for pattern, word in _UNIT_WORDS:
        text = re.sub(r"(?<=\d)\s*" + pattern + r"\b", " " + word, text, flags=re.IGNORECASE)
    # Numeric rates only ("5/month" -> "5 per month").  Requiring digit-then-letter
    # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " per ", text)
    # Money and percentages. The integer part must END in a digit so a trailing
    # comma ("A$50, ...") is not swallowed into the spoken amount. Prefixed
    # currencies run first so "$" doesn't eat "NZ$".
    for symbol, word, flags in _CURRENCY_WORDS:
        text = re.sub(symbol + r"\s*([\d,]*\d(?:\.\d+)?)", r"\1 " + word, text, flags=flags)
    text = re.sub(r"(?<=\d)\s*%", " percent", text)
    # Operators and separators that commonly leak from formatted answers.
    text = re.sub("[•◦▪▫]", " ", text.replace("&", " and "))  # bullet glyphs
    for symbol, word in (("→", " to "), ("⇒", " to "), ("≈", " about "), ("~", " about ")):
        text = text.replace(symbol, word)
    return _EMOJI_RE.sub("", _VARIATION_SELECTOR_RE.sub("", text))


# --- Identifier-dense tokens (#119207) ---------------------------------------
# Filenames with extensions, hashes, UUIDs, dense model/version IDs and paths
# are read character-by-character ("peyton-sample-20260922.wav" becomes
# "peyton dash sample dash two zero two six..."), making clean synthesized
# audio sound corrupted. They become silence, never a hardcoded English
# placeholder word (#86602: the reply may be in any language). Ordinary
# prose, dates, ratios and short hyphenated words (COVID-19, well-known)
# pass through untouched. Mirrored token-for-token by
# apps/desktop/src/lib/speech-text.ts — keep both in lockstep and extend the
# shared corpus (tests/fixtures/identifier_speech_corpus.json) first.

_FILENAME_EXT_RE = re.compile(
    r"[\w.-]{0,60}\.(?:wav|ogg|mp3|flac|m4a|aac|py|pyc|ts|tsx|js|jsx|mjs|cjs|json|yaml|yml|toml|"
    r"md|mdx|txt|csv|xlsx|xls|pdf|png|jpg|jpeg|gif|webp|svg|log|sql|sh|bash|zsh|rs|go|java|rb|php|"
    r"html|css|lock|tar|gz|zip|db|sqlite|sqlite3|onnx|pt|bin|env|ini|conf|cfg|xml)\b",
    re.IGNORECASE)
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HASH_PREFIX_HEX_RE = re.compile(
    r"\b(?:sha(?:-?256|-?512|-?1|3)?|blake2[ab]?|md5|crc32?)[:\s]+[0-9a-fA-F]{7,64}", re.IGNORECASE)
_HEX_RUN_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{7,64}(?![0-9a-fA-F])")
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9_./~@-]+")
_DATE_TOKEN_RE = re.compile(r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?")


def _is_dense_identifier(token: str) -> bool:
    """True for a machine token a voice would spell out character by character."""
    if "@" in token or _DATE_TOKEN_RE.fullmatch(token):
        return False  # email addresses, dates ("2026-09-28", "2026/06/02")
    if token.startswith(("~/", "./", "../", "/")):
        return True  # filesystem paths
    if "/" in token and (_FILENAME_EXT_RE.search(token) or any(char.isdigit() for char in token)):
        return True  # paths and dense model IDs ("meta-llama/Llama-3.3-70B-Instruct")
    if _FILENAME_EXT_RE.search(token) or _UUID_RE.search(token):
        return True
    # Hex-hash runs ("73688014f78", "e3b0c442...") — digit-free runs
    # ("defaced") are legitimate words and stay.
    if _HEX_RUN_RE.fullmatch(token) and re.search(r"\d", token):
        return True
    digits = any(char.isdigit() for char in token)
    if not digits:
        return False
    seps = sum(token.count(char) > 0 for char in "_./")
    hyphens = token.count("-")
    if seps >= 2 or hyphens >= 2:
        return True  # v2.1.0-beta.3, Llama-3.3-70B, dated filename slugs
    return "/" in token or (hyphens >= 1 and seps >= 1)


def prune_identifier_tokens_for_tts(text: str) -> str:
    """Silence identifier-dense tokens; keep ordinary prose verbatim."""
    if not text:
        return ""
    text = _HASH_PREFIX_HEX_RE.sub(" ", text)
    return _IDENTIFIER_TOKEN_RE.sub(
        lambda m: " " if _is_dense_identifier(m.group(0)) else m.group(0), text)


def smooth_whitespace_for_tts(text: str) -> str:
    """Collapse visual formatting into calm spoken paragraphs. A _HEAD-marked heading folds into
    the next content line as a lead-in ("Weather, It will be sunny."); a heading with no content
    after it becomes its own sentence."""
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
            # Hold a pending heading across blank lines so it still folds into the next content line.
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
                # A trailing colon promises a continuation ("the list:"); the raw
                # token after it (file link, code fence) is already gone from
                # speech, so the voice would hang on the open pause. Close it.
                line = line.rstrip(":;").rstrip() + "."
            elif line[-1] not in ".!?":
                line += "."
        lines.append(line)
    flush_pending()
    text = "\n".join(lines)
    for pattern, repl in ((r"\n{3,}", "\n\n"), (r"[ \t]{2,}", " "), (r"\s+([,.;:!?])", r"\1"),
                          # A sentence mark only gains a space when NOT glued to
                          # a word char on both sides — "user@example.com" is a
                          # single token, not "example. com".
                          (r"(?<![A-Za-z0-9])[,.;:!?](?=[A-Za-z])", r"\g<0> "),
                          (r"\.{4,}", "...")):
        text = re.sub(pattern, repl, text)
    # Single-line text ("Here is the list:") skips the per-line pause pass, so close a
    # final colon here too -- never a digit-preceded one ("final score 3:2" is a ratio).
    text = re.sub(r"([^0-9])(?:[:;]\s*)+$", r"\1.", text)
    return text.strip()


# ``/reasoning show`` emits ``<think>...</think>`` in the final message: users want to
# SEE reasoning, not hear it. An unterminated block (streaming cut-off) is also silenced.
# Reasoning blocks: models with ``/reasoning show`` enabled emit ``<think>...</think>`` blocks in the final
# assistant message. See #34213.
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_OPEN_RE = re.compile(r"<think[\s>].*\Z", flags=re.DOTALL | re.IGNORECASE)

# run_agent.py's turn-end file-mutation verifier footer (a ``⚠️ File-mutation verifier:``
# header line plus indented ``•`` bullets) is a UI affordance, not speech.
_VERIFIER_FOOTER_RE = re.compile(r"^\s*⚠️?\s*File-mutation verifier:.*(?:\n[ \t]+•.*)*", flags=re.MULTILINE)


def strip_nonspoken_blocks(text: str) -> str:
    """Remove ``<think>`` reasoning blocks and the file-mutation verifier footer."""
    if not text:
        return ""
    for pattern in (_THINK_BLOCK_RE, _THINK_BLOCK_OPEN_RE, _VERIFIER_FOOTER_RE):
        text = pattern.sub(" ", text)
    return text


def flatten_newlines_for_payload(text: str) -> str:
    """Collapse newlines into sentence breaks for single-line TTS payloads: some OpenAI-compatible
    backends (e.g. Kokoro) truncate at the first newline; smoothing already ends each line with
    punctuation, so this is safe.

    See #9004.
    """
    if not text:
        return ""
    for pattern, repl in ((r"\n{2,}", ". "), (r"(?<=[.!?;:,])\n", " "), (r"\n", ". "), (r"\.\s*\.", "."),
                          (r"[ \t]{2,}", " ")):
        text = re.sub(pattern, repl, text)
    # Single-line text ("Here is the list:") skips the per-line pause pass, so close a
    # final colon here too -- never a digit-preceded one ("final score 3:2" is a ratio).
    text = re.sub(r"([^0-9])(?:[:;]\s*)+$", r"\1.", text)
    return text.strip()


def prepare_spoken_text(text: str, max_chars: int | None = 4000) -> str:
    """Return a TTS-friendly script from assistant text (deterministic cleanup, not a rewrite).
    Pipeline: non-spoken blocks > Markdown > symbols/units > identifier-dense tokens >
    line formatting into sentence pauses > single line (for newline-sensitive providers),
    then ``max_chars``."""
    spoken = text
    for step in (strip_nonspoken_blocks, strip_markdown_for_tts, normalize_symbols_for_tts,
                 prune_identifier_tokens_for_tts,
                 smooth_whitespace_for_tts, flatten_newlines_for_payload):
        spoken = step(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken


def _strip_markdown_for_tts(text: str) -> str:
    """``prepare_spoken_text`` without a length cap (``tts_tool`` compatibility name)."""
    return prepare_spoken_text(text, max_chars=None)
