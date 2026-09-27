"""Deterministic cleanup turning assistant Markdown into a spoken script.

Shared by explicit TTS calls, gateway auto-TTS, voice-mode streaming and the web
dashboard. Non-ASCII characters are written as escapes on purpose so the file
stays free of invisible/look-alike glyphs.
"""

from __future__ import annotations

import functools
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
# Bare domains ("le site est hyperbio.fr") are as unspeakable as URLs: voices read
# "hyperbio point f r". Conservative on purpose: a short list of common TLDs so file
# names ("rapport.md", "script.py") survive, an alpha-led label so versions
# ("python 3.13") survive, and no match after "@" so e-mail addresses stay whole.
# Trailing sentence punctuation is kept, like _MEDIA_PATH_RE.
_BARE_DOMAIN_RE = re.compile(
    r"(?<![\w@./-])(?:[A-Za-z0-9-]+\.)*[A-Za-z][A-Za-z0-9-]+\.(?:fr|com|net|org|io|eu|co|dev|app|ai)"
    r"(?![\w@-])(?:/\S*?)?(?=[.,;:!?)\]]*(?:\s|$))")

_DEGREE_UNITS = (("C", "Celsius"), ("F", "Fahrenheit"))
# Unit suffix (regex, after a digit) -> spoken word; km/h variants before the bare "m".
_UNIT_WORDS = (
    (r"km\s*/\s*h", "kilometres per hour"), (r"km/h", "kilometres per hour"),
    (r"mm", "millimetres"), (r"cm", "centimetres"), (r"m", "metres"), (r"kg", "kilograms"))
# Currency prefix (regex) -> spoken word; order matters (NZ$/A$/US$ before bare $).
_CURRENCY_WORDS = (
    (r"NZ\$", "New Zealand dollars", re.IGNORECASE), (r"A\$", "Australian dollars", re.IGNORECASE),
    (r"US\$", "US dollars", re.IGNORECASE), ("€", "euros", 0), ("£", "pounds", 0), (r"\$", "dollars", 0),
)

# French expansions: English words injected into a French voice are read with a
# French accent ("kilometres per hour"), so a French voice gets French words.
# "°C" is plain "degr\u00e9s" (Celsius is implied in French speech); "km" before "m".
_DEGREE_UNITS_FR = (("C", ""), ("F", "Fahrenheit"))
_UNIT_WORDS_FR = (
    (r"km\s*/\s*h", "kilom\u00e8tres par heure"), (r"km", "kilom\u00e8tres"),
    (r"mm", "millim\u00e8tres"), (r"cm", "centim\u00e8tres"), (r"m", "m\u00e8tres"), (r"kg", "kilogrammes"))
_CURRENCY_WORDS_FR = (
    (r"NZ\$", "dollars n\u00e9o-z\u00e9landais", re.IGNORECASE), (r"A\$", "dollars australiens", re.IGNORECASE),
    (r"US\$", "dollars am\u00e9ricains", re.IGNORECASE), ("\u20ac", "euros", 0), ("\u00a3", "livres sterling", 0),
    (r"\$", "dollars", 0),
)
# French writes the symbol after the amount ("12 \u20ac"); English keeps prefix-only.
_SUFFIX_CURRENCY_WORDS_FR = (("\u20ac", "euros"), ("\u00a3", "livres sterling"), (r"\$", "dollars"))

# Per-language spoken words; an unknown or missing language keeps the English table,
# the historical behaviour every caller without a language still relies on.
_SPOKEN_WORDS = {
    "en": {"degrees": "degrees", "to": "to", "percent": "percent", "degree_units": _DEGREE_UNITS,
           "units": _UNIT_WORDS, "currencies": _CURRENCY_WORDS, "suffix_currencies": ()},
    "fr": {"degrees": "degr\u00e9s", "to": "\u00e0", "percent": "pour cent", "degree_units": _DEGREE_UNITS_FR,
           "units": _UNIT_WORDS_FR, "currencies": _CURRENCY_WORDS_FR,
           "suffix_currencies": _SUFFIX_CURRENCY_WORDS_FR},
}


def _spoken_words(spoken_lang: str | None) -> dict:
    """Word table for a language code ("fr", "fr-FR", "en"...); English when unknown."""
    primary = (spoken_lang or "").strip().lower().replace("_", "-").split("-")[0]
    return _SPOKEN_WORDS.get(primary, _SPOKEN_WORDS["en"])

# Broad emoji / pictograph cleanup: most voice providers read emojis as awkward labels.
_EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF☀-➿]+",
    flags=re.UNICODE)
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
    text = _BARE_DOMAIN_RE.sub("", text)
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


def _degree_phrase(words: dict, unit_word: str) -> str:
    return f"{words['degrees']} {unit_word}".rstrip()


def _normalize_temperature_ranges(text: str, words: dict | None = None) -> str:
    """``11-17°C`` -> ``11 to 17 degrees Celsius`` (en/em dash or hyphen; unicode minus normalized)."""
    words = words or _SPOKEN_WORDS["en"]
    number = r"([-+\u2212]?\d+(?:\.\d+)?)"
    for unit, word in words["degree_units"]:
        text = re.sub(
            r"(?<!\w)" + number + r"\s*[\u2013\u2014-]\s*" + number + r"\s*°\s*" + unit + r"\b",
            lambda m, phrase=_degree_phrase(words, word): (
                f"{m.group(1).replace(chr(0x2212), '-')} {words['to']} {m.group(2).replace(chr(0x2212), '-')} {phrase}"
            ),
            text, flags=re.IGNORECASE)
    return text


def normalize_symbols_for_tts(text: str, spoken_lang: str | None = None) -> str:
    """Expand common symbols/shorthand into words a TTS engine reads well, in the voice's
    language (``spoken_lang`` such as ``"fr"``; English when missing or unsupported)."""
    if not text:
        return ""
    words = _spoken_words(spoken_lang)
    degrees = words["degrees"]
    text = re.sub("[   ]", " ", str(text))  # non-breaking / thin spaces
    text = text.replace("\u2212", "-").replace("…", "...")  # minus sign, ellipsis
    text = _normalize_temperature_ranges(text, words)
    # Temperatures with a number first, then bare units ("measured in degrees C"),
    # then any remaining degree symbol (angles, stray cases).
    for unit, word in words["degree_units"]:
        text = re.sub(
            r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*" + unit + r"\b", r"\1 " + _degree_phrase(words, word), text,
            flags=re.IGNORECASE,
        )
    for unit, word in words["degree_units"]:
        text = re.sub(r"°\s*" + unit + r"\b", _degree_phrase(words, word), text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°", r"\1 " + degrees, text).replace("°", " " + degrees)
    # Common weather/travel units.
    for pattern, word in words["units"]:
        text = re.sub(r"(?<=\d)\s*" + pattern + r"\b", " " + word, text, flags=re.IGNORECASE)
    # Numeric rates only ("5/month" -> "5 per month").  Requiring digit-then-letter
    # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " per ", text)
    # Money and percentages. The integer part must END in a digit so a trailing
    # comma ("A$50, ...") is not swallowed into the spoken amount. Prefixed
    # currencies run first so "$" doesn't eat "NZ$".
    for symbol, word, flags in words["currencies"]:
        text = re.sub(symbol + r"\s*([\d,]*\d(?:\.\d+)?)", r"\1 " + word, text, flags=flags)
    for symbol, word in words["suffix_currencies"]:
        text = re.sub(r"(?<=\d)\s*" + symbol, " " + word, text)
    text = re.sub(r"(?<=\d)\s*%", " " + words["percent"], text)
    # Operators and separators that commonly leak from formatted answers.
    text = re.sub("[•◦▪▫]", " ", text.replace("&", " and "))  # bullet glyphs
    for symbol, word in (("→", " to "), ("⇒", " to "), ("≈", " about "), ("~", " about ")):
        text = text.replace(symbol, word)
    return _EMOJI_RE.sub("", _VARIATION_SELECTOR_RE.sub("", text))


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
                          (r"([,.;:!?])([A-Za-z])", r"\1 \2"), (r"\.{4,}", "...")):
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


def prepare_spoken_text(text: str, max_chars: int | None = 4000, *, spoken_lang: str | None = None) -> str:
    """Return a TTS-friendly script from assistant text (deterministic cleanup, not a rewrite).
    Pipeline: non-spoken blocks > Markdown > symbols/units > line formatting into sentence
    pauses > single line (for newline-sensitive providers), then ``max_chars``. ``spoken_lang``
    is the voice's language: units are expanded in it so a French voice never reads English."""
    spoken = text
    for step in (strip_nonspoken_blocks, strip_markdown_for_tts,
                 functools.partial(normalize_symbols_for_tts, spoken_lang=spoken_lang),
                 smooth_whitespace_for_tts, flatten_newlines_for_payload):
        spoken = step(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken


def _strip_markdown_for_tts(text: str) -> str:
    """``prepare_spoken_text`` without a length cap (``tts_tool`` compatibility name)."""
    return prepare_spoken_text(text, max_chars=None)
