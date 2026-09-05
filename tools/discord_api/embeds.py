"""Typed Discord embed builder, aligned to the Discord REST v10 API.

Feature M4 of the Discord Omniscience campaign (EPIC #79564): a safe, typed
outbound embed surface so an agent can attach rich embeds to Discord messages
without hand-rolling discord.py objects or trusting untrusted JSON.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence
from urllib.parse import urlparse

__all__ = [
    "EMBED_LIMITS",
    "EmbedField",
    "EmbedAuthor",
    "EmbedFooter",
    "Embed",
    "EmbedValidationError",
    "embed_to_plain_text",
    "MENTION_PATTERNS",
    "contains_mention",
    "validate_embeds",
]

EMBED_LIMITS = {
    "title": 256,
    "description": 4096,
    "field_name": 256,
    "field_value": 1024,
    "fields": 25,
    "footer_text": 2048,
    "author_name": 256,
    "total": 6000,
    "per_message": 10,
}

_ISO_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})$"
)
_BAD_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_PERCENT_HEX_RE = re.compile(r"%([0-9A-Fa-f]{2})")


class EmbedValidationError(ValueError):
    """Raised when an embed violates the typed Discord payload contract."""


def _check_len(value: Optional[str], limit: int, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EmbedValidationError(f"embed {what} must be a string, got {value!r}")
    if len(value) > limit:
        raise EmbedValidationError(
            f"embed {what} is {len(value)} chars, exceeds Discord limit {limit}"
        )
    return value


def _has_forbidden_url_chars(value: str) -> bool:
    # Literal control chars (Cc), bidi/format (Cf: RLO, ZWSP, ZWJ), and
    # backslash are forbidden. Unicode whitespace (Zs, via isspace) too.
    return any(
        ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf") for ch in value
    ) or "\\" in value


def _has_forbidden_percent_encoded(value: str) -> bool:
    """Reject percent-encoded control/whitespace/backslash (e.g. %0a, %0d%0a,
    %09, %00, %20, %5c) even though the literal chars are absent. Well-formed
    escapes otherwise pass, matching how Discord's clients treat them."""
    for m in _PERCENT_HEX_RE.finditer(value):
        code = int(m.group(1), 16)
        ch = chr(code)
        if ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf") or ch == "\\":
            return True
    return False


def _check_url(
    value: Optional[str],
    what: str,
    *,
    allow_attachment: bool = False,
) -> Optional[str]:
    """Validate an embed URL without allowing parser normalization to hide bad input.

    Link targets must use HTTP(S). Media/icon targets may additionally use
    Discord's ``attachment://filename`` reference form.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise EmbedValidationError(f"embed {what} must be a URL string, got {value!r}")
    if _has_forbidden_url_chars(value):
        raise EmbedValidationError(
            f"embed {what} contains forbidden whitespace/control/backslash characters, "
            f"got {value!r}"
        )
    if _BAD_PERCENT_RE.search(value):
        raise EmbedValidationError(
            f"embed {what} contains malformed percent encoding, got {value!r}"
        )
    if _has_forbidden_percent_encoded(value):
        raise EmbedValidationError(
            f"embed {what} contains percent-encoded whitespace/control/backslash "
            f"characters, got {value!r}"
        )

    try:
        parsed = urlparse(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        # Accessing .port performs numeric/range validation and can raise.
        parsed.port
    except ValueError as exc:
        raise EmbedValidationError(
            f"embed {what} is not a structurally valid URL, got {value!r}"
        ) from exc

    # Reject any userinfo (username/password) — credentials in a URL are never
    # legitimate here and open host-confusion/phishing (e.g. evil.com@good.com).
    if parsed.username is not None or "@" in (parsed.netloc or ""):
        raise EmbedValidationError(
            f"embed {what} must not contain userinfo/credentials, got {value!r}"
        )

    if scheme in ("http", "https"):
        if not hostname:
            raise EmbedValidationError(
                f"embed {what} must have a non-empty hostname, got {value!r}"
            )
        # Reject trailing garbage after a bracketed IPv6 literal (e.g. "[::1]x"):
        # urlparse silently strips it, hiding a malformed authority.
        if "[" in parsed.netloc:
            bracket_end = parsed.netloc.find("]")
            if bracket_end == -1:
                raise EmbedValidationError(
                    f"embed {what} has an unclosed IPv6 literal, got {value!r}"
                )
            after = parsed.netloc[bracket_end + 1:]
            if after and not after.startswith(":"):
                raise EmbedValidationError(
                    f"embed {what} has garbage after the IPv6 literal, got {value!r}"
                )
        return value

    if allow_attachment and scheme == "attachment":
        filename = parsed.netloc
        if (
            not filename
            or filename in (".", "..")
            or "/" in filename
            or "\\" in filename
            or not re.fullmatch(r"[A-Za-z0-9._-]+", filename)
            or parsed.path
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise EmbedValidationError(
                f"embed {what} must use attachment://filename, got {value!r}"
            )
        return value

    allowed = "http(s) or attachment" if allow_attachment else "http(s)"
    raise EmbedValidationError(f"embed {what} must use {allowed}, got {value!r}")


@dataclass(frozen=True)
class EmbedField:
    """A single embed field (name/value/inline)."""

    name: str
    value: str
    inline: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _check_len(self.name, EMBED_LIMITS["field_name"], "field name"),
        )
        object.__setattr__(
            self,
            "value",
            _check_len(self.value, EMBED_LIMITS["field_value"], "field value"),
        )
        if type(self.inline) is not bool:
            raise EmbedValidationError(
                f"embed field inline must be a bool, got {self.inline!r}"
            )


@dataclass(frozen=True)
class EmbedAuthor:
    """Embed author line (name <=256, optional url/icon)."""

    name: str
    url: Optional[str] = None
    icon_url: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _check_len(self.name, EMBED_LIMITS["author_name"], "author name"),
        )
        object.__setattr__(self, "url", _check_url(self.url, "author url"))
        object.__setattr__(
            self,
            "icon_url",
            _check_url(self.icon_url, "author icon_url", allow_attachment=True),
        )


@dataclass(frozen=True)
class EmbedFooter:
    """Embed footer (text <=2048, optional icon)."""

    text: str
    icon_url: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _check_len(self.text, EMBED_LIMITS["footer_text"], "footer text"),
        )
        object.__setattr__(
            self,
            "icon_url",
            _check_url(self.icon_url, "footer icon_url", allow_attachment=True),
        )


@dataclass(frozen=True)
class Embed:
    """A validated Discord embed with an immutable external field surface."""

    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    color: Optional[int] = None
    timestamp: Optional[str] = None
    author: Optional[EmbedAuthor] = None
    footer: Optional[EmbedFooter] = None
    fields: Sequence[EmbedField] = field(default_factory=tuple)
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "title",
            _check_len(self.title, EMBED_LIMITS["title"], "title"),
        )
        object.__setattr__(
            self,
            "description",
            _check_len(self.description, EMBED_LIMITS["description"], "description"),
        )
        object.__setattr__(self, "url", _check_url(self.url, "url"))
        object.__setattr__(
            self,
            "image_url",
            _check_url(self.image_url, "image url", allow_attachment=True),
        )
        object.__setattr__(
            self,
            "thumbnail_url",
            _check_url(self.thumbnail_url, "thumbnail url", allow_attachment=True),
        )

        if self.author is not None and not isinstance(self.author, EmbedAuthor):
            raise EmbedValidationError(
                f"embed author must be EmbedAuthor, got {self.author!r}"
            )
        if self.footer is not None and not isinstance(self.footer, EmbedFooter):
            raise EmbedValidationError(
                f"embed footer must be EmbedFooter, got {self.footer!r}"
            )

        if self.timestamp is not None:
            if not isinstance(self.timestamp, str) or not _ISO_TS_RE.match(self.timestamp):
                raise EmbedValidationError(
                    f"embed timestamp must be ISO-8601, got {self.timestamp!r}"
                )
            normalized = self.timestamp
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            # Reject out-of-range UTC-offset fields before fromisoformat
            # normalizes them (e.g. "+12:60" silently becomes "+13:00").
            _tz = re.search(r"([+-])(\d{2}):(\d{2})$", normalized)
            if _tz:
                _oh, _om = int(_tz.group(2)), int(_tz.group(3))
                if _oh > 23 or _om > 59:
                    raise EmbedValidationError(
                        f"embed timestamp has an out-of-range UTC offset, "
                        f"got {self.timestamp!r}"
                    )
            try:
                datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise EmbedValidationError(
                    f"embed timestamp is not a valid date/time, got {self.timestamp!r}"
                ) from exc

        if self.color is not None:
            if type(self.color) is not int or not (0 <= self.color <= 0xFFFFFF):
                raise EmbedValidationError(
                    f"embed color must be a 24-bit int, got {self.color!r}"
                )

        try:
            immutable_fields = tuple(self.fields)
        except TypeError as exc:
            raise EmbedValidationError(
                f"embed fields must be a sequence of EmbedField, got {self.fields!r}"
            ) from exc
        if any(not isinstance(item, EmbedField) for item in immutable_fields):
            raise EmbedValidationError("embed fields must contain only EmbedField values")
        object.__setattr__(self, "fields", immutable_fields)

        if len(self.fields) > EMBED_LIMITS["fields"]:
            raise EmbedValidationError(
                f"embed has {len(self.fields)} fields, exceeds Discord limit "
                f"{EMBED_LIMITS['fields']}"
            )
        total = self._total_chars()
        if total > EMBED_LIMITS["total"]:
            raise EmbedValidationError(
                f"embed is {total} total chars, exceeds Discord limit "
                f"{EMBED_LIMITS['total']}"
            )

    def _total_chars(self) -> int:
        total = len(self.title or "") + len(self.description or "")
        total += len(self.author.name) if self.author else 0
        total += len(self.footer.text) if self.footer else 0
        total += sum(len(item.name) + len(item.value) for item in self.fields)
        return total

    def to_payload(self) -> dict:
        """Render to the Discord REST embed object."""
        payload: dict = {}
        for key in ("title", "description", "url", "color", "timestamp"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value

        if self.author is not None:
            author: dict = {"name": self.author.name}
            if self.author.url:
                author["url"] = self.author.url
            if self.author.icon_url:
                author["icon_url"] = self.author.icon_url
            payload["author"] = author

        if self.footer is not None:
            footer: dict = {"text": self.footer.text}
            if self.footer.icon_url:
                footer["icon_url"] = self.footer.icon_url
            payload["footer"] = footer

        if self.image_url:
            payload["image"] = {"url": self.image_url}
        if self.thumbnail_url:
            payload["thumbnail"] = {"url": self.thumbnail_url}
        if self.fields:
            payload["fields"] = [
                {"name": item.name, "value": item.value, "inline": item.inline}
                for item in self.fields
            ]
        return payload


def validate_embeds(embeds: Sequence[Embed]) -> None:
    """Enforce Discord's per-message count and aggregate-character limits."""
    try:
        immutable_embeds = tuple(embeds)
    except TypeError as exc:
        raise EmbedValidationError("embeds must be a sequence of Embed values") from exc
    if any(not isinstance(embed, Embed) for embed in immutable_embeds):
        raise EmbedValidationError("embeds must contain only Embed values")
    if len(immutable_embeds) > EMBED_LIMITS["per_message"]:
        raise EmbedValidationError(
            f"message has {len(immutable_embeds)} embeds, exceeds Discord limit "
            f"{EMBED_LIMITS['per_message']}"
        )
    aggregate = sum(embed._total_chars() for embed in immutable_embeds)
    if aggregate > EMBED_LIMITS["total"]:
        raise EmbedValidationError(
            f"message embed content is {aggregate} chars, exceeds Discord "
            f"per-message limit {EMBED_LIMITS['total']}"
        )


MENTION_PATTERNS = (
    re.compile(r"@everyone"),
    re.compile(r"@here"),
    re.compile(r"<@!?[0-9]+>"),
    re.compile(r"<@&[0-9]+>"),
)


def contains_mention(text: str) -> bool:
    """Return True when ``text`` carries a Discord mention that can ping."""
    if not isinstance(text, str):
        raise EmbedValidationError(f"mention text must be a string, got {text!r}")
    return any(pattern.search(text) for pattern in MENTION_PATTERNS)


def embed_to_plain_text(embed: Embed) -> str:
    """Render an embed's user-visible content as a plain Markdown fallback."""
    if not isinstance(embed, Embed):
        raise EmbedValidationError(f"fallback input must be Embed, got {embed!r}")

    lines: List[str] = []
    if embed.author:
        if embed.author.url:
            lines.append(f"**[{embed.author.name}]({embed.author.url})**")
        else:
            lines.append(f"**{embed.author.name}**")

    if embed.title:
        if embed.url:
            lines.append(f"# [{embed.title}]({embed.url})")
        else:
            lines.append(f"# {embed.title}")
    elif embed.url:
        lines.append(embed.url)

    if embed.description:
        lines.append(embed.description)
    for item in embed.fields:
        lines.append(f"**{item.name}:** {item.value}")
    if embed.timestamp:
        lines.append(f"`{embed.timestamp}`")
    if embed.image_url:
        lines.append(embed.image_url)
    if embed.thumbnail_url:
        lines.append(embed.thumbnail_url)
    if embed.footer:
        lines.append(f"_{embed.footer.text}_")

    return "\n".join(lines)
