"""Shared URL-intent guard for browser/web side-effect tools.

A URL in the current user turn is not permission to fetch or navigate it. This
module intentionally has no imports from model_tools or tool implementations so
it can be used from the model dispatcher and low-level tool entry points without
circular imports or fail-open fallback paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

URL_ACTION_TOOLS = {"browser_navigate", "web_extract"}

_URL_RE = re.compile(
    r"https?://[^\s)\]>\"'`]+|"
    r"\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3})(?::\d+)?(?:/[^\s)\]>\"'`]*)?|"
    r"\b[a-z0-9][a-z0-9.-]+\.[a-z]{2,}(?::\d+)?(?:/[^\s)\]>\"'`]*)?",
    re.I,
)
_ENGLISH_EXPLICIT_URL_ACTION_RE = re.compile(
    r"\b(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)\b",
    re.I,
)
_EXPLICIT_URL_ACTION_RE = re.compile(
    r"("
    r"열어(?:줘|봐|주세요)|접속(?:해|해줘|해주세요|시켜|시켜줘)|들어가(?:줘|봐|주세요)|"
    r"확인(?:해|해줘|해주세요)|검증(?:해|해줘|해주세요)|"
    r"캡처(?:해|해줘|해주세요)|스크린샷\s*(?:찍어|찍어줘|캡처|해줘)|증거\s*(?:(?:확보|수집)(?:해|해줘|해주세요)?|봐줘)|"
    r"봐줘|살펴(?:봐|봐줘|줘)|분석(?:해|해줘|해주세요)|사용(?:해|해줘|해주세요)|"
    r"써줘|요약(?:해|해줘|해주세요)|읽어(?:줘|봐|주세요)|읽고\s*(?:요약|정리|분석|설명)(?:해|해줘|해주세요)?|가져와(?:줘|주세요)|추출(?:해|해줘|해주세요)|"
    r"\b(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)\b"
    r")",
    re.I,
)
_NEGATED_URL_ACTION_RE = re.compile(
    r"("
    r"열지\s*마|읽지\s*마|들어가지\s*마|접속하지\s*마|가져오지\s*마|추출하지\s*마|"
    r"요약하지\s*마|확인하지\s*마|검증하지\s*마|사용하지\s*마|"
    r"접속\s*(?:안\s*함|금지)|"
    r"do\s+not\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)|"
    r"don['’]?t\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)|"
    r"won['’]?t\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)|"
    r"no\s+need\s+to\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)|"
    r"not\s+necessary\s+to\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)|"
    r"never\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)|"
    r"will\s+not\s+(?:open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract)"
    r")",
    re.I,
)
_PASSIVE_PASTE_MARKER_RE = re.compile(r"복사|cop(?:y|ied)|메모|기록|후보", re.I)
_PASSIVE_MARKER_IMMEDIATELY_BEFORE_URL_RE = re.compile(
    r"(?:복사(?:해놓음|해둠|해놨음|함|만)?|cop(?:y|ied)|메모|기록|후보)\s*[:：-]?\s*$",
    re.I,
)
_OTHER_OBJECT_AFTER_PASTE_RE = re.compile(
    r"(?:(?:내|다른)\s*(?:코드|브라우저|파일|문서|앱)|메뉴|브라우저|결과)",
    re.I,
)
_ACTION_TO_URL_PREFIX_RE = re.compile(
    r"(?:^|[\s:：;,.!?。！？])(?:"
    r"open|browse|visit|navigate|fetch|check|verify|screenshot|use|summari[sz]e|read|extract"
    r")\s*$",
    re.I,
)

AMBIGUOUS_PASTED_URL_ERROR = (
    "Ambiguous pasted URL: the current user message contains a URL but does not "
    "explicitly ask to open, browse, verify, summarize, read, or use it. Do not "
    "navigate or fetch it. Ask a concise clarification first if using that URL "
    "is necessary."
)
NON_HTTP_SCHEME_URL_ERROR = (
    "Blocked non-http URL action: browser/web URL tools must not navigate or "
    "fetch file:, javascript:, data:, or other non-http schemes from this path."
)
MALFORMED_URL_ERROR = (
    "Blocked malformed URL action: browser/web URL tools require a valid http(s) "
    "URL or host before navigation/fetch."
)
MISSING_USER_INTENT_CONTEXT_ERROR = (
    "Missing user intent context: URL navigation/fetch tools require the current "
    "user message so Hermes can distinguish a passive pasted URL from an explicit "
    "instruction. Do not navigate or fetch the URL from this path."
)


@dataclass(frozen=True)
class UrlIntentTarget:
    """Normalized URL-ish target for intent comparison.

    Matching is host-based on purpose: if the user merely pasted ``example.com``,
    navigating to ``https://example.com/path`` is still using that pasted URL.
    """

    host: str
    port: int | None = None
    path: str = ""


@dataclass(frozen=True)
class UrlIntentMention:
    """URL-like mention plus its span in the original user text."""

    target: UrlIntentTarget
    start: int
    end: int


def _strip_urlish_punctuation(value: str) -> str:
    return str(value or "").strip().strip("<>(){}.,;:'\"`")


def _normalize_target(value: str) -> Optional[UrlIntentTarget]:
    text = _strip_urlish_punctuation(value)
    if not text:
        return None
    scheme_match = re.match(r"^([a-z][a-z0-9+.-]*):", text, flags=re.I)
    looks_like_host_port = bool(re.match(r"^[a-z0-9.-]+:\d+(?:/|$)", text, flags=re.I))
    if scheme_match and not looks_like_host_port and not text.lower().startswith(("http://", "https://")):
        return UrlIntentTarget(host=f"__non_http_scheme__:{scheme_match.group(1).lower()}")
    if not re.match(r"^[a-z][a-z0-9+.-]*://", text, flags=re.I):
        text = f"http://{text}"
    try:
        parsed = urlsplit(text)
        host = (parsed.hostname or "").lower().strip(".")
    except ValueError:
        return UrlIntentTarget(host="__malformed_url__")
    if not host:
        if text.lower().startswith(("http://", "https://")):
            return UrlIntentTarget(host="__malformed_url__")
        return None
    path = parsed.path or ""
    if path == "/":
        path = ""
    try:
        port = parsed.port
    except ValueError:
        return UrlIntentTarget(host="__malformed_url__")
    return UrlIntentTarget(host=host, port=port, path=path)


def _normalize_host(value: str) -> Optional[str]:
    target = _normalize_target(value)
    return target.host if target else None


def extract_url_intent_targets(text: str) -> set[UrlIntentTarget]:
    """Extract URL-like hosts from free text for current-turn intent matching."""

    if not text:
        return set()
    targets: set[UrlIntentTarget] = set()
    for match in _URL_RE.finditer(text):
        target = _normalize_target(match.group(0))
        if target:
            targets.add(target)
    return targets


def extract_url_intent_mentions(text: str) -> list[UrlIntentMention]:
    """Extract URL-like mentions with spans for URL-scoped action matching."""

    if not text:
        return []
    mentions: list[UrlIntentMention] = []
    for match in _URL_RE.finditer(text):
        target = _normalize_target(match.group(0))
        if target:
            mentions.append(UrlIntentMention(
                target=target,
                start=match.start(),
                end=match.end(),
            ))
    return mentions


def _target_from_tool_value(value: str) -> set[UrlIntentTarget]:
    """Extract a target from a tool argument, including non-http schemes."""

    text = str(value or "")
    target = _normalize_target(text)
    if target:
        return {target}
    return extract_url_intent_targets(text)


def tool_target_url_intents(function_name: str, function_args: Mapping[str, Any] | None) -> set[UrlIntentTarget]:
    """Extract target URL hosts from a URL-action tool's arguments."""

    args = function_args or {}
    if function_name == "browser_navigate":
        return _target_from_tool_value(str(args.get("url", "")))
    if function_name == "web_extract":
        urls = args.get("urls") or args.get("url") or ""
        if isinstance(urls, list):
            values: set[UrlIntentTarget] = set()
            for item in urls:
                values.update(_target_from_tool_value(str(item)))
            return values
        return _target_from_tool_value(str(urls))
    return set()


def _canonical_www_host(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _hosts_overlap(left: str, right: str) -> bool:
    """Return True for the exact same host, treating only www. as cosmetic."""

    if left.startswith("__non_http_scheme__:") or right.startswith("__non_http_scheme__:"):
        return left == right
    return _canonical_www_host(left) == _canonical_www_host(right)


def _paths_compatible(left: str, right: str) -> bool:
    return (left or "") == (right or "")


def _targets_compatible(left: UrlIntentTarget, right: UrlIntentTarget) -> bool:
    if not _hosts_overlap(left.host, right.host):
        return False
    if left.port != right.port:
        return False
    return _paths_compatible(left.path, right.path)


def _target_sets_overlap(left: set[UrlIntentTarget], right: set[UrlIntentTarget]) -> bool:
    for a in left:
        for b in right:
            if _hosts_overlap(a.host, b.host):
                return True
    return False


def _action_context_for_mention(text: str, mentions: list[UrlIntentMention], index: int) -> str:
    """Return the text segment whose action words apply to one URL mention.

    The URL itself plus the text after it up to the next URL belongs to that
    URL. For any URL, an immediately adjacent imperative prefix like
    ``open example.com`` is included. For later URLs this prefix starts after
    the previous URL, so it cannot authorize both sides.
    """

    mention = mentions[index]
    next_start = mentions[index + 1].start if index + 1 < len(mentions) else len(text)
    suffix = text[mention.end:next_start]
    prefix_start = 0 if index == 0 else mentions[index - 1].end
    prefix = text[prefix_start:mention.start]
    if _ACTION_TO_URL_PREFIX_RE.search(prefix):
        return prefix + suffix
    return suffix


def _has_action_prefix_for_mention(text: str, mentions: list[UrlIntentMention], index: int) -> bool:
    prefix_start = 0 if index == 0 else mentions[index - 1].end
    return bool(_ACTION_TO_URL_PREFIX_RE.search(text[prefix_start:mentions[index].start]))


def _has_passive_marker_after_mention(text: str, mentions: list[UrlIntentMention], index: int) -> bool:
    next_start = mentions[index + 1].start if index + 1 < len(mentions) else len(text)
    return bool(_PASSIVE_PASTE_MARKER_RE.search(text[mentions[index].end:next_start]))


def _has_passive_marker_before_mention(text: str, mentions: list[UrlIntentMention], index: int) -> bool:
    prefix_start = 0 if index == 0 else mentions[index - 1].end
    prefix = text[prefix_start:mentions[index].start]
    return bool(_PASSIVE_MARKER_IMMEDIATELY_BEFORE_URL_RE.search(prefix)) and not _has_action_prefix_for_mention(text, mentions, index)


def _literal_mentions_for_tool_targets(text: str, targets: set[UrlIntentTarget]) -> list[UrlIntentMention]:
    """Recover exact target-host mentions that the generic URL regex misses.

    This covers IDN hosts, bare IPv6 literals, and single-label local hostnames
    without teaching the broad URL regex to classify arbitrary words as URLs.
    """

    lowered = text.lower()
    mentions: list[UrlIntentMention] = []
    for target in targets:
        if target.host.startswith("__"):
            continue
        needle = target.host.lower()
        if not needle:
            continue
        start = 0
        while True:
            found = lowered.find(needle, start)
            if found == -1:
                break
            end = found + len(needle)
            before = lowered[found - 1] if found > 0 else ""
            after = lowered[end] if end < len(lowered) else ""
            before_is_inner = bool(before) and (before.isalnum() or before in ".-_:")
            after_is_inner = bool(after) and (after.isalnum() or after in ".-_:")
            if not (before_is_inner or after_is_inner):
                mentions.append(UrlIntentMention(UrlIntentTarget(host=target.host), found, end))
            start = end
    return mentions


def ambiguous_user_pasted_url_block(
    function_name: str,
    function_args: Mapping[str, Any] | None,
    user_task: Any,
    *,
    require_user_task: bool = True,
) -> Optional[str]:
    """Return a block reason for passive pasted URL use, else ``None``.

    ``require_user_task=True`` is fail-closed for live tool paths: a URL action
    without current-turn context cannot prove that the user explicitly asked for
    the URL to be fetched.
    """

    if function_name not in URL_ACTION_TOOLS:
        return None

    target_urls = tool_target_url_intents(function_name, function_args)
    if not target_urls:
        return MALFORMED_URL_ERROR

    if any(target.host.startswith("__non_http_scheme__:") for target in target_urls):
        return NON_HTTP_SCHEME_URL_ERROR
    if any(target.host == "__malformed_url__" for target in target_urls):
        return MALFORMED_URL_ERROR

    if not user_task:
        return MISSING_USER_INTENT_CONTEXT_ERROR if require_user_task else None
    if not isinstance(user_task, str):
        return MISSING_USER_INTENT_CONTEXT_ERROR if require_user_task else None

    user_mentions = extract_url_intent_mentions(user_task)
    user_mentions.extend(_literal_mentions_for_tool_targets(user_task, target_urls))
    deduped_mentions: list[UrlIntentMention] = []
    for mention in sorted(user_mentions, key=lambda item: (-(item.end - item.start), item.start)):
        overlaps_existing = any(
            _hosts_overlap(mention.target.host, existing.target.host)
            and mention.start < existing.end
            and existing.start < mention.end
            for existing in deduped_mentions
        )
        if not overlaps_existing:
            deduped_mentions.append(mention)
    user_mentions = sorted(deduped_mentions, key=lambda mention: (mention.start, mention.end))
    user_urls = {mention.target for mention in user_mentions}
    if not user_urls or not _target_sets_overlap(target_urls, user_urls):
        return AMBIGUOUS_PASTED_URL_ERROR

    for target in target_urls:
        same_host_mentions = [
            (index, mention)
            for index, mention in enumerate(user_mentions)
            if _hosts_overlap(target.host, mention.target.host)
        ]
        target_contexts = [
            (
                index,
                _action_context_for_mention(user_task, user_mentions, index),
                _has_action_prefix_for_mention(user_task, user_mentions, index),
            )
            for index, mention in same_host_mentions
            if _targets_compatible(target, mention.target)
        ]
        # Every URL-action target must be explicitly authorized by the current
        # user turn. A generic follow-up like "확인해줘" or a different URL in
        # the same turn is not enough to browse/fetch this target.
        if not same_host_mentions:
            return AMBIGUOUS_PASTED_URL_ERROR
        # If the same host was mentioned but only at a different explicit path
        # or port, fail closed instead of sharing permission across targets.
        if not target_contexts:
            return AMBIGUOUS_PASTED_URL_ERROR

        # English imperatives normally precede their URL ("open example.com").
        # When an English action word appears between two URL mentions, do not
        # let it authorize the previous URL; fail closed instead of guessing.
        if any(
            _ENGLISH_EXPLICIT_URL_ACTION_RE.search(context)
            and (index + 1 < len(user_mentions) or not has_action_prefix)
            for index, context, has_action_prefix in target_contexts
        ):
            return AMBIGUOUS_PASTED_URL_ERROR

        contexts = [context for _, context, _ in target_contexts]
        if any(_NEGATED_URL_ACTION_RE.search(context) for context in contexts):
            return AMBIGUOUS_PASTED_URL_ERROR
        if any(
            _has_passive_marker_after_mention(user_task, user_mentions, index)
            or _has_passive_marker_before_mention(user_task, user_mentions, index)
            for index, _, _ in target_contexts
        ):
            return AMBIGUOUS_PASTED_URL_ERROR
        if not any(_EXPLICIT_URL_ACTION_RE.search(context) for context in contexts):
            return AMBIGUOUS_PASTED_URL_ERROR
    return None
