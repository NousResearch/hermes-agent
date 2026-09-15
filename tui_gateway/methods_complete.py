"""Completion / model-key / paste JSON-RPC handlers.

Reaches server.py state through ``srv`` (method_ctx.py).
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module
from .contracts.config_free_tier_control import ModelOptionsParams, ModelOptionsResult
from .contracts.profiles_vault_complete_foreign_subagents import (
    CompletePathParams,
    CompleteSlashParams,
    CompleteSlashResult,
    CompletionItemsResult,
    ModelDisconnectParams,
    ModelDisconnectResult,
    ModelSaveKeyParams,
    ModelSaveKeyResult,
    PasteCollapseParams,
    PasteCollapseResult,
)
import contextlib
import os

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_BUILTIN_AT_PREFIXES = frozenset({"file", "folder", "url", "git", "diff", "staged"})
_AT_DIRECTIVE_HINTS = [
    ("@diff", "git diff"), ("@staged", "staged diff"), ("@file:", "attach file"),
    ("@folder:", "attach folder"), ("@url:", "fetch url"), ("@git:", "git log")]
_SLASH_EXTRAS = [
    ("/density", "Toggle compact display mode"), ("/details", "Control agent detail visibility"),
    ("/logs", "Show recent gateway log lines"),
    ("/mouse", "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]")]


def _item(text: str, meta: str, display: str | None = None) -> dict:
    return {"text": text, "display": display if display is not None else text, "meta": meta}


def _catch(fail_code: int):
    """Handler body exceptions → ``_err(rid, fail_code, str(e))``."""

    def deco(body):
        def handler(rid, params):
            try:
                return body(rid, params)
            except Exception as e:
                return srv._err(rid, fail_code, str(e))
        handler.__doc__ = body.__doc__
        return handler
    return deco


@method("paste.collapse")
def _(rid, params: PasteCollapseParams) -> PasteCollapseResult | dict:
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import PasteCollapseResult
    text = params.text
    if not text:
        return srv._err(rid, 4004, "empty paste")
    srv._paste_counter += 1
    line_count = text.count("\n") + 1
    paste_dir = srv._hermes_home / "pastes"
    paste_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    paste_file = paste_dir / f"paste_{srv._paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
    paste_file.write_text(text, encoding="utf-8")
    placeholder = f"[Pasted text #{srv._paste_counter}: {line_count} lines \u2192 {paste_file}]"
    return PasteCollapseResult(placeholder=placeholder, path=str(paste_file), lines=line_count)


def _profile_mention_items(prefix: str) -> list[dict]:
    """`@<profile>` completions (multi-agent UIs route `@<profile>` text to another
    profile). Bare-word matches only, never `@kind:` directives; the primary profile
    is also offered as 'hermes' when no real profile claims that name."""
    out: list[dict] = []
    try:
        from hermes_cli.profiles import list_profiles
        seen: set[str] = set()
        for p in list_profiles():
            if not (name := (p.name or "").strip()):
                continue
            seen.add(name.lower())
            if name.lower().startswith(prefix.lower()):
                out.append(srv._item(f"@{name}", (getattr(p, "description", "") or "").strip() or "agent profile"))
        if "hermes".startswith(prefix.lower()) and "hermes" not in seen:
            out.append(srv._item("@hermes", "agent profile (primary)"))
    except Exception:
        return []
    return out


def _plugin_reference_items(pfx: str, qval: str) -> list[dict] | None:
    """`@<prefix>:<query>` autocomplete for a plugin ContextReferenceProvider; None when
    no provider owns ``pfx`` or it fails."""
    try:
        from agent.context_references import get_context_reference_providers
        import asyncio
        if (prov := get_context_reference_providers().get(pfx)) is None:
            return None
        coro = prov.autocomplete(qval, limit=20)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            ac = asyncio.run(coro)
        else:  # already inside a running loop: run the coroutine on a side thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                ac = pool.submit(asyncio.run, coro).result()
        return [{"text": f"@{pfx}:{it.text}", "display": it.display, "meta": it.meta} for it in ac]
    except Exception:
        return None


def _fuzzy_basename_items(root: str, path_part: str, prefix_tag: str) -> list[dict]:
    """Cmd-P style fuzzy basename search for a bare `@name`; path-ish queries take the listing path."""
    ranked: list[tuple[tuple[int, int], str, str, bool]] = []
    walked_dirs: set[str] = set()
    seen: set[str] = set()
    want_hidden = path_part.startswith(".")

    def _consider(rel: str, name: str, is_dir: bool) -> None:
        if rel in seen or (name.startswith(".") and not want_hidden):
            return
        if (rank := srv._fuzzy_basename_rank(name, path_part)) is not None:
            seen.add(rel)
            ranked.append((rank, rel, name, is_dir))

    # Seed with root's immediate children: `_list_repo_files` is capped at _FUZZY_CACHE_MAX_FILES
    # and the non-git fallback walk can burn the whole budget on one deep subtree.
    with contextlib.suppress(OSError):
        for entry in os.listdir(root):
            if entry not in srv._FUZZY_FALLBACK_EXCLUDES:
                _consider(entry, entry, os.path.isdir(os.path.join(root, entry)))
    for rel in srv._list_repo_files(root):
        _consider(rel, os.path.basename(rel), False)
        # Rank each ancestor dir too — a folder with no name-matching file inside is otherwise invisible.
        parent = os.path.dirname(rel)
        while parent and parent not in walked_dirs:
            walked_dirs.add(parent)
            _consider(parent, os.path.basename(parent), True)
            parent = os.path.dirname(parent)

    # Same rank tier: folders first, so `@Desktop` leads with the folder.
    ranked.sort(key=lambda r: (r[0], not r[3], len(r[1]), r[1]))
    tag = prefix_tag or "file"
    return [
        srv._item(
            f"@{'folder' if is_dir else tag}:{rel}{'/' if is_dir else ''}",
            "dir" if is_dir else os.path.dirname(rel), basename + ("/" if is_dir else ""))
        for _, rel, basename, is_dir in ranked[:30]]


def _at_root_items() -> list[dict]:
    """Completions for a bare ``@``: directive hints, agent profiles, plugin ``@<prefix>:`` providers."""
    items = [srv._item(t, m) for t, m in srv._AT_DIRECTIVE_HINTS] + srv._profile_mention_items("")
    with contextlib.suppress(Exception):
        from agent.context_references import get_context_reference_providers
        for pfx, prov in sorted(get_context_reference_providers().items()):
            items.append(srv._item(f"@{pfx}:", prov.description or f"plugin: {pfx}"))
    return items


def _dir_listing_items(root: str, word: str, path_part: str, prefix_tag: str, is_context: bool) -> list[dict]:
    """Prefix-match entries of the directory ``path_part`` points at (max 30)."""
    expanded = srv._normalize_completion_path(path_part) if path_part else "."
    if expanded == "." or not expanded or expanded.endswith("/"):
        search_dir, match = (expanded or "."), ""
    else:
        search_dir, match = os.path.dirname(expanded) or ".", os.path.basename(expanded)
    search_dir = search_dir if os.path.isabs(search_dir) else os.path.join(root, search_dir)
    items: list[dict] = []
    if not os.path.isdir(search_dir):
        return items
    for entry in sorted(os.listdir(search_dir)):
        if match and not entry.lower().startswith(match.lower()):
            continue
        if is_context and (entry in srv._FUZZY_FALLBACK_EXCLUDES or (not prefix_tag and entry.startswith("."))):
            continue
        full = os.path.join(search_dir, entry)
        is_dir = os.path.isdir(full)
        if prefix_tag and (prefix_tag == "folder") != is_dir:  # explicit `@folder:`/`@file:` skip the other kind
            continue
        rel = os.path.relpath(full, root).replace(os.sep, "/")
        suffix = "/" if is_dir else ""
        if is_context:
            text = f"@{prefix_tag or ('folder' if is_dir else 'file')}:{rel}{suffix}"
        elif word.startswith("~"):
            text = "~/" + os.path.relpath(full, os.path.expanduser("~")) + suffix
        else:
            text = ("./" if word.startswith("./") else "") + rel + suffix
        items.append(srv._item(text, "dir" if is_dir else "", entry + suffix))
        if len(items) >= 30:
            break
    return items


@method("complete.path")
@_catch(5021)
def _(rid, params: CompletePathParams) -> CompletionItemsResult | dict:
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import CompletionItemsResult
    word = params.word
    if not word:
        return CompletionItemsResult(items=[])
    root = srv._completion_cwd({"cwd": params.cwd, "session_id": params.session_id, "profile": params.profile})
    is_context = word.startswith("@")
    query = word[1:] if is_context else word
    if is_context and not query:
        return CompletionItemsResult(items=srv._at_root_items())
    # Plugin `@<prefix>:<query>` runs before the built-in file/folder branching.
    if is_context and ":" in query:
        pfx, _, qval = query.partition(":")
        if pfx not in srv._BUILTIN_AT_PREFIXES and (plugin_items := srv._plugin_reference_items(pfx, qval)) is not None:
            return CompletionItemsResult(items=plugin_items)
    # Bare `@folder` lists as soon as the keyword is typed (the static `@folder:` hint is not accepted).
    if is_context and (query in {"file", "folder"} or query.startswith(("file:", "folder:"))):
        prefix_tag, _, path_part = query.partition(":")
    else:
        prefix_tag, path_part = "", query
    # `@/foo` usually means "foo, from here": absolute only when that prefix exists,
    # else resolve relative to cwd (`@/Desktop` must not dead-end; `@/usr/local` still resolves).
    if (
        is_context and path_part.startswith("/") and not path_part.startswith("//")
        and not srv._abs_completion_prefix_exists(path_part)):
        path_part = path_part.lstrip("/")
    bare_word = is_context and path_part and "/" not in path_part
    if bare_word and len(path_part.strip()) >= 2 and prefix_tag != "folder":
        items = srv._fuzzy_basename_items(root, path_part, prefix_tag)
    else:
        items = srv._dir_listing_items(root, word, path_part, prefix_tag, is_context)
    # Bare-word `@name` may be an agent mention: profiles rank ABOVE file hits.
    if bare_word and not prefix_tag:
        with contextlib.suppress(Exception):
            items = srv._profile_mention_items(path_part) + items
    return CompletionItemsResult(items=items)


@method("complete.slash")
@_catch(5020)
def _(rid, params: CompleteSlashParams) -> CompleteSlashResult | dict:
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import CompleteSlashResult
    text = params.text
    if not text.startswith("/"):
        return CompleteSlashResult(items=[])
    from hermes_cli.commands_completion import SlashCommandCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import to_plain_text
    from agent.skill_commands import get_skill_commands
    from agent.skill_bundles import get_skill_bundles
    completer = SlashCommandCompleter(
        skill_commands_provider=lambda: get_skill_commands(), skill_bundles_provider=lambda: get_skill_bundles())
    # `kind` reaches the TUI as data (from the providers, not sniffed from ⚡/▣ glyphs):
    # skills/bundles are the only completions for an inline `/skill` typed mid-message.
    skill_names = {key.lstrip("/").lower() for key in (*get_skill_commands(), *get_skill_bundles())}

    def to_items(doc: Document) -> list[dict]:
        # display/display_meta are FormattedText; the TUI contract is a plain string
        # (the raw list trips Ink's row layout into 1-char truncation).
        return [
            {
                "text": c.text, "display": to_plain_text(c.display) if c.display else c.text,
                "meta": to_plain_text(c.display_meta) if c.display_meta else "",
                "kind": "skill" if c.text.strip().lstrip("/").lower() in skill_names else "command"}
            for c in completer.get_completions(doc, None)]
    items = to_items(Document(text, len(text)))
    # Rank + bound while a `/token` is under the cursor (the one stage skills are
    # offered at); an argument stage (`/personality `) keeps its command's order.
    if text.rsplit(" ", 1)[-1].startswith("/"):
        score_of = None
        # Command-token stage: the completer only emits name-prefix matches, so merge in
        # catalog entries whose name SUBSTRING or DESCRIPTION words match (name outranks description).
        if " " not in text and len(text) > 1:
            from tui_gateway.slash_fuzzy import fuzzy_rank_slash_items, normalize_slash_search_query
            items, score_of = fuzzy_rank_slash_items(
                items, to_items(Document("/", 1)), normalize_slash_search_query(text))
        usage, origin_of = srv._skill_usage_lookup()
        items = srv._rank_slash_completions(items, usage, origin_of, browsing=text == "/", score_of=score_of)
    else:
        items = items[:srv._SLASH_COMPLETION_LIMIT]
    text_lower = text.lower()
    for extra_text, extra_meta in srv._SLASH_EXTRAS:
        if extra_text.startswith(text_lower) and not any(item["text"] == extra_text for item in items):
            items.append({**srv._item(extra_text, extra_meta), "kind": "command"})
    if (details_items := srv._details_completions(text)) is not None:
        return CompleteSlashResult(items=details_items, replace_from=text.rfind(" ") + 1 if " " in text else len(text))
    return CompleteSlashResult(items=items, replace_from=text.rfind(" ") + 1 if " " in text else 1)


def _session_agent(params):
    session = srv._sessions.get(params.session_id or "")
    return session.get("agent") if session else None


@method("model.options")
@_profile_scoped
@_catch(5033)
def _(rid, params: ModelOptionsParams) -> ModelOptionsResult | dict:
    from tui_gateway.contracts.config_free_tier_control import ModelOptionsResult
    from hermes_cli.inventory import build_model_options_payload
    # A spawned agent owns the live provider/model/base_url; empty attributes must
    # NOT clobber disk config (with_overrides is truthy-only).
    return ModelOptionsResult.model_validate(build_model_options_payload(
        srv._model_picker_context(srv._session_agent(params)), explicit_only=params.explicit_only,
        include_unconfigured=params.include_unconfigured, refresh=params.refresh))


@method("model.save_key")
@_catch(5034)
def _(rid, params: ModelSaveKeyParams) -> ModelSaveKeyResult | dict:
    """Save an API key for ``slug``; return its refreshed provider row (model.options shape + ``authenticated``)."""
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import ModelSaveKeyResult
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import is_managed
    slug, api_key = params.slug.strip(), params.api_key.strip()
    if not slug or not api_key:
        return srv._err(rid, 4001, "slug and api_key are required")
    if is_managed():
        return srv._err(rid, 4006, "managed install — credentials are read-only")
    if not (pconfig := PROVIDER_REGISTRY.get(slug)):
        return srv._err(rid, 4002, f"unknown provider: {slug}")
    if pconfig.auth_type != "api_key":
        return srv._err(rid, 4003, f"{pconfig.name} uses {pconfig.auth_type} auth — run `hermes model` to configure")
    if not pconfig.api_key_env_vars:
        return srv._err(rid, 4004, f"no env var defined for {pconfig.name}")
    # Save the key to ~/.hermes/.env via the unified credential lifecycle so any stale config.yaml mirror of
    # the previous key (model.api_key, custom_providers[*].api_key) is rotated in the same action (#62269).
    env_var = pconfig.api_key_env_vars[0]
    from hermes_cli.credential_lifecycle import save_provider_env_credential  # also rotates stale config.yaml mirrors
    save_provider_env_credential(env_var, api_key)
    os.environ[env_var] = api_key  # so the refreshed inventory sees it
    # Shared inventory builder (lock-step with model.options / dashboard); picker_hints carries `authenticated`.
    from hermes_cli.inventory import build_models_payload
    payload = build_models_payload(srv._model_picker_context(srv._session_agent(params)), picker_hints=True, max_models=50)
    provider_data = next((p for p in payload["providers"] if p["slug"] == slug), None)
    if provider_data is None:  # key saved but provider didn't appear — still success
        provider_data = {"slug": slug, "name": pconfig.name, "is_current": False, "models": [], "total_models": 0}
    provider_data["authenticated"] = True  # synthetic fallback bypasses picker_hints
    return ModelSaveKeyResult(provider=provider_data)


@method("model.disconnect")
@_catch(5035)
def _(rid, params: ModelDisconnectParams) -> ModelDisconnectResult | dict:
    """Remove all credentials (env keys AND OAuth/pool state) for provider ``slug``."""
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import ModelDisconnectResult
    from hermes_cli.auth import PROVIDER_REGISTRY, clear_provider_auth
    from hermes_cli.credential_lifecycle import remove_provider_env_credential
    if not (slug := params.slug.strip()):
        return srv._err(rid, 4001, "slug is required")
    pconfig = PROVIDER_REGISTRY.get(slug)
    # Remove EVERY env var plus its mirrors or the provider resurrects in the picker after restart.
    env_vars = (pconfig.api_key_env_vars if pconfig else None) or ()
    cleared_env = any([remove_provider_env_credential(ev).get("found") for ev in env_vars])
    cleared_auth = clear_provider_auth(slug)  # full disconnect: OAuth grants go too
    if not cleared_env and not cleared_auth:
        return srv._err(rid, 4005, f"no credentials found for {slug}")
    return ModelDisconnectResult(slug=slug, name=pconfig.name if pconfig else slug, disconnected=True)


def register(server) -> None:
    """Rebind this module's helpers + handlers onto ``server`` and register the handlers."""
    bind_module(globals(), server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
