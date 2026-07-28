"""Discord View and Modal classes for the clarify tool's rich options.

Provides ``InteractivePromptView`` (button grid) and ``InteractivePromptModal``
(form popup) used by the Discord adapter to render rich clarify prompts and
collect structured user responses.

These classes are intentionally kept in a standalone module so the adapter
(~6000 LOC) doesn't grow further and the contribution stays modular.

Resolution goes through ``tools.clarify_gateway.resolve_gateway_clarify`` —
the same primitive used by the simple-choices clarify path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set

# Conditional discord import — gracefully degrades when discord.py is absent
# or stubbed (e.g. another test replaces sys.modules["discord"] with a mock).
try:
    import discord
    from discord import ui as _ui
    # Verify the real discord.py API surface exists — test mocks replace
    # sys.modules["discord"] with a SimpleNamespace that lacks these.
    _ui.View  # noqa: B018
    _ui.Modal  # noqa: B018
except (ImportError, AttributeError):  # pragma: no cover
    discord = None  # type: ignore[assignment]
    _ui = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discord API limits
# ---------------------------------------------------------------------------
_DISCORD_MODAL_TITLE_MAX = 45
_DISCORD_LABEL_MAX = 45
_DISCORD_LABEL_DESCRIPTION_MAX = 100
_DISCORD_MODAL_CHILD_MAX = 5

# ---------------------------------------------------------------------------
# Modal upload resource limits
# ---------------------------------------------------------------------------
# Modal file-upload submissions are written to disk under the active Hermes
# home.  Without explicit caps a single oversized attachment (Discord Nitro
# permits 500 MB) or a large aggregate submission can spike RAM/disk and OOM
# the gateway.  Both bounds are checked against each attachment's reported
# ``size`` *before* any bytes are read or written, so a rejected upload never
# produces a partial cache file and never resolves the prompt as successful.
#
# The defaults live in config.yaml under ``gateway`` so operators can tune
# them for constrained deployments; per-field overrides live in the field's
# ``file_policy`` (part of the prompt definition the agent controls):
# ``max_bytes`` (per-file) and ``max_total_bytes`` (aggregate).  Unspecified /
# invalid values fall back to the configured default — uploads are always
# bounded by at least the floor.
_MODAL_UPLOAD_MAX_PER_FILE_DEFAULT = 10 * 1024 * 1024   # 10 MiB (Discord free-tier cap)
_MODAL_UPLOAD_MAX_AGGREGATE_DEFAULT = 25 * 1024 * 1024  # 25 MiB per submission

# Reusable, safe rejection copy for unreadable attachments (no paths/exceptions).
_UPLOAD_READ_FAILURE_MESSAGE = (
    "Upload rejected: could not read one of your files. "
    "Please re-attach it and try again."
)


def _get_modal_upload_limits() -> tuple:
    """Resolve (per_file_bytes, aggregate_bytes) from config.yaml ``gateway``.

    Falls back to the module defaults when config is unreadable or a value is
    missing/non-positive.  Mirrors the defensive ``load_config()`` pattern used
    by ``clarify_gateway.get_clarify_timeout``.
    """
    per_file = _MODAL_UPLOAD_MAX_PER_FILE_DEFAULT
    aggregate = _MODAL_UPLOAD_MAX_AGGREGATE_DEFAULT
    try:
        from hermes_cli.config import load_config
        gw = (load_config() or {}).get("gateway", {}) or {}
    except Exception:
        return per_file, aggregate
    raw_pf = gw.get("modal_upload_max_per_file_bytes")
    if isinstance(raw_pf, int) and raw_pf >= 0:
        per_file = raw_pf
    raw_agg = gw.get("modal_upload_max_aggregate_bytes")
    if isinstance(raw_agg, int) and raw_agg >= 0:
        aggregate = raw_agg
    return per_file, aggregate


def _humanize_bytes(n: int) -> str:
    """Render a byte count as a short human-readable string (e.g. ``"10.0 MB"``)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _coerce_positive_int(value: Any, default: int) -> int:
    """Return ``int(value)`` when it is a positive int, else ``default``."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------


def unwrap_modal_children(children):
    """Unwrap discord.ui.Label wrappers to get inner components.

    After the Sep 2025 modal API change, all interactive components inside
    modals are wrapped in ``discord.ui.Label`` (type 18).  This helper
    returns the inner ``TextInput`` / ``Select`` / ``RadioGroup`` /
    ``CheckboxGroup`` components, falling back to the raw child when no
    Label wrapper is present (e.g. legacy contexts or tests without Labels).
    """
    result = []
    for child in children:
        if discord is not None and isinstance(child, discord.ui.Label):
            result.append(child.component)
        else:
            result.append(child)
    return result


# ---------------------------------------------------------------------------
# Style mapping: human-readable names → discord.ButtonStyle
# ---------------------------------------------------------------------------

STYLE_MAP: Dict[str, Any] = {}
if discord is not None:
    STYLE_MAP = {
        "primary": discord.ButtonStyle.primary,
        "secondary": discord.ButtonStyle.secondary,
        "success": discord.ButtonStyle.green,
        "green": discord.ButtonStyle.green,
        "danger": discord.ButtonStyle.red,
        "red": discord.ButtonStyle.red,
    }


# ---------------------------------------------------------------------------
# Helper: embed builder
# ---------------------------------------------------------------------------

def build_prompt_embed(
    question: str,
    status: str = "pending",
) -> Any:
    """Build a Discord embed for an interactive prompt.

    Parameters
    ----------
    question:
        The question text shown as the embed description (max 4088 chars).
    status:
        ``"pending"`` → orange, ``"resolved"`` → green, ``"awaiting"`` → blue.

    Returns
    -------
    discord.Embed
    """
    if discord is None:  # pragma: no cover
        return None

    colour_map = {
        "pending": discord.Color.orange(),
        "resolved": discord.Color.green(),
        "awaiting": discord.Color.blue(),
    }
    colour = colour_map.get(status, discord.Color.orange())

    description = question if len(question) <= 4088 else question[:4085] + "..."
    embed = discord.Embed(
        title="🔘 Hermes has a question",
        description=description,
        color=colour,
    )
    return embed


# ---------------------------------------------------------------------------
# Auth helpers — delegates to shared utilities to avoid duplication
# ---------------------------------------------------------------------------

from tools.discord_auth_helpers import (
    component_check_auth as _component_check_auth,
    session_owner_check_auth as _session_owner_check_auth,
)

from dataclasses import dataclass as _dataclass


@_dataclass
class FileResult:
    """Metadata for an uploaded file from a modal submission."""
    field_key: str
    attachment_id: str
    filename: str
    content_type: str
    size: int
    cached_path: str = ""


class _FileSpec(NamedTuple):
    """A collected file-upload field awaiting validation / caching."""

    field_key: str
    attachments: List[Any]
    file_policy: Dict[str, Any]


# =========================================================================
# InteractivePromptView
# =========================================================================

if discord is not None:

    class InteractivePromptView(discord.ui.View):
        """Button-grid view for a rich-options ``clarify`` question.

        Renders one button per option (max 25, respecting Discord's 5-per-row
        ActionRow limit).  Supports:

        * ``"return"`` options — immediately resolve via
          ``clarify_gateway.resolve_gateway_clarify``
        * ``"modal"`` options — open an ``InteractivePromptModal`` whose
          ``on_submit`` handler resolves via ``resolve_modal``

        Auth gating mirrors the adapter's ``_component_check_auth`` with an
        additional ``session_owner_only`` fast-path.
        """

        def __init__(
            self,
            prompt_id: str,
            question: str,
            options: List[Dict[str, Any]],
            allowed_user_ids: Set[str],
            allowed_role_ids: Optional[Set[str]] = None,
            auth_policy: str = "session_owner_only",
            origin_user_id: Optional[str] = None,
            timeout_seconds: float = 900,
        ) -> None:
            # Discord.ui.View timeout must match the agent's wait_for_response
            # deadline.  Cap at 3600s (Discord's effective max for views).
            super().__init__(timeout=min(timeout_seconds, 3600))
            self.prompt_id = prompt_id
            self.question = question
            self.options = list(options)[:25]  # Discord max 25 buttons
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.auth_policy = auth_policy
            self.origin_user_id = origin_user_id
            self.resolved = False
            # Store the original message so the modal can update it later.
            self._message = None  # type: ignore[assignment]

            for index, option in enumerate(self.options):
                label = option.get("label", f"Option {index + 1}")
                if len(label) > 80:
                    label = label[:77] + "..."

                style_name = option.get("style", "secondary")
                style = STYLE_MAP.get(style_name, discord.ButtonStyle.secondary)

                button = discord.ui.Button(
                    label=label,
                    style=style,
                    custom_id=f"hermes:ip:{prompt_id}:{index}",
                )
                button.callback = self._make_callback(index, option)
                self.add_item(button)

        # ---- Auth --------------------------------------------------------

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Check whether *interaction.user* may respond to this prompt.

            Each policy is enforced independently:

              * ``session_owner_only`` — admits the session initiator
                (``origin_user_id``); an existing user allowlist gates
                independently (union).  With no owner and no allowlist the
                policy fails closed rather than admit everyone.
              * ``any_allowed_user`` — only the user allowlist is consulted;
                roles are ignored even if present.
              * ``any_allowed_role`` — only the role allowlist is consulted;
                user ID matches are ignored.
              * ``any_allowed_user_or_role`` — user OR role allowlist (union).
            """
            if self.auth_policy == "session_owner_only":
                return _session_owner_check_auth(
                    interaction, self.origin_user_id, self.allowed_user_ids,
                )

            # Policy-specific checks for the remaining three policies.
            if self.auth_policy == "any_allowed_user":
                # Only user allowlist — ignore roles entirely.
                user_set = self.allowed_user_ids or set()
                if not user_set:
                    return True  # no-allowlist deployment
                try:
                    return str(interaction.user.id) in user_set
                except AttributeError:
                    return False

            if self.auth_policy == "any_allowed_role":
                # Only role allowlist — ignore user IDs entirely.
                role_set = self.allowed_role_ids or set()
                if not role_set:
                    return True  # no-allowlist deployment
                roles_attr = getattr(interaction.user, "roles", None)
                if roles_attr is None:
                    return False  # fail closed (DM context)
                try:
                    user_role_ids = {getattr(r, "id", None) for r in roles_attr}
                except TypeError:
                    return False
                return bool(user_role_ids & role_set)

            # any_allowed_user_or_role — union of user and role allowlists.
            return _component_check_auth(
                interaction,
                self.allowed_user_ids,
                self.allowed_role_ids,
            )

        # ---- Button factory ----------------------------------------------

        def _make_callback(self, index: int, option: Dict[str, Any]) -> Callable:
            """Return an async callback wired to a specific option."""

            async def _callback(interaction: discord.Interaction) -> None:
                await self._resolve_choice(interaction, index, option)

            return _callback

        # ---- Choice resolution -------------------------------------------

        async def _resolve_choice(
            self,
            interaction: discord.Interaction,
            index: int,
            option: Dict[str, Any],
        ) -> None:
            """Handle a button click — resolve or open modal."""
            if self.resolved:
                await interaction.response.send_message(
                    "This prompt has already been answered.",
                    ephemeral=True,
                )
                return

            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized to answer this prompt.",
                    ephemeral=True,
                )
                return

            action = option.get("action", "return")

            # ── Modal path ──────────────────────────────────────────────
            # Send the modal FIRST — this consumes the interaction response.
            # Buttons stay enabled so the user can retry if they dismiss
            # the modal. Disabling + resolving happens in on_submit().
            if action == "modal":
                modal_spec = option.get("modal", {})
                modal = InteractivePromptModal(
                    prompt_id=self.prompt_id,
                    option_index=index,
                    modal_spec=modal_spec,
                    original_view=self,
                )
                await interaction.response.send_modal(modal)
                return

            self.resolved = True
            self._disable_all()

            # ── Return (default) path ──────────────────────────────────
            embed = None
            if interaction.message and interaction.message.embeds:
                embed = interaction.message.embeds[0]
                if embed:
                    user = getattr(interaction, "user", None)
                    display_name = (getattr(user, "display_name", "user") or "user")[:32]
                    embed.color = discord.Color.green()
                    embed.set_footer(
                        text=f"Answered by {display_name}: {option.get('label', '')}"
                    )

            try:
                await interaction.response.edit_message(embed=embed, view=self)
            except Exception:
                logger.debug(
                    "InteractivePrompt edit_message failed for %s",
                    self.prompt_id,
                    exc_info=True,
                )
                try:
                    await interaction.response.defer()
                except Exception:
                    pass

            # Resolve via the clarify_gateway.
            try:
                from tools.clarify_gateway import resolve_gateway_clarify
                import json as _json

                user = getattr(interaction, "user", None)
                result_json = _json.dumps({
                    "status": "answered",
                    "value": option.get("value", ""),
                    "label": option.get("label", ""),
                    "user_id": str(getattr(user, "id", "")),
                    "user_name": getattr(user, "display_name", ""),
                }, ensure_ascii=False)
                resolved = resolve_gateway_clarify(self.prompt_id, result_json)
                logger.info(
                    "InteractivePrompt button resolved (id=%s, value=%r, ok=%s)",
                    self.prompt_id,
                    option.get("value", ""),
                    resolved,
                )
            except Exception as exc:
                logger.error(
                    "InteractivePrompt resolve failed (id=%s): %s",
                    self.prompt_id,
                    exc,
                )

        # ---- Timeout / disable -------------------------------------------

        async def on_timeout(self) -> None:
            """Disable all buttons when the view times out."""
            self.resolved = True
            self._disable_all()

        def _disable_all(self) -> None:
            """Set ``disabled=True`` on every child component."""
            for child in self.children:
                child.disabled = True


# =========================================================================
# InteractivePromptModal
# =========================================================================

if discord is not None:

    class InteractivePromptModal(discord.ui.Modal):
        """Modal (form popup) for interactive-prompt options with ``action: "modal"``.

        Supports ``"text"``, ``"select"``, ``"radio"``, ``"checkbox"``,
        and ``"file_upload"`` field types via ``discord.ui`` components.
        """

        def __init__(
            self,
            prompt_id: str,
            option_index: int,
            modal_spec: Dict[str, Any],
            original_view: Optional[InteractivePromptView] = None,
        ) -> None:
            title = modal_spec.get("title", "Respond")
            if len(title) > _DISCORD_MODAL_TITLE_MAX:
                title = title[:42] + "..."

            super().__init__(
                title=title,
                custom_id=f"hermes:ip-modal:{prompt_id}:{option_index}",
            )
            self.prompt_id = prompt_id
            self.option_index = option_index
            self.modal_spec = modal_spec
            self.original_view = original_view

            # Store the field keys in order so we can map submitted values.
            self._field_keys: List[str] = []
            # field_key → file_policy dict, so ``on_submit`` can enforce the
            # file-count and byte limits declared in the prompt definition
            # *before* it reads or caches any attachment bytes.
            self._field_policies: Dict[str, Dict[str, Any]] = {}

            for field_spec in modal_spec.get("fields", []):
                field_type = field_spec.get("type", "text")
                key = field_spec.get("key", "")
                field_label = field_spec.get("label", key)[:_DISCORD_LABEL_MAX]
                field_description = field_spec.get("description", "")

                # Discord modals support a maximum of 5 children.
                if len(self._field_keys) >= _DISCORD_MODAL_CHILD_MAX:
                    logger.warning(
                        "Discord modal max 5 fields reached; skipping "
                        "field %r (prompt_id=%s)",
                        key,
                        self.prompt_id,
                    )
                    continue

                if field_type == "text":
                    multiline = field_spec.get("multiline", False)
                    text_input = discord.ui.TextInput(
                        label=None,
                        placeholder=field_spec.get("placeholder", "")[:100],
                        required=field_spec.get("required", False),
                        max_length=field_spec.get("max_length", 1000),
                        min_length=field_spec.get("min_length", 0),
                        style=(
                            discord.TextStyle.paragraph
                            if multiline
                            else discord.TextStyle.short
                        ),
                        default=field_spec.get("default", None),
                    )
                    label = _ui.Label(
                        text=field_label,
                        description=field_description[:_DISCORD_LABEL_DESCRIPTION_MAX] if field_description else None,
                        component=text_input,
                    )
                    self._field_keys.append(key)
                    self.add_item(label)

                elif field_type == "select":
                    field_options = field_spec.get("options", [])
                    field_required = field_spec.get("required", False)
                    select = discord.ui.Select(
                        custom_id=key[:100],
                        placeholder=field_spec.get("placeholder", "")[:100],
                        required=field_required,
                        min_values=0 if not field_required else 1,
                        max_values=1,
                    )
                    for opt_val in field_options[:25]:
                        label = str(opt_val)
                        select.add_option(
                            label=label[:100],
                            value=label[:100],
                        )
                    label = _ui.Label(
                        text=field_label,
                        description=field_description[:_DISCORD_LABEL_DESCRIPTION_MAX] if field_description else None,
                        component=select,
                    )
                    self._field_keys.append(key)
                    self.add_item(label)

                elif field_type == "radio":
                    field_options = field_spec.get("options", [])
                    radio = discord.ui.RadioGroup(
                        custom_id=key[:100],
                        required=field_spec.get("required", False),
                    )
                    for opt_val in field_options[:10]:
                        label = str(opt_val)
                        radio.add_option(
                            label=label[:100],
                            value=label[:100],
                        )
                    label = _ui.Label(
                        text=field_label,
                        description=field_description[:_DISCORD_LABEL_DESCRIPTION_MAX] if field_description else None,
                        component=radio,
                    )
                    self._field_keys.append(key)
                    self.add_item(label)

                elif field_type == "checkbox":
                    field_options = field_spec.get("options", [])
                    checkbox = discord.ui.CheckboxGroup(
                        custom_id=key[:100],
                        required=field_spec.get("required", False),
                    )
                    for opt_val in field_options[:10]:
                        label = str(opt_val)
                        checkbox.add_option(
                            label=label[:100],
                            value=label[:100],
                        )
                    label = _ui.Label(
                        text=field_label,
                        description=field_description[:_DISCORD_LABEL_DESCRIPTION_MAX] if field_description else None,
                        component=checkbox,
                    )
                    self._field_keys.append(key)
                    self.add_item(label)

                elif field_type == "file_upload":
                    file_policy = field_spec.get("file_policy", {})
                    file_upload = _ui.FileUpload(
                        custom_id=key[:100],
                        required=field_spec.get("required", False),
                        max_values=file_policy.get("max_files", 1),
                        min_values=file_policy.get("min_files", 0),
                    )
                    label = _ui.Label(
                        text=field_label,
                        description=field_description[:_DISCORD_LABEL_DESCRIPTION_MAX] if field_description else None,
                        component=file_upload,
                    )
                    self._field_keys.append(key)
                    self._field_policies[key] = dict(file_policy or {})
                    self.add_item(label)

                else:
                    logger.warning(
                        "Unknown modal field type %r (prompt_id=%s, field=%s)",
                        field_type,
                        self.prompt_id,
                        key,
                    )

        def to_dict(self) -> Dict[str, Any]:
            """Override to strip 'disabled' from Select components in modal payload.

            Discord rejects 'disabled' on Select inside modals (50035).
            The base Modal.to_dict() includes it because discord.ui.Select
            has a default .disabled=False attribute.

            Only Select (type 3) needs stripping — TextInput, RadioGroup,
            and CheckboxGroup do not serialise a ``disabled`` field.
            """
            base = super().to_dict()
            for comp in base.get("components", []):
                inner = comp.get("component")
                if inner and inner.get("type") == 3:  # Select
                    inner.pop("disabled", None)
            return base

        async def on_submit(self, interaction: discord.Interaction) -> None:
            """Collect field values and resolve the prompt via the gateway."""
            # Gather values from children in order.
            fields: Dict[str, Any] = {}
            # One entry per file-upload field, collected without reading any
            # bytes so Phase 2 can enforce count/size limits first.
            file_specs: List[_FileSpec] = []
            children = getattr(self, "children", [])
            unwrapped = unwrap_modal_children(children)
            for idx, inner in enumerate(unwrapped):
                if idx >= len(self._field_keys):
                    break
                field_key = self._field_keys[idx]
                # text → .value (str|None), radio → .value (str|None)
                if isinstance(inner, (discord.ui.TextInput, discord.ui.RadioGroup)):
                    fields[field_key] = getattr(inner, "value", None)
                # select → .values (list[str]), checkbox → .values (list[str])
                elif isinstance(inner, (discord.ui.Select, discord.ui.CheckboxGroup)):
                    fields[field_key] = getattr(inner, "values", [])
                # file_upload → defer reading to Phase 2/3 so count/size
                # limits are enforced before any bytes are pulled into
                # memory or written to disk.
                elif isinstance(inner, _ui.FileUpload):
                    attachments = list(getattr(inner, "values", []) or [])
                    file_specs.append(
                        _FileSpec(field_key, attachments, self._field_policies.get(field_key, {}))
                    )
                else:
                    fields[field_key] = getattr(inner, "value", None)

            # ── Phase 2: validate counts + sizes BEFORE any read/write ──
            # A rejected upload must not resolve the prompt and must not leave
            # partial cache files behind, so every bound is enforced against
            # each attachment's reported ``size`` here, before Phase 3 reads
            # or writes anything.
            rejection = self._validate_file_uploads(file_specs)
            if rejection is not None:
                await self._send_rejection(interaction, rejection)
                return

            # ── Phase 3: read + cache files ──
            # Validation already proved every file is within bounds; the only
            # remaining failure is a read/write error.  On any such error we
            # purge every cache file written so far and reject the whole
            # submission so no partial upload survives and the prompt stays
            # pending for the user to retry.
            files_collected, read_rejection = await self._collect_files(file_specs)
            if read_rejection is not None:
                self._purge_cached_files(files_collected)
                await self._send_rejection(interaction, read_rejection)
                return

            # Build actor info.
            user = getattr(interaction, "user", None)

            # Resolve the choice value from the option index via the view.
            if self.original_view is not None:
                opt = self.original_view.options[self.option_index]
            else:
                opt = None
            choice_value = opt.get("value", "") if opt else ""
            if opt is None:
                logger.warning(
                    "InteractivePromptModal could not resolve option "
                    "(id=%s, index=%d); using empty choice_value",
                    self.prompt_id,
                    self.option_index,
                )

            try:
                from tools.clarify_gateway import resolve_gateway_clarify
                import json as _json2

                result_json = _json2.dumps({
                    "status": "answered",
                    "value": choice_value,
                    "label": opt.get("label", "") if opt else "",
                    "fields": fields,
                    "files": [vars(f) for f in files_collected] if files_collected else [],
                    "user_id": str(getattr(user, "id", "")),
                    "user_name": getattr(user, "display_name", ""),
                }, ensure_ascii=False)
                resolved = resolve_gateway_clarify(self.prompt_id, result_json)
                logger.info(
                    "InteractivePrompt modal resolved (id=%s, value=%r, ok=%s)",
                    self.prompt_id,
                    choice_value,
                    resolved,
                )
            except Exception as exc:
                logger.error(
                    "InteractivePrompt resolve_modal failed (id=%s): %s",
                    self.prompt_id,
                    exc,
                )

            # Mark the original prompt as resolved and disable buttons.
            if self.original_view is not None:
                self.original_view.resolved = True
                self.original_view._disable_all()

            # Acknowledge the modal submission.
            try:
                await interaction.response.send_message(
                    "✅ Response submitted", ephemeral=True,
                )
            except Exception:
                logger.debug(
                    "InteractivePromptModal send_message ack failed",
                    exc_info=True,
                )

            # Update the original prompt message embed to green.
            if self.original_view is not None:
                try:
                    msg = getattr(self.original_view, "_message", None)
                    if msg is None:
                        # The view's message may have been set externally
                        # after the view was sent.  Try to grab it from
                        # the view's internal state or skip silently.
                        pass
                    if msg is not None and msg.embeds:
                        embed = msg.embeds[0]
                        embed.color = discord.Color.green()
                        display_name = (getattr(user, "display_name", "user") or "user")[:32]
                        embed.set_footer(
                            text=f"Answered by {display_name} (modal)"
                        )
                        await msg.edit(embed=embed, view=self.original_view)
                except Exception:
                    logger.debug(
                        "InteractivePromptModal original message update failed",
                        exc_info=True,
                    )

        # ---- File-upload bounding helpers (issue #10) -------------------

        def _validate_file_uploads(
            self,
            file_specs: List[_FileSpec],
        ) -> Optional[str]:
            """Enforce file-count and byte limits before any read or write.

            ``file_specs`` is a list of :class:`_FileSpec` collected in Phase 1.
            Returns a safe, actionable rejection message string when the
            submission violates a bound, or ``None`` when it is acceptable.

            Every check runs against each attachment's reported ``size`` —
            no bytes are read here — so an oversized upload never reaches the
            cache and the prompt is never resolved as successful.
            """
            default_per_file, default_aggregate = _get_modal_upload_limits()
            specified_aggregate: List[int] = []
            total_size = 0

            for spec in file_specs:
                policy = spec.file_policy or {}
                max_files = _coerce_positive_int(policy.get("max_files", 1), 1)
                min_files = max(
                    0, _coerce_positive_int(policy.get("min_files", 0), 0)
                )
                count = len(spec.attachments)
                if count > max_files:
                    return (
                        f"Upload rejected: you attached {count} file(s) to "
                        f"'{spec.field_key}', but the limit is {max_files}."
                    )
                if count < min_files:
                    return (
                        f"Upload rejected: '{spec.field_key}' requires at least "
                        f"{min_files} file(s); you attached {count}."
                    )

                per_file_limit = _coerce_positive_int(
                    policy.get("max_bytes"),
                    default_per_file,
                )
                mtb = policy.get("max_total_bytes")
                if mtb is not None:
                    specified_aggregate.append(_coerce_positive_int(mtb, default_aggregate))

                for att in spec.attachments:
                    size = getattr(att, "size", None)
                    if size is None:
                        # An unknown size cannot be bounded ahead of the read.
                        # Fail closed rather than perform an unbounded read.
                        return (
                            "Upload rejected: could not determine the size of "
                            "an attached file. Please re-attach it and try "
                            "again."
                        )
                    try:
                        size = int(size)
                    except (TypeError, ValueError):
                        return (
                            "Upload rejected: could not determine the size of "
                            "an attached file. Please re-attach it and try "
                            "again."
                        )
                    if per_file_limit and size > per_file_limit:
                        return (
                            f"Upload rejected: a file in '{spec.field_key}' is too "
                            f"large ({_humanize_bytes(size)} exceeds the "
                            f"{_humanize_bytes(per_file_limit)} per-file limit)."
                        )
                    total_size += size

            # Most-restrictive aggregate cap across all file fields.
            effective_aggregate = (
                min(specified_aggregate) if specified_aggregate else default_aggregate
            )
            if effective_aggregate and total_size > effective_aggregate:
                return (
                    f"Upload rejected: total upload size "
                    f"({_humanize_bytes(total_size)}) exceeds the "
                    f"{_humanize_bytes(effective_aggregate)} limit."
                )
            return None

        async def _collect_files(
            self,
            file_specs: List[_FileSpec],
        ) -> "tuple[List[FileResult], Optional[str]]":
            """Read and cache each attachment under the active Hermes home.

            Returns ``(results, rejection)`` where ``rejection`` is ``None``
            on full success, or a safe message string when any read or write
            failed.  On failure the caller purges ``results`` (the files
            written before the failure) so no partial upload survives.
            """
            import os
            import uuid

            from hermes_constants import get_hermes_home

            cache_dir = os.path.join(get_hermes_home(), "cache", "uploads")
            os.makedirs(cache_dir, exist_ok=True)

            default_per_file, _default_aggregate = _get_modal_upload_limits()
            results: List[FileResult] = []
            for spec in file_specs:
                per_file_limit = _coerce_positive_int(
                    (spec.file_policy or {}).get("max_bytes"),
                    default_per_file,
                )
                for att in spec.attachments:
                    if not hasattr(att, "read"):
                        return results, _UPLOAD_READ_FAILURE_MESSAGE
                    try:
                        data = await att.read()
                    except Exception as read_err:
                        logger.warning(
                            "Failed to download attachment %s for prompt %s: %s",
                            getattr(att, "id", "?"),
                            self.prompt_id,
                            read_err,
                        )
                        return results, _UPLOAD_READ_FAILURE_MESSAGE
                    if data is None:
                        return results, _UPLOAD_READ_FAILURE_MESSAGE
                    # Defense-in-depth: a lying/absent size could slip past the
                    # reported-size gate in Phase 2.  Re-check the actual byte
                    # count before caching so an oversized payload is rejected
                    # (and purged) rather than written to disk.
                    if per_file_limit and len(data) > per_file_limit:
                        return results, (
                            f"Upload rejected: a file in '{spec.field_key}' is too "
                            f"large ({_humanize_bytes(len(data))} exceeds the "
                            f"{_humanize_bytes(per_file_limit)} per-file limit)."
                        )

                    ext = os.path.splitext(getattr(att, "filename", "") or "")[1] or ".bin"
                    cached_path = os.path.join(cache_dir, f"{uuid.uuid4().hex}{ext}")
                    try:
                        with open(cached_path, "wb") as f:
                            f.write(data)
                    except Exception as write_err:
                        logger.warning(
                            "Failed to cache uploaded file for prompt %s: %s",
                            self.prompt_id,
                            write_err,
                        )
                        # The half-written file (if any) must not survive.
                        try:
                            if os.path.exists(cached_path):
                                os.remove(cached_path)
                        except Exception:
                            pass
                        return results, (
                            "Upload rejected: could not save one of your "
                            "files. Please try again."
                        )

                    results.append(
                        FileResult(
                            field_key=spec.field_key,
                            attachment_id=str(getattr(att, "id", "")),
                            filename=getattr(att, "filename", None) or "unknown",
                            content_type=getattr(att, "content_type", None)
                            or "application/octet-stream",
                            size=int(getattr(att, "size", 0) or 0),
                            cached_path=cached_path,
                        )
                    )
            return results, None

        def _purge_cached_files(self, files: List["FileResult"]) -> None:
            """Remove every cache file in ``files`` (cleanup on rejection)."""
            import os

            for fr in files:
                path = getattr(fr, "cached_path", "") or ""
                if not path:
                    continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    logger.debug(
                        "Failed to purge partial cache file %s", path, exc_info=True,
                    )

        async def _send_rejection(
            self,
            interaction: discord.Interaction,
            message: str,
        ) -> None:
            """Acknowledge a rejected submission without resolving the prompt.

            The original prompt is left pending and its buttons stay enabled
            so the user can retry.  ``message`` is always safe — it never
            contains internal paths or exception details.
            """
            logger.info(
                "InteractivePrompt modal upload rejected (id=%s): %s",
                self.prompt_id,
                message,
            )
            try:
                await interaction.response.send_message(
                    f"⚠️ {message}", ephemeral=True,
                )
            except Exception:
                try:
                    await interaction.followup.send(
                        f"⚠️ {message}", ephemeral=True,
                    )
                except Exception:
                    logger.debug(
                        "Could not deliver modal upload rejection", exc_info=True,
                    )

        async def on_error(
            self,
            interaction: discord.Interaction,
            error: Exception,
        ) -> None:
            """Log errors and notify the user."""
            logger.error(
                "InteractivePromptModal on_error (id=%s): %s",
                self.prompt_id,
                error,
                exc_info=error,
            )
            try:
                await interaction.response.send_message(
                    "❌ Something went wrong", ephemeral=True,
                )
            except Exception:
                pass
