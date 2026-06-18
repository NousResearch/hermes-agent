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
from typing import Any, Callable, Dict, List, Optional, Set

# Conditional discord import — gracefully degrades when discord.py is absent.
try:
    import discord
    from discord import ui as _ui
except ImportError:  # pragma: no cover
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
# Auth helper — delegates to shared utility to avoid duplication
# ---------------------------------------------------------------------------

from tools.discord_auth_helpers import component_check_auth as _component_check_auth

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

              * ``session_owner_only`` — restricts to the user who originated
                the session.
              * ``any_allowed_user`` — only the user allowlist is consulted;
                roles are ignored even if present.
              * ``any_allowed_role`` — only the role allowlist is consulted;
                user ID matches are ignored.
              * ``any_allowed_user_or_role`` — user OR role allowlist (union).
            """
            if (
                self.auth_policy == "session_owner_only"
                and self.origin_user_id is not None
            ):
                return str(interaction.user.id) == str(self.origin_user_id)

            # session_owner_only with no origin_user_id: fall through to
            # user allowlist (same as any_allowed_user).  If no allowlists
            # are configured, fail CLOSED — don't silently allow everyone.
            if (
                self.auth_policy == "session_owner_only"
                and self.origin_user_id is None
            ):
                user_set = self.allowed_user_ids or set()
                if not user_set:
                    return False  # fail closed: no owner + no allowlist
                try:
                    return str(interaction.user.id) in user_set
                except AttributeError:
                    return False

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

            self.resolved = True
            self._disable_all()

            action = option.get("action", "return")

            # ── Modal path ──────────────────────────────────────────────
            if action == "modal":
                # Defer resolved + disable until modal actually submits.
                # If the user dismisses the modal, the prompt stays active
                # and can be re-attempted (or a different option chosen).
                self.resolved = False
                self._disable_all()
                modal_spec = option.get("modal", {})
                modal = InteractivePromptModal(
                    prompt_id=self.prompt_id,
                    option_index=index,
                    modal_spec=modal_spec,
                    original_view=self,
                )
                # Disable buttons while modal is open (prevents double-submit)
                self._disable_all()
                try:
                    await interaction.response.edit_message(view=self)
                except Exception:
                    pass
                await interaction.response.send_modal(modal)
                return

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
            files_collected: List[FileResult] = []
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
                # file_upload → download attachments to cache
                elif isinstance(inner, _ui.FileUpload):
                    attachments = getattr(inner, "values", [])
                    file_results = []
                    for att in attachments:
                        try:
                            data = await att.read() if hasattr(att, "read") else None
                        except Exception as read_err:
                            logger.warning(
                                "Failed to download attachment %s for prompt %s: %s",
                                getattr(att, "id", "?"),
                                self.prompt_id,
                                read_err,
                            )
                            data = None
                        cached_path = ""
                        if data:
                            import os
                            import uuid

                            from hermes_constants import get_hermes_home
                            cache_dir = os.path.join(get_hermes_home(), "cache", "uploads")
                            os.makedirs(cache_dir, exist_ok=True)
                            ext = os.path.splitext(att.filename)[1] or ".bin"
                            cached_path = os.path.join(
                                cache_dir, f"{uuid.uuid4().hex}{ext}"
                            )
                            with open(cached_path, "wb") as f:
                                f.write(data)
                        file_results.append(
                            FileResult(
                                field_key=field_key,
                                attachment_id=str(att.id),
                                filename=att.filename or "unknown",
                                content_type=att.content_type
                                or "application/octet-stream",
                                size=att.size or 0,
                                cached_path=cached_path,
                            )
                        )
                    if file_results:
                        files_collected.extend(file_results)
                else:
                    fields[field_key] = getattr(inner, "value", None)

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
