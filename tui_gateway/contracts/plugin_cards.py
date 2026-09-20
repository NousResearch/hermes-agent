"""Shared declarative plugin-card value shapes (no registry side effects)."""

from pydantic import Field

from .base import Result


class PluginCardActionWire(Result):
    label: str
    command: str
    args: str


class PluginCardWire(Result):
    plugin_id: str
    plugin_name: str
    title: str
    body: str
    actions: list[PluginCardActionWire] = Field(min_length=1)
