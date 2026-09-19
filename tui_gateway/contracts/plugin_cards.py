"""Shared declarative plugin-card value shapes (no registry side effects)."""

from .base import Result


class PluginCardActionWire(Result):
    id: str
    label: str
    command: str
    args: str


class PluginCardWire(Result):
    id: str
    plugin_id: str
    plugin_name: str
    title: str
    body: str
    actions: list[PluginCardActionWire]
