"""
Hermes Agent UI Package

A modular PySide6 UI for the Hermes AI Agent with real-time event streaming.

Modules:
- websocket_client: WebSocket communication
- event_widgets: Event display components
- main_window: Main application window
- hermes_ui: Application entry point
"""

from .websocket_client import WebSocketClient
from .event_widgets import CollapsibleEventWidget, InteractiveEventDisplayWidget
from .main_window import HermesMainWindow

__all__ = [
    'WebSocketClient',
    'CollapsibleEventWidget',
    'InteractiveEventDisplayWidget',
    'HermesMainWindow',
]

