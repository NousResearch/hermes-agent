"""
WebSocket client for real-time event streaming from Hermes Agent.

This module provides a WebSocket client that runs in a separate thread
and emits Qt signals when events are received from the server.
"""

import json
import threading
import websocket
from PySide6.QtCore import QObject, Signal


class WebSocketClient(QObject):
    """
    WebSocket client for receiving real-time agent events.
    
    Runs in a separate thread and emits Qt signals when events arrive.
    """
    
    # Signals for event communication
    event_received = Signal(dict)  # Emits parsed event data
    connected = Signal()
    disconnected = Signal()
    error = Signal(str)
    
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        super().__init__()
        self.url = url
        self.ws = None
        self.running = False
        self.thread = None
    
    def connect(self):
        """Start WebSocket connection in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def disconnect(self):
        """Stop WebSocket connection."""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
    
    def _run(self):
        """WebSocket event loop (runs in background thread)."""
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run forever with reconnection
            self.ws.run_forever(ping_interval=300, ping_timeout=60)
            
        except Exception as e:
            self.error.emit(f"WebSocket error: {str(e)}")
    
    def _on_open(self, ws):
        """Called when WebSocket connection is established."""
        print("WebSocket connected")
        self.connected.emit()
    
    def _on_message(self, ws, message):
        """Called when a message is received from the server."""
        try:
            data = json.loads(message)
            self.event_received.emit(data)
        except json.JSONDecodeError as e:
            print(f" Failed to parse WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Called when an error occurs."""
        print(f"WebSocket error: {error}")
        self.error.emit(str(error))
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket connection is closed."""
        print(f"🔌 WebSocket disconnected: {close_status_code} - {close_msg}")
        self.disconnected.emit()

