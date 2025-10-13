"""
Main window for Hermes Agent UI.

This module provides the main application window with controls for
submitting queries, configuring settings, and viewing real-time events.
"""

import requests
from typing import Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QListWidget, QListWidgetItem,
    QSpinBox, QMessageBox
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont

from .websocket_client import WebSocketClient
from .event_widgets import InteractiveEventDisplayWidget


class HermesMainWindow(QMainWindow):
    """
    Main window for Hermes Agent UI.
    
    Provides interface for:
    - Submitting queries
    - Configuring agent settings
    - Viewing real-time events
    - Managing sessions
    """
    
    def __init__(self):
        super().__init__()
        self.api_base_url = "http://localhost:8000"
        self.ws_client = None
        self.current_session_id = None
        self.available_toolsets = []
        self.is_closing = False  # Flag to prevent reconnection during shutdown
        
        self.init_ui()
        self.setup_websocket()
        self.load_available_tools()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Hermes Agent - AI Assistant UI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (horizontal split)
        main_layout = QHBoxLayout()
        
        # Left panel: Controls
        left_panel = self.create_control_panel()
        
        # Right panel: Event display
        right_panel = self.create_event_panel()
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Control panel
        splitter.setStretchFactor(1, 2)  # Event panel (larger)
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🤖 Hermes Agent Control")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Query input group
        query_group = QGroupBox("Query Input")
        query_layout = QVBoxLayout()
        
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter your query here...")
        self.query_input.setMaximumHeight(150)
        query_layout.addWidget(self.query_input)
        
        self.submit_btn = QPushButton("🚀 Submit Query")
        self.submit_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.submit_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; }")
        self.submit_btn.clicked.connect(self.submit_query)
        query_layout.addWidget(self.submit_btn)
        
        query_group.setLayout(query_layout)
        layout.addWidget(query_group)
        
        # Model configuration group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Model selection
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-20250514",
            "gpt-4",
            "gpt-4-turbo"
        ])
        model_layout.addWidget(self.model_combo)
        
        # API Base URL
        model_layout.addWidget(QLabel("API Base URL:"))
        self.base_url_input = QLineEdit("https://api.anthropic.com/v1/")
        model_layout.addWidget(self.base_url_input)
        
        # Max turns
        model_layout.addWidget(QLabel("Max Turns:"))
        self.max_turns_spin = QSpinBox()
        self.max_turns_spin.setMinimum(1)
        self.max_turns_spin.setMaximum(50)
        self.max_turns_spin.setValue(10)
        model_layout.addWidget(self.max_turns_spin)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Tools configuration group
        tools_group = QGroupBox("Tools & Toolsets")
        tools_layout = QVBoxLayout()
        
        tools_layout.addWidget(QLabel("Select Toolsets:"))
        self.toolsets_list = QListWidget()
        self.toolsets_list.setSelectionMode(QListWidget.MultiSelection)
        self.toolsets_list.setMaximumHeight(150)
        tools_layout.addWidget(self.toolsets_list)
        
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        self.mock_mode_checkbox = QCheckBox("Mock Web Tools (Testing)")
        options_layout.addWidget(self.mock_mode_checkbox)
        
        self.verbose_checkbox = QCheckBox("Verbose Logging")
        options_layout.addWidget(self.verbose_checkbox)
        
        options_layout.addWidget(QLabel("Mock Delay (seconds):"))
        self.mock_delay_spin = QSpinBox()
        self.mock_delay_spin.setMinimum(1)
        self.mock_delay_spin.setMaximum(300)
        self.mock_delay_spin.setValue(60)
        options_layout.addWidget(self.mock_delay_spin)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Connection status
        self.connection_status = QLabel("🔴 Disconnected")
        self.connection_status.setAlignment(Qt.AlignCenter)
        self.connection_status.setStyleSheet("QLabel { padding: 5px; background-color: #F44336; color: white; border-radius: 3px; }")
        layout.addWidget(self.connection_status)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_event_panel(self) -> QWidget:
        """Create the right event display panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Event display widget
        self.event_widget = InteractiveEventDisplayWidget()
        layout.addWidget(self.event_widget)
        
        panel.setLayout(layout)
        return panel
    
    def setup_websocket(self):
        """Setup WebSocket connection for real-time events."""
        self.ws_client = WebSocketClient("ws://localhost:8000/ws")
        
        # Connect signals
        self.ws_client.connected.connect(self.on_ws_connected)
        self.ws_client.disconnected.connect(self.on_ws_disconnected)
        self.ws_client.error.connect(self.on_ws_error)
        self.ws_client.event_received.connect(self.on_event_received)
        
        # Start connection
        self.ws_client.connect()
    
    @Slot()
    def on_ws_connected(self):
        """Called when WebSocket connection is established."""
        self.connection_status.setText("🟢 Connected")
        self.connection_status.setStyleSheet("QLabel { padding: 5px; background-color: #4CAF50; color: white; border-radius: 3px; }")
        self.statusBar().showMessage("WebSocket connected")
    
    @Slot()
    def on_ws_disconnected(self):
        """Called when WebSocket connection is lost."""
        # Don't attempt reconnection if we're closing the application
        if self.is_closing:
            return
        
        self.connection_status.setText("🔴 Disconnected")
        self.connection_status.setStyleSheet("QLabel { padding: 5px; background-color: #F44336; color: white; border-radius: 3px; }")
        self.statusBar().showMessage("WebSocket disconnected - attempting reconnect...")
        
        # Attempt reconnect after 5 seconds
        QTimer.singleShot(5000, self.ws_client.connect)
    
    @Slot(str)
    def on_ws_error(self, error: str):
        """Called when WebSocket error occurs."""
        self.statusBar().showMessage(f"WebSocket error: {error}")
    
    @Slot(dict)
    def on_event_received(self, event: Dict[str, Any]):
        """
        Called when an event is received from WebSocket.
        
        Args:
            event: Event data from server
        """
        self.event_widget.add_event(event)
        
        # Update status for specific events
        event_type = event.get("event_type")
        if event_type == "query":
            self.statusBar().showMessage("Query received - agent processing...")
        elif event_type == "complete":
            self.statusBar().showMessage("Agent completed!")
            self.submit_btn.setEnabled(True)
    
    def load_available_tools(self):
        """Load available toolsets from the API."""
        try:
            response = requests.get(f"{self.api_base_url}/tools", timeout=5)
            if response.status_code == 200:
                data = response.json()
                toolsets = data.get("toolsets", [])
                
                self.available_toolsets = toolsets
                self.toolsets_list.clear()
                
                for toolset in toolsets:
                    name = toolset.get("name", "")
                    description = toolset.get("description", "")
                    tool_count = toolset.get("tool_count", 0)
                    
                    item_text = f"{name} ({tool_count} tools) - {description}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, name)  # Store toolset name
                    self.toolsets_list.addItem(item)
                
                self.statusBar().showMessage(f"Loaded {len(toolsets)} toolsets")
            else:
                self.statusBar().showMessage("Failed to load toolsets from API")
                
        except requests.exceptions.RequestException as e:
            self.statusBar().showMessage(f"Error loading toolsets: {str(e)}")
            # Add some default toolsets
            default_toolsets = ["web", "vision", "terminal", "research"]
            for ts in default_toolsets:
                item = QListWidgetItem(f"{ts} (default)")
                item.setData(Qt.UserRole, ts)
                self.toolsets_list.addItem(item)
    
    @Slot()
    def submit_query(self):
        """Submit query to the agent API."""
        query = self.query_input.toPlainText().strip()
        
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter a query first!")
            return
        
        # Get selected toolsets
        selected_toolsets = []
        for i in range(self.toolsets_list.count()):
            item = self.toolsets_list.item(i)
            if item.isSelected():
                toolset_name = item.data(Qt.UserRole)
                selected_toolsets.append(toolset_name)
        
        # Build request payload
        payload = {
            "query": query,
            "model": self.model_combo.currentText(),
            "base_url": self.base_url_input.text(),
            "max_turns": self.max_turns_spin.value(),
            "enabled_toolsets": selected_toolsets if selected_toolsets else None,
            "mock_web_tools": self.mock_mode_checkbox.isChecked(),
            "mock_delay": self.mock_delay_spin.value(),
            "verbose": self.verbose_checkbox.isChecked()
        }
        
        # Disable submit button during execution
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("⏳ Running...")
        self.statusBar().showMessage("Submitting query to agent...")
        
        # Submit to API
        try:
            response = requests.post(
                f"{self.api_base_url}/agent/run",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                session_id = result.get("session_id", "")
                self.current_session_id = session_id
                
                self.statusBar().showMessage(f"Agent started! Session: {session_id[:8]}...")
                
                # Clear event display for new session (optional)
                # self.event_widget.clear_events()
                
            else:
                QMessageBox.warning(
                    self,
                    "API Error",
                    f"Failed to start agent: {response.status_code}\n{response.text}"
                )
                self.submit_btn.setEnabled(True)
                self.submit_btn.setText("🚀 Submit Query")
                
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Failed to connect to API server:\n{str(e)}\n\nMake sure the server is running:\npython logging_server.py"
            )
            self.submit_btn.setEnabled(True)
            self.submit_btn.setText("🚀 Submit Query")
        
        # Re-enable button after short delay (UI feedback)
        QTimer.singleShot(2000, lambda: self.submit_btn.setText("🚀 Submit Query"))
    
    def cleanup(self):
        """Clean up resources before exit."""
        print("Cleaning up resources...")
        self.is_closing = True
        
        if self.ws_client:
            try:
                self.ws_client.disconnect()
            except Exception as e:
                print(f"Error disconnecting WebSocket: {e}")
    
    def closeEvent(self, event):
        """Handle window close event - ensures clean shutdown."""
        print("Closing application...")
        self.cleanup()
        event.accept()

