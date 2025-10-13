"""
Event display widgets for Hermes Agent UI.

This module provides widgets for displaying and managing real-time agent events
in a collapsible, filterable interface.
"""

import json
from datetime import datetime
from typing import Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont


class CollapsibleEventWidget(QFrame):
    """
    A single collapsible event with expand/collapse functionality.
    """

    def __init__(self, event: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.event = event
        self.is_expanded = False
        self.event_type = event.get("event_type", "unknown")

        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setup_ui()

    def setup_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header (clickable)
        self.header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.expand_indicator = QLabel("▶")
        self.expand_indicator.setFixedWidth(20)
        header_layout.addWidget(self.expand_indicator)

        self.summary_label = QLabel()
        self.summary_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.update_summary()
        header_layout.addWidget(self.summary_label, 1)

        # Timestamp
        timestamp = self.event.get("timestamp", datetime.now().isoformat())
        time_str = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%H:%M:%S")
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #888;")
        header_layout.addWidget(time_label)

        self.header_widget.setLayout(header_layout)
        self.header_widget.mousePressEvent = lambda e: self.toggle_expand()
        self.header_widget.setCursor(Qt.PointingHandCursor)
        
        layout.addWidget(self.header_widget)
        
        # Details (collapsible)
        self.details_widget = QWidget()
        self.details_layout = QVBoxLayout()
        self.details_layout.setContentsMargins(25, 5, 5, 5)
        self.populate_details()
        self.details_widget.setLayout(self.details_layout)
        self.details_widget.setVisible(False)
        
        layout.addWidget(self.details_widget)
        
        self.setLayout(layout)
        self.apply_colors()

    def apply_colors(self):
        """Apply color scheme based on event type."""
        colors = {
            "query": "#E8F5E9",      # Light green
            "api_call": "#E3F2FD",   # Light blue
            "response": "#F3E5F5",   # Light purple
            "tool_call": "#FFF3E0",  # Light orange
            "tool_result": "#E8F5E9", # Light green
            "complete": "#E8F5E9",   # Light green
            "error": "#FFEBEE",      # Light red
            "session_start": "#F5F5F5" # Light gray
        }
        
        bg_color = colors.get(self.event_type, "#FAFAFA")
        self.setStyleSheet(f"""
            CollapsibleEventWidget {{
                background-color: {bg_color};
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
        """)

    def update_summary(self):
        """Update the summary label with event type."""
        self.summary_label.setText(f"- {self.event_type.upper()}")

    def populate_details(self):
        """Populate the details section with event data."""
        data = self.event.get("data", {})

        # Clear existing details
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.add_detail("Raw Data", json.dumps(data, indent=2), multiline=True)

    def add_detail(self, label: str, value: str, multiline: bool = True):
        """Add a detail row to the details section."""
        detail_widget = QWidget()
        detail_layout = QVBoxLayout() if multiline else QHBoxLayout()
        detail_layout.setContentsMargins(0, 2, 0, 2)
        
        label_widget = QLabel(f"<b>{label}:</b>")
        label_widget.setTextFormat(Qt.RichText)
        
        value_widget = QLabel(value)
        value_widget.setWordWrap(True)
        value_widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        if multiline:
            font = QFont()
            font.setStyleHint(QFont.Monospace)
            font.setPointSize(9)
            value_widget.setFont(font)
            value_widget.setStyleSheet("background-color: #f5f5f5; padding: 5px; border-radius: 3px;")
            detail_layout.addWidget(label_widget)
            detail_layout.addWidget(value_widget)
        else:
            detail_layout.addWidget(label_widget)
            detail_layout.addWidget(value_widget, 1)
        
        detail_widget.setLayout(detail_layout)
        self.details_layout.addWidget(detail_widget)

    def toggle_expand(self):
        """Toggle expanded/collapsed state."""
        self.is_expanded = not self.is_expanded
        self.details_widget.setVisible(self.is_expanded)
        self.expand_indicator.setText("▼" if self.is_expanded else "▶")


class InteractiveEventDisplayWidget(QWidget):
    """
    Interactive widget for displaying real-time agent events.
    
    Features:
    - Collapsible event items
    - Event type filtering
    - Expand/collapse all
    - Auto-scroll to latest events
    """

    def __init__(self):
        super().__init__()
        self.events = []
        self.event_widgets = []
        self.current_session = None
        self.filters = {
            "query": True,
            "api_call": True,
            "response": True,
            "tool_call": True,
            "tool_result": True,
            "complete": True,
            "error": True,
            "session_start": True
        }
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        title = QLabel("📡 Real-time Event Stream")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Expand/Collapse All buttons
        expand_all_btn = QPushButton("Expand All")
        expand_all_btn.clicked.connect(self.expand_all)
        header_layout.addWidget(expand_all_btn)
        
        collapse_all_btn = QPushButton("Collapse All")
        collapse_all_btn.clicked.connect(self.collapse_all)
        header_layout.addWidget(collapse_all_btn)
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.clear_events)
        header_layout.addWidget(clear_btn)
        
        layout.addLayout(header_layout)
        
        # Filter controls
        filter_group = QGroupBox("Event Filters (Show/Hide)")
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        
        self.filter_checkboxes = {}
        filter_configs = [
            ("query", "📝 Queries"),
            ("api_call", "🔄 API Calls"),
            ("response", "🤖 Responses"),
            ("tool_call", "🔧 Tool Calls"),
            ("tool_result", "✅ Results"),
            ("complete", "🎉 Complete"),
            ("error", "❌ Errors"),
        ]
        
        for event_type, label in filter_configs:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, et=event_type: self.toggle_filter(et, state))
            self.filter_checkboxes[event_type] = checkbox
            filter_layout.addWidget(checkbox)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Scroll area for events
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container for event widgets
        self.events_container = QWidget()
        self.events_layout = QVBoxLayout()
        self.events_layout.setSpacing(5)
        self.events_layout.addStretch()  # Push events to top
        self.events_container.setLayout(self.events_layout)
        
        scroll_area.setWidget(self.events_container)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
    
    def clear_events(self):
        """Clear all displayed events."""
        self.events.clear()
        self.event_widgets.clear()
        
        # Remove all widgets
        while self.events_layout.count() > 1:  # Keep the stretch
            item = self.events_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.current_session = None
    
    def add_event(self, event: Dict[str, Any]):
        """Add an event to the display."""
        event_type = event.get("event_type", "unknown")
        session_id = event.get("session_id", "")
        
        # Track session changes - add session start event
        if self.current_session != session_id:
            self.current_session = session_id
            session_event = {
                "event_type": "session_start",
                "session_id": session_id,
                "timestamp": event.get("timestamp", datetime.now().isoformat()),
                "data": {
                    "session_id": session_id,
                    "start_time": event.get("timestamp", datetime.now().isoformat())
                }
            }
            self._add_event_widget(session_event)
        
        # Add the actual event
        self._add_event_widget(event)
    
    def _add_event_widget(self, event: Dict[str, Any]):
        """Internal method to add event widget."""
        event_widget = CollapsibleEventWidget(event)
        
        # Apply filter visibility
        event_type = event.get("event_type", "unknown")
        event_widget.setVisible(self.filters.get(event_type, True))
        
        # Insert before the stretch
        self.events_layout.insertWidget(self.events_layout.count() - 1, event_widget)
        
        self.events.append(event)
        self.event_widgets.append(event_widget)
        
        # Auto-scroll to bottom after widget is rendered
        QTimer.singleShot(50, self._scroll_to_bottom)
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the events list."""
        scroll_area = self.events_container.parent()
        if isinstance(scroll_area, QScrollArea):
            scroll_bar = scroll_area.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
    
    def expand_all(self):
        """Expand all event widgets."""
        for widget in self.event_widgets:
            if not widget.is_expanded:
                widget.toggle_expand()
    
    def collapse_all(self):
        """Collapse all event widgets."""
        for widget in self.event_widgets:
            if widget.is_expanded:
                widget.toggle_expand()
    
    def toggle_filter(self, event_type: str, state: int):
        """Toggle visibility of events by type."""
        self.filters[event_type] = bool(state)
        
        # Update visibility of existing widgets
        for event, widget in zip(self.events, self.event_widgets):
            if event.get("event_type") == event_type:
                widget.setVisible(self.filters[event_type])

