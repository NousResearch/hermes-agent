#!/usr/bin/env python3
"""
Hermes Agent - PySide6 Frontend

A modern desktop UI for the Hermes AI Agent with real-time event streaming.

Features:
- Query input with multi-line support
- Tool/toolset selection
- Model and API configuration
- Real-time event display via WebSocket
- Beautiful, responsive UI with dark theme
- Session history
- Safe exit handling (no segfaults)

Usage:
    python hermes_ui.py
"""

import sys
import signal
import os

# Suppress Qt logging warnings BEFORE importing Qt
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from main_window import HermesMainWindow


def setup_signal_handlers(app: QApplication) -> QTimer:
    """
    Setup signal handlers for graceful shutdown on Ctrl+C.
    
    This prevents segmentation faults by:
    1. Catching SIGINT/SIGTERM signals
    2. Creating a timer that keeps Python responsive to signals
    3. Calling app.quit() for proper Qt cleanup
    
    Args:
        app: The QApplication instance
        
    Returns:
        Timer that keeps Python interpreter responsive to signals
    """
    def signal_handler(signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n🛑 Interrupt received, shutting down gracefully...")
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # CRITICAL: Create a timer to wake up Python interpreter periodically
    # This allows Python to process signals while Qt's event loop is running
    # Without this, Ctrl+C will not work and may cause segfaults
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # Empty callback just to wake up Python
    timer.start(100)  # Check every 100ms
    
    return timer


def main():
    """Main entry point for the application."""
    # Create application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Hermes Agent")
    app.setOrganizationName("Hermes")
    app.setApplicationVersion("1.0.0")
    
    # Setup signal handlers for safe Ctrl+C handling (prevents segfaults!)
    timer = setup_signal_handlers(app)
    
    # Apply dark theme (optional)
    # Uncomment to enable dark mode
    # app.setStyle("Fusion")
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # app.setPalette(palette)
    
    # Create and show main window
    window = HermesMainWindow()
    window.show()
    
    print("✨ Hermes Agent UI started")
    print("   Press Ctrl+C to exit gracefully")
    
    # Start event loop
    exit_code = app.exec()
    
    print("👋 Hermes Agent UI closed")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()