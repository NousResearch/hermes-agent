#!/bin/bash
# Hermes Agent UI Launcher
# 
# This script starts both the API server and UI application.
# It will run them in the background and provide a clean shutdown.

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Hermes Agent UI Launcher${NC}"
echo "================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ -d "../../env" ]; then
    echo -e "${GREEN}✓ Activating virtual environment${NC}"
    source ../../env/bin/activate
else
    echo -e "${BLUE}ℹ No virtual environment found, using system Python${NC}"
fi

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import PySide6" 2>/dev/null || {
    echo -e "${RED}❌ PySide6 not installed${NC}"
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r ../requirements.txt
}

# Check for API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}⚠️  Warning: ANTHROPIC_API_KEY not set${NC}"
    echo "   Set it with: export ANTHROPIC_API_KEY='your-key'"
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${BLUE}🛑 Shutting down Hermes Agent...${NC}"
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✓ API Server stopped${NC}"
    fi
    if [ ! -z "$UI_PID" ]; then
        kill $UI_PID 2>/dev/null || true
        echo -e "${GREEN}✓ UI Application stopped${NC}"
    fi
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Start API server in background
echo -e "${BLUE}Starting API Server...${NC}"
cd ../api_endpoint
python3 logging_server.py > /tmp/hermes_server.log 2>&1 &
SERVER_PID=$!
cd ../ui

# Wait for server to start
echo -e "${BLUE}Waiting for server to start...${NC}"
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start. Check /tmp/hermes_server.log${NC}"
    tail -20 /tmp/hermes_server.log
    exit 1
fi

# Check if server is responding
if curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✓ API Server running on http://localhost:8000${NC}"
else
    echo -e "${RED}❌ Server not responding. Check /tmp/hermes_server.log${NC}"
    exit 1
fi

# Start UI application
echo -e "${BLUE}Starting UI Application...${NC}"
python3 hermes_ui.py &
UI_PID=$!

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ Hermes Agent UI is running!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${BLUE}📊 Component Status:${NC}"
echo -e "   API Server:  http://localhost:8000 (PID: $SERVER_PID)"
echo -e "   UI App:      Running (PID: $UI_PID)"
echo -e "   Server Log:  /tmp/hermes_server.log"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for UI to exit
wait $UI_PID

# Cleanup will be triggered by trap

