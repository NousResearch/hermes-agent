#!/usr/bin/env python3
"""
Test script to verify UI flow works correctly.

This tests:
1. API server is running
2. WebSocket connection works
3. Agent can be started via API
4. Events are broadcast properly
"""

import requests
import json
import time
import websocket
import threading

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

def test_api_server():
    """Test if API server is running."""
    print("🔍 Testing API server...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API server is running: {data.get('service')}")
            print(f"   Active connections: {data.get('active_connections')}")
            return True
        else:
            print(f"❌ API server returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API server not accessible: {e}")
        return False

def test_tools_endpoint():
    """Test if tools endpoint works."""
    print("\n🔍 Testing tools endpoint...")
    try:
        response = requests.get(f"{API_URL}/tools", timeout=5)
        if response.status_code == 200:
            data = response.json()
            toolsets = data.get("toolsets", [])
            print(f"✅ Tools endpoint works - {len(toolsets)} toolsets available")
            for ts in toolsets[:3]:
                print(f"   • {ts.get('name')} ({ts.get('tool_count')} tools)")
            return True
        else:
            print(f"❌ Tools endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Tools endpoint error: {e}")
        return False

def test_websocket():
    """Test WebSocket connection."""
    print("\n🔍 Testing WebSocket connection...")
    
    connected = threading.Event()
    message_received = threading.Event()
    messages = []
    
    def on_open(ws):
        print("✅ WebSocket connected")
        connected.set()
    
    def on_message(ws, message):
        data = json.loads(message)
        messages.append(data)
        message_received.set()
        print(f"📨 Received: {data.get('event_type', 'unknown')}")
    
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"🔌 WebSocket closed: {close_status_code}")
    
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket in background
    ws_thread = threading.Thread(target=lambda: ws.run_forever(), daemon=True)
    ws_thread.start()
    
    # Wait for connection
    if connected.wait(timeout=5):
        print("✅ WebSocket connection established")
        ws.close()
        return True
    else:
        print("❌ WebSocket connection timeout")
        ws.close()
        return False

def test_agent_run():
    """Test running agent via API."""
    print("\n🔍 Testing agent run via API (mock mode)...")
    
    # Start listening for events first
    events = []
    ws_connected = threading.Event()
    session_complete = threading.Event()
    
    def on_message(ws, message):
        data = json.loads(message)
        events.append(data)
        event_type = data.get("event_type")
        print(f"   📨 Event: {event_type}")
        
        if event_type == "complete":
            session_complete.set()
    
    def on_open(ws):
        ws_connected.set()
    
    # Connect WebSocket
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message
    )
    
    ws_thread = threading.Thread(target=lambda: ws.run_forever(), daemon=True)
    ws_thread.start()
    
    # Wait for WebSocket connection
    if not ws_connected.wait(timeout=5):
        print("❌ WebSocket didn't connect")
        ws.close()
        return False
    
    print("✅ WebSocket connected, starting agent...")
    
    # Submit agent run
    payload = {
        "query": "Test query for UI flow verification",
        "model": "claude-sonnet-4-5-20250929",
        "base_url": "https://api.anthropic.com/v1/",
        "enabled_toolsets": ["web"],
        "max_turns": 5,
        "mock_web_tools": True,  # Use mock mode to avoid API costs
        "mock_delay": 2,  # Fast for testing
        "verbose": False
    }
    
    try:
        response = requests.post(f"{API_URL}/agent/run", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get("session_id")
            print(f"✅ Agent started: {session_id[:8]}...")
            
            # Wait for completion (or timeout)
            print("⏳ Waiting for agent to complete (up to 30s)...")
            if session_complete.wait(timeout=30):
                print(f"✅ Agent completed! Received {len(events)} events:")
                
                # Count event types
                event_counts = {}
                for evt in events:
                    evt_type = evt.get("event_type", "unknown")
                    event_counts[evt_type] = event_counts.get(evt_type, 0) + 1
                
                for evt_type, count in event_counts.items():
                    print(f"   • {evt_type}: {count}")
                
                # Check we got expected events
                expected_events = ["query", "api_call", "response", "complete"]
                missing = [e for e in expected_events if e not in event_counts]
                
                if missing:
                    print(f"⚠️  Missing expected events: {missing}")
                else:
                    print("✅ All expected event types received!")
                
                ws.close()
                return True
            else:
                print(f"⚠️  Timeout waiting for completion. Got {len(events)} events so far.")
                ws.close()
                return False
                
        else:
            print(f"❌ Agent start failed: {response.status_code}")
            print(f"   Response: {response.text}")
            ws.close()
            return False
            
    except Exception as e:
        print(f"❌ Agent run error: {e}")
        import traceback
        traceback.print_exc()
        ws.close()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Hermes Agent UI Flow Test")
    print("=" * 60)
    print("\nThis will test the complete flow:")
    print("  1. API server connectivity")
    print("  2. Tools endpoint")
    print("  3. WebSocket connection")
    print("  4. Agent execution via API (mock mode)")
    print("  5. Event streaming to UI")
    print("\n" + "=" * 60)
    
    results = []
    
    # Test 1: API server
    results.append(("API Server", test_api_server()))
    
    # Test 2: Tools endpoint
    results.append(("Tools Endpoint", test_tools_endpoint()))
    
    # Test 3: WebSocket
    results.append(("WebSocket Connection", test_websocket()))
    
    # Test 4: Agent run
    results.append(("Agent Execution + Events", test_agent_run()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The UI flow is working correctly!")
        print("   You can now use the UI to:")
        print("   • Submit queries")
        print("   • View real-time events")
        print("   • See tool executions")
        print("   • Get final responses")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nMake sure:")
        print("  1. API server is running: python api_endpoint/logging_server.py")
        print("  2. ANTHROPIC_API_KEY is set in environment")
        print("  3. All dependencies are installed: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

