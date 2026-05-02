
import sys
from pathlib import Path

# Add the contrib directory to sys.path so we can import the gateway server
sys.path.append(str(Path(__file__).parent))

try:
    from tui_gateway.server import _get_usage
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class MockResponse:
    def __init__(self, usage):
        self.usage = usage

class MockAgent:
    def __init__(self, usage_in_last_resp=None, attrs=None):
        # Simulate the 'last_response' object
        if usage_in_last_resp:
            self.last_response = MockResponse(usage_in_last_resp)
        else:
            self.last_response = None
        
        # Simulate attributes on the agent itself
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

def test_usage_recovery():
    print("--- Starting Usage Logic Verification ---")
    
    # TEST 1: Data is ONLY in last_response (The "Data Void" scenario)
    # This is exactly what was happening in your Hermes.
    mock_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    agent_void = MockAgent(usage_in_last_resp=mock_usage)
    
    result = _get_usage(agent_void)
    
    print(f"Test 1 (Data Void):")
    print(f"  Expected total: 150")
    print(f"  Actual total:   {result.get('total')}")
    
    if result.get('total') == 150:
        print("✅ PASSED: Recovered data from last_response!")
    else:
        print("❌ FAILED: Still returning 0 or wrong value.")

    # TEST 2: Data is in agent attributes (The "Standard" scenario)
    agent_std = MockAgent(attrs={"session_total_tokens": 300})
    result_std = _get_usage(agent_std)
    
    print(f"\nTest 2 (Standard Attrs):")
    print(f"  Expected total: 300")
    print(f"  Actual total:   {result_std.get('total')}")
    
    if result_std.get('total') == 300:
        print("✅ PASSED: Recovered data from attributes!")
    else:
        print("❌ FAILED: Failed to read attributes.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_usage_recovery()
