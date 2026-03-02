"""
Tests for the weather tool.
Run: python tests/test_weather.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.weather import run


def run_test(name: str, coro):
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    print('='*50)
    result = asyncio.run(coro)
    print(f"Result: {result}")
    return result


def test_valid_city():
    result = run_test("Valid city (Istanbul)", run(city="Istanbul"))
    assert result["success"] is True
    assert "temperature" in result["result"]
    assert "condition" in result["result"]
    print("✅ PASS")
    return result


def test_celsius_default():
    result = run_test("Celsius default", run(city="Tokyo"))
    assert result["success"] is True
    assert "°C" in result["result"]["temperature"]
    print("✅ PASS")


def test_fahrenheit():
    result = run_test("Fahrenheit units", run(city="New York", units="fahrenheit"))
    assert result["success"] is True
    assert "°F" in result["result"]["temperature"]
    print("✅ PASS")


def test_invalid_city():
    result = run_test("Invalid city", run(city="XxNotACityxX123"))
    assert result["success"] is False
    print("✅ PASS (correctly returned error)")


def test_response_structure():
    result = run_test("Response structure check", run(city="London"))
    assert result["success"] is True
    r = result["result"]
    for key in ["city", "country", "condition", "temperature", "feels_like", "humidity", "wind_speed"]:
        assert key in r, f"Missing key: {key}"
    print("✅ PASS — all fields present")


if __name__ == "__main__":
    print("\n🌤️  Weather Tool Test Suite")
    print("Testing Hermes self-generated tool...\n")

    passed = 0
    failed = 0

    tests = [
        test_valid_city,
        test_celsius_default,
        test_fahrenheit,
        test_invalid_city,
        test_response_structure,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed.")
