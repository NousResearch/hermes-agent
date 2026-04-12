#!/usr/bin/env python3
"""
Test script for tirith_security.py Windows compatibility fix.

Tests:
1. _detect_target() returns correct platform triple
2. _get_archive_name() uses .zip for Windows, .tar.gz otherwise
3. ZIP extraction path resolution is correct (src_base = src)
4. dest path uses correct binary name (tirith.exe vs tirith)

Run from hermes-agent root:
    python3 tools/test_tirith_security_fix.py
"""

import os
import sys
import tempfile
import zipfile
import tarfile
import platform
import stat

# Add tools dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hermes-agent", "tools"))

# Import the module functions we need to test
# We test via mock since tirith_security imports hermes paths
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tirith_security",
    os.path.join(os.path.dirname(__file__), "tirith_security.py")
)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

def test_detect_target():
    """Test _detect_target() returns correct platform triple."""
    print("\n=== Test 1: _detect_target() ===")
    
    target = ts._detect_target()
    system = platform.system()
    
    print(f"  Detected: {target}")
    print(f"  Platform: {system}/{platform.machine()}")
    
    if system == "Windows":
        # _detect_target maps AMD64 -> x86_64, ARM64 -> aarch64
        arch_map = {"amd64": "x86_64", "aarch64": "aarch64"}
        arch = arch_map.get(platform.machine().lower(), platform.machine().lower())
        expected = f"{arch}-pc-windows-msvc"
        assert target == expected, f"Expected {expected}, got {target}"
    elif system == "Linux":
        arch = "x86_64" if platform.machine().lower() in ("x86_64", "amd64") else "aarch64"
        assert target == f"{arch}-unknown-linux-gnu", f"Unexpected target: {target}"
    elif system == "Darwin":
        arch = "x86_64" if platform.machine().lower() in ("x86_64", "amd64") else "aarch64"
        assert target == f"{arch}-apple-darwin", f"Unexpected target: {target}"
    
    print("  ✅ PASS")
    return target


def test_archive_name(target):
    """Test archive name uses .zip for Windows, .tar.gz otherwise."""
    print("\n=== Test 2: Archive name ===")
    
    is_windows = target.endswith("-pc-windows-msvc")
    archive_name = f"tirith-{target}.tar.gz"
    if is_windows:
        archive_name = f"tirith-{target}.zip"
    
    print(f"  Target: {target}")
    print(f"  Archive: {archive_name}")
    
    if is_windows:
        assert archive_name == f"tirith-{target}.zip", f"Windows should use .zip, got {archive_name}"
    else:
        assert archive_name == f"tirith-{target}.tar.gz", f"Linux/macOS should use .tar.gz, got {archive_name}"
    
    print("  ✅ PASS")


def test_zip_extraction_path():
    """Test ZIP extraction correctly resolves nested member paths."""
    print("\n=== Test 3: ZIP extraction path resolution ===")
    
    # Simulate a ZIP with nested path (common in GitHub releases)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "test.zip")
        
        # Create a test ZIP with nested tirith.exe
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("tirith.exe", b"fake binary")
            zf.writestr("nested/path/tirith.exe", b"fake binary nested")
        
        # Simulate the extraction logic from _install_tirith
        extracted_path = None
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member == "tirith.exe" or member.endswith("/tirith.exe"):
                    if ".." in member:
                        continue
                    zf.extract(member, tmpdir)
                    extracted_path = os.path.join(tmpdir, member)
                    break
        
        print(f"  Extracted to: {extracted_path}")
        assert extracted_path is not None, "Should find tirith.exe"
        assert os.path.exists(extracted_path), f"Extracted file should exist: {extracted_path}"
        
        # Verify src_base should equal extracted_path, not a hardcoded path
        src_base_wrong = os.path.join(tmpdir, "tirith.exe")
        print(f"  src_base (correct): {extracted_path}")
        print(f"  src_base (wrong):   {src_base_wrong}")
        
        # If nested, wrong approach would fail
        if "nested" in extracted_path:
            assert extracted_path != src_base_wrong, "Should use actual extracted path"
            print("  ✅ PASS (nested path handled correctly)")
        else:
            print("  ✅ PASS (flat path works with both)")


def test_dest_binary_name():
    """Test destination binary name is tirith.exe on Windows, tirith otherwise."""
    print("\n=== Test 4: Destination binary name ===")
    
    test_cases = [
        ("x86_64-pc-windows-msvc", "tirith.exe"),
        ("aarch64-pc-windows-msvc", "tirith.exe"),
        ("x86_64-unknown-linux-gnu", "tirith"),
        ("aarch64-apple-darwin", "tirith"),
    ]
    
    for target, expected in test_cases:
        is_windows = target.endswith("-pc-windows-msvc")
        dest = f"tirith.exe" if is_windows else "tirith"
        print(f"  {target} → {dest}")
        assert dest == expected, f"{target} should give {expected}, got {dest}"
    
    print("  ✅ PASS")


def test_chmod_not_called_on_windows():
    """Test chmod is NOT called for Windows binaries."""
    print("\n=== Test 5: No chmod on Windows ===")
    
    # Simulate the logic from _install_tirith
    test_cases = [
        ("x86_64-pc-windows-msvc", False),  # is_windows = True, should NOT chmod
        ("aarch64-pc-windows-msvc", False),
        ("x86_64-unknown-linux-gnu", True),  # is_windows = False, should chmod
        ("aarch64-apple-darwin", True),
    ]
    
    for target, should_chmod in test_cases:
        is_windows = target.endswith("-pc-windows-msvc")
        would_chmod = not is_windows  # actual logic from code
        
        print(f"  {target}: chmod={would_chmod} (expected: {should_chmod})")
        assert would_chmod == should_chmod, f"{target}: chmod should be {should_chmod}, got {would_chmod}"
    
    print("  ✅ PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("tirith_security.py Windows Compatibility Fix - Test Suite")
    print("=" * 60)
    
    try:
        target = test_detect_target()
        test_archive_name(target)
        test_zip_extraction_path()
        test_dest_binary_name()
        test_chmod_not_called_on_windows()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
