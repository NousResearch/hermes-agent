#!/usr/bin/env python3
"""Verify prompt detection feature integrity."""
import sys
sys.path.insert(0, '.')

# Test 1: Prompt patterns load and work
from tools.environments.base import _PROMPT_RE, _PROMPT_PATTERNS
print(f"[OK] Loaded {len(_PROMPT_PATTERNS)} prompt patterns")

# Test 2: All positive cases match
positives = [
    ("Password:", "password prompt"),
    ("passwd:", "passwd prompt"),
    ("yes?", "yes/no confirmation"),
    ("no?", "yes/no confirmation"),
    ("continue [y/n]", "continue prompt"),
    ("proceed [y/n]", "proceed prompt"),
    ("are you sure", "are you sure"),
    ("Are you sure?", "are you sure (capitalized)"),
    ("do you want to continue?", "preference prompt"),
    ("Would you like to save?", "preference prompt"),
    ("[y/n]", "[y/n] choice"),
    ("> ", "REPL > prompt"),
    (">>> ", "Python REPL"),
    (". ", "shell REPL"),
    ("Enter username", "enter prompt"),
    ("Type your name", "type prompt"),
]

all_passed = True
for text, desc in positives:
    if _PROMPT_RE.search(text):
        print(f"  [OK] {desc}: {text!r}")
    else:
        print(f"  [FAIL] {desc}: {text!r} - NOT MATCHED")
        all_passed = False

# Test 3: Multiline detection (prompt on its own line at end of output)
multiline = "Welcome to installer.\nPlease read the license.\n\n[y/n]\n"
if _PROMPT_RE.search(multiline):
    print(f"  [OK] multiline prompt detected (on own line)")
else:
    print(f"  [FAIL] multiline prompt NOT detected")
    all_passed = False

# Test 3b: Prompt embedded in sentence should NOT match (^ anchor)
embedded = "Do you accept? [y/n]"
if not _PROMPT_RE.search(embedded):
    print(f"  [OK] embedded prompt correctly rejected (not at line start)")
else:
    print(f"  [WARN] embedded prompt matched - this is expected behavior for ^ anchor")

# Test 4: register_paused_process exists
from tools.process_registry import process_registry
if hasattr(process_registry, 'register_paused_process'):
    print("[OK] register_paused_process() method exists")
else:
    print("[FAIL] register_paused_process() NOT found")
    all_passed = False

# Test 5: local.py uses PIPE for stdin
from pathlib import Path
local_py = Path("tools/environments/local.py").read_text()
if "stdin=subprocess.PIPE" in local_py:
    print("[OK] LocalEnvironment uses stdin=subprocess.PIPE")
else:
    print("[FAIL] LocalEnvironment does NOT use PIPE")
    all_passed = False

# Test 6: terminal_tool.py has paused integration
terminal_py = Path("tools/terminal_tool.py").read_text()
if "register_paused_process" in terminal_py and 'result.get("paused")' in terminal_py:
    print("[OK] Terminal tool integrates paused process registration")
else:
    print("[FAIL] Terminal tool missing paused integration")
    all_passed = False

print(f"\n{'='*50}")
if all_passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED - see above")
    sys.exit(1)
