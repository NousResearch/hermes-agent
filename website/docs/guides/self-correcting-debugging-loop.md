---
sidebar_position: 11
title: "Build a Self-Correcting Debugging Loop with Hermes"
description: "Run code, inspect failures, patch files, and verify fixes in a tight debugging loop with Hermes Agent"
---

# Build a Self-Correcting Debugging Loop with Hermes

Hermes works best on tasks with a clear feedback loop:

1. run something
2. inspect the result
3. make a targeted change
4. verify the fix

Debugging is a perfect example.

In this guide, you'll use Hermes to run a failing test suite, read the failures, patch the code, and re-run verification until the project is green again.

## What You'll Build

By the end of this guide, you'll have a small broken Python project and a repeatable Hermes workflow that:

- runs tests
- identifies the real failure
- patches the source code
- re-runs the tests
- stops only when everything passes

```text
Broken code
    ↓
Hermes runs tests
    ↓
Hermes reads the failures
    ↓
Hermes patches the code
    ↓
Hermes re-runs verification
    ↓
Passing test suite
```

## Why This Pattern Works

Hermes is strongest when:

- the task is bounded
- the failure signal is concrete
- success is easy to verify
- the next step depends on tool output

A debugging loop checks all four boxes.

Good verification targets include:

- `pytest -q`
- `python app.py`
- `npm test`
- a script with a known expected output

The key is to define "done" clearly.

## Prerequisites

- Hermes Agent installed
- terminal and file tools enabled
- Python 3 installed
- `pytest` installed

If Hermes is not installed yet, start with the [Installation guide](/docs/getting-started/installation).

## Step 1: Create a Small Broken Project

Create a new demo directory:

```bash
mkdir -p ~/hermes-debug-demo
cd ~/hermes-debug-demo
```

Create `calculator.py`:

```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        return 0
    return a / b
```

Create `test_calculator.py`:

```python
import pytest
from calculator import add, multiply, divide

def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(4, 3) == 12

def test_divide():
    assert divide(8, 2) == 4

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
```

This project contains two intentional bugs:

- `multiply()` adds instead of multiplying
- `divide()` returns `0` instead of raising `ZeroDivisionError`

## Step 2: Verify the Failure Manually

Run the tests yourself first:

```bash
pytest -q
```

You should see failures.

This gives you a clean baseline and confirms the project is actually broken before Hermes touches anything.

A typical result looks like this:

```text
.F.F                                                                       [100%]
=================================== FAILURES ===================================
______________________________ test_multiply ______________________________

    def test_multiply():
>       assert multiply(4, 3) == 12
E       assert 7 == 12

test_calculator.py:8: AssertionError

___________________________ test_divide_by_zero ___________________________

    def test_divide_by_zero():
>       with pytest.raises(ZeroDivisionError):
E       Failed: DID NOT RAISE <class 'ZeroDivisionError'>

test_calculator.py:14: Failed
=========================== short test summary info ===========================
FAILED test_calculator.py::test_multiply
FAILED test_calculator.py::test_divide_by_zero
2 failed, 2 passed in 0.03s
```

## Step 3: Launch Hermes in the Project Directory

Start Hermes from inside the demo folder:

```bash
hermes
```

Now give it a bounded debugging task:

```text
Debug this project in the current directory.

Requirements:
1. Run the full test suite.
2. Read the failures carefully.
3. Fix the source code, not the tests, unless a test is obviously wrong.
4. Re-run the tests after each meaningful fix.
5. Stop only when the full test suite passes.
6. Summarize what you changed and why.
```

This prompt works well because it defines:

- scope: the current directory
- verification target: the full test suite
- stopping condition: all tests pass
- repair strategy: fix the implementation first

## Step 4: Let Hermes Close the Loop

Hermes should now perform a workflow like this:

1. run `pytest -q`
2. inspect the failing assertions
3. open `calculator.py`
4. patch the broken functions
5. re-run `pytest -q`
6. stop only when the suite passes

A typical result looks like this:

```text
- multiply(4, 3) returned 7 because multiply() was using addition
- divide(10, 0) returned 0 instead of raising ZeroDivisionError
- calculator.py was updated
- pytest was re-run
- all 4 tests passed
```

The important thing here is not just that Hermes proposes a fix.

It executes the full loop:

- observe
- patch
- verify

## Step 5: Ask for a Final Verification Pass

Once Hermes reports success, ask it to verify one more time:

```text
Verify the fix again.

1. Re-run the full test suite.
2. Explain why each failure happened.
3. Confirm there are no remaining failing tests.
4. If anything still fails, continue debugging.
```

That extra pass helps prevent false confidence.

A healthy final output should confirm both:

- the test suite is now green
- the root causes were actually addressed

Expected final result:

```text
....                                                                       [100%]
4 passed in 0.02s
```

## What Hermes Usually Changes

In this example, the correct patch is:

```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
```

The exact phrasing of Hermes' explanation may vary, but the verified end state should be the same: all tests pass, and the implementation now matches the test expectations.

## Best Practices

### Keep the task bounded

Start with:

- one small project
- one script
- one failing command
- one verification target

Avoid prompts like:

```text
Fix everything in this repository.
```

Prefer prompts like:

```text
Run pytest in this directory and fix the source code until all tests pass.
```

### Define "done" explicitly

Hermes performs best when success is concrete.

Good examples:

- "Stop only when `pytest -q` passes"
- "Re-run `python main.py` after each fix"
- "Verify the script output matches the expected result"

### Prefer implementation fixes over silent test rewrites

If your goal is debugging, say so.

Otherwise, an agent may reduce the failure by weakening the tests instead of repairing the code.

### Re-run verification after every meaningful patch

A debugging loop is only reliable if the verification step stays inside the loop.

Don't stop at the first plausible patch.

## Common Pitfalls

### The project is too large

For a first pass, don't start with a large production repository.

Use a small, reproducible target first.

### The environment is broken, not the code

If Hermes cannot run the tests at all, the problem may be:

- missing dependencies
- wrong Python version
- missing environment variables
- incorrect working directory

Get the environment into a runnable state first.

### The prompt is too vague

"Debug this" is weaker than:

```text
Run pytest, fix the source code, and stop only when the full suite passes.
```

### The model is too weak for multi-step debugging

Smaller models may struggle to reason over test failures, tool output, and code edits in the same loop.

If the loop becomes unreliable, use a stronger model.

## Variations

Once this pattern works, you can adapt it to:

- `python app.py` instead of `pytest`
- JavaScript projects with `npm test`
- lint + test loops
- one-file bugfix tasks
- cron-based health checks
- CI-oriented repair workflows

## What’s Next?

If you want to extend this workflow, these docs pair well with it:

- [Tools & Toolsets](/docs/user-guide/features/tools)
- [Sessions](/docs/user-guide/sessions)
- [Profiles](/docs/user-guide/profiles)
- [Cron](/docs/user-guide/features/cron)
- [Skills System](/docs/user-guide/features/skills)

The main pattern is simple:
- Give Hermes a bounded task
- Define a concrete verification command
- Tell it when to stop
- Keep the fix loop grounded in tool output
