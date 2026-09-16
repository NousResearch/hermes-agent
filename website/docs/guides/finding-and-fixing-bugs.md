---
sidebar_position: 2
title: "Finding and Fixing Bugs in Hermes: A Complete Contributor Guide"
description: "How to find real bugs in Hermes Agent, trace root causes, write fixes that get merged, and build a track record as a contributor"
---

# Finding and Fixing Bugs in Hermes: A Complete Contributor Guide

This guide walks you through the full lifecycle of a real contribution to Hermes Agent — from spotting a bug, to understanding the code, writing the fix, opening an issue, and getting your PR merged.

Every example in this guide comes from real bugs found and fixed in the project. No generic open source advice — this is specific to Hermes.

---

## Why Contribute?

Hermes is actively developed and maintained. Bug fixes are the highest-priority contributions — they get reviewed fastest and merged most often. A single well-written bug fix PR builds more credibility than ten feature proposals.

Teknium's merge pattern is clear:

| Gets merged | Gets rejected |
|-------------|---------------|
| Single net bug fix, small scope | Broad refactors |
| Follows existing architecture | Adds complexity |
| Has a regression test | No test |
| One clean commit | Multiple messy commits |
| Doesn't touch "works well" code | Rewrites working systems |

Fix bugs. Keep PRs small. Write tests. That's the formula.

---

## Part 1: How to Find Bugs

There are three ways to find bugs worth fixing. All three work. The best contributors use all three.

### Method 1: Use Hermes and Notice What's Wrong

The most reliable bugs are the ones you experience yourself. When something doesn't behave the way the documentation says it should, that's a bug.

Run Hermes. Use it seriously. Pay attention to:

- Commands that don't respond or respond incorrectly
- Features that work in one context but not another
- Behavior that contradicts what the docs say
- Error messages that appear when they shouldn't
- Silence when you expect output

**Example:** A user sent `/status` while Hermes was processing a task. Instead of getting system metrics, the command was passed to the agent as plain text. The user expected one thing, got another — that's a bug.

### Method 2: Read the Code

You don't need to experience a bug to find one. Reading the code carefully reveals bugs that haven't been reported yet.

Start with a file that interests you. Read it fully. Look for:

- Asymmetric handling — similar cases treated differently without reason
- Missing branches — one code path handles something, another doesn't
- Functions that return values that are never used
- Error handling that silently swallows exceptions
- Comments that describe behavior that doesn't match the code

**Example:** Reading `gateway/platforms/matrix.py` revealed that password login discarded the `LoginResponse` entirely — `access_token`, `device_id`, and `user_id` were never applied to the client. The access-token path handled this correctly. The password path was simply missing the equivalent. No user had reported it — the bug was found by reading.

### Method 3: Read Open Issues

GitHub issues often contain bug reports where the root cause hasn't been found yet. The reporter describes symptoms; your job is to find the cause.

When reading issues, ask:

1. **Is this actually a bug, or a design question?** Issues asking "I want to understand the intended design" often get closed as intended behavior. Look for concrete, reproducible symptoms.

2. **Does this fix really solve the problem, or does it escape from somewhere else?** Trace the full code path before deciding where to fix.

3. **Is there already a competing PR?** Check the issue's linked PRs. If someone opened a PR 10 minutes ago, your chances of getting merged first are low.

---
## Part 2: Is This Really a Bug?

Before writing a single line of code, confirm that you have found a real bug.

### Reproduce it

Can you make the wrong behavior happen reliably? If you cannot reproduce it, you cannot fix it.

For bugs found by code reading, write a small script that demonstrates the incorrect behavior:

    cd ~/hermes-agent
    python3 -c "
    from agent.redact import redact_sensitive_text
    test = 'api_key=api_key,'
    result = redact_sensitive_text(test)
    print('Input: ', repr(test))
    print('Output:', repr(result))
    print('Bug confirmed:', result != test)
    "

If the output shows the wrong behavior, you have confirmed the bug.

### Check the docs

Does the behavior contradict what the documentation says? If the docs say /status returns system metrics and it does not — that is a bug.

### Check if it is intentional

Some things look like bugs but are deliberate design decisions. Teknium keeps the system intentionally simple. Look for comments near the relevant code explaining the choice before assuming something is wrong.

---

## Part 3: Finding the Root Cause

This is the most important step. Fixing the symptom without finding the root cause produces fixes that break other things or do not actually work.

### Read the full code path

Never stop at the first relevant function. Trace the problem upstream and downstream.

Start with grep to find where the relevant code lives:

    cd ~/hermes-agent
    grep -rn "keyword" --include="*.py" | grep -v test | grep -v ".pyc" | head -20

For gateway issues, the path typically looks like:

    User message
      -> Platform adapter  (gateway/platforms/telegram.py)
      -> Base adapter      (gateway/platforms/base.py)
      -> Gateway runner    (gateway/run.py)
      -> Command handler or agent

Read all of them. The bug is rarely in the obvious place.

### Read similar working code

Find something that works correctly and is similar to what is broken. The working code shows you the pattern the broken code should follow.

Example: when /status was not working during active sessions, reading how /approve and /deny were handled revealed the bypass pattern they used. /status was missing the same treatment. The fix was obvious once you saw the working code.

    grep -n "approve" gateway/platforms/base.py | head -20

### Verify with a targeted test

Before writing the fix, write a test that fails because of the bug:

    python3 -m pytest tests/gateway/test_your_file.py -q -p no:xdist -o "addopts=" -v

If you can make a test fail in exactly the way the bug manifests, you are ready to fix it.

---

## Part 4: Opening an Issue

Open the issue before or alongside your PR. Issues create a record of the problem and give Teknium context for why the PR exists.

### What makes a good issue

A good issue has:

- Clear title: describes the bug, not the fix. "Matrix password login silently ignores new messages" not "Fix login"
- Steps to reproduce: exact steps that reliably produce the bug
- Expected behavior: what should happen
- Actual behavior: what actually happens
- Relevant logs: error messages, debug output, tracebacks
- Environment: OS, Python version, Hermes version

### Opening the issue via terminal

    gh issue create \
      --repo NousResearch/hermes-agent \
      --title "Your bug title here" \
      --body "## Steps to Reproduce
    1. Step one
    2. Step two

    ## Expected Behavior
    What should happen.

    ## Actual Behavior
    What actually happens.

    ## Environment
    - OS: Ubuntu 24.04
    - Python: 3.12
    - Hermes: 0.7.0"

### What to skip

Do not include your proposed fix in the issue — save that for the PR. Do not speculate about causes you have not verified. Do not report multiple bugs in one issue.

---

## Part 5: Writing the Fix

### The principle: minimal and targeted

The best fixes change as little as possible in exactly the right place. A 10-line fix that solves the problem cleanly beats a 200-line refactor every time.

Before writing, answer:
- What is the exact root cause?
- What is the minimal change that fixes it?
- Does this break anything else?
- Is there existing code that solves a similar problem I can follow?

### Follow existing patterns

Hermes has consistent patterns throughout the codebase. When fixing something, find how similar problems are already solved and follow the same approach. This makes your fix feel native to the codebase and dramatically increases merge probability.

Example: the /status fix followed the exact same pattern as /approve and /deny:

    # Existing pattern for /approve and /deny:
    if cmd in ("approve", "deny"):
        try:
            response = await self._message_handler(event)
            if response:
                await self._send_with_retry(
                    chat_id=event.source.chat_id,
                    content=response,
                    reply_to=event.message_id,
                    metadata=_thread_meta,
                )
        except Exception as e:
            logger.error("[%s] Approval dispatch failed: %s", self.name, e, exc_info=True)
        return

    # New pattern for /status -- identical structure:
    if cmd == "status":
        try:
            response = await self._message_handler(event)
            if response:
                await self._send_with_retry(
                    chat_id=event.source.chat_id,
                    content=response,
                    reply_to=event.message_id,
                    metadata=_thread_meta,
                )
        except Exception as e:
            logger.error("[%s] Status dispatch failed: %s", self.name, e, exc_info=True)
        return

The fix is immediately recognizable as belonging to the codebase.

### Apply the fix safely

Read the file fully before touching it:

    grep -n "relevant_function" gateway/platforms/base.py | head -20

Apply changes with Python to avoid heredoc issues:

    python3 - << 'APPLY'
    with open("gateway/platforms/base.py", "r") as f:
        content = f.read()

    old = "        # exact text to replace"
    new = "        # new text"

    assert old in content, "Pattern not found -- file may have changed"
    content = content.replace(old, new, 1)

    with open("gateway/platforms/base.py", "w") as f:
        f.write(content)
    print("OK")
    APPLY

Always use assert to confirm the pattern exists before replacing.

### Syntax check after every edit

    python3 -c "import ast; ast.parse(open('gateway/platforms/base.py').read()); print('OK')"

### Check all code paths

    grep -rn "the_function_you_changed" --include="*.py" . | grep -v test | grep -v ".pyc"

Read every result. A fix that works in one code path but breaks another is worse than no fix.

---

## Part 6: Writing Tests

Tests are not optional. A PR without a test has a significantly lower chance of being merged.

### What a good test does

A good test:
- Imports and calls the actual function being fixed, never copies its logic
- Sets up only what is needed, no unnecessary mocks
- Fails before the fix, passes after
- Has a name that describes what it is testing

### Find the test file

    ls tests/gateway/
    ls tests/tools/

Read existing tests in that file before writing yours. Follow the same patterns -- same helper functions, same mock style, same fixture structure.

### Run the test

    python3 -m pytest tests/gateway/test_your_file.py::your_test_name -q -p no:xdist -o "addopts=" -v

It should pass. Then verify the full test file still passes:

    python3 -m pytest tests/gateway/test_your_file.py -q -p no:xdist -o "addopts=" 2>&1 | tail -10

---

## Part 7: The Pre-PR Checklist

Run through this before opening the PR. Every skipped item reduces merge probability.

    # 1. Syntax check
    python3 -c "import ast; ast.parse(open('your_file.py').read()); print('OK')"

    # 2. Full test run
    python3 -m pytest tests/ -q -p no:xdist -o "addopts=" \
      --ignore=tests/run_interrupt_test.py \
      --ignore=tests/test_interactive_interrupt.py \
      --ignore=tests/acp \
      --ignore=tests/test_real_interrupt_subagent.py \
      --ignore=tests/test_quick_commands.py 2>&1 | tail -15

    # 3. Check diff -- only relevant files?
    git diff upstream/main..HEAD --stat

    # 4. Check commit -- one commit, correct message?
    git log --oneline upstream/main..HEAD

    # 5. Rebase
    git fetch upstream && git rebase upstream/main

To check if a failure is pre-existing:

    git stash
    python3 -m pytest tests/the/failing/test.py -q -p no:xdist -o "addopts="
    git stash pop

---

## Part 8: Opening the PR

### Create the branch correctly

Always branch from upstream/main, not from your fork's main:

    git fetch upstream
    git checkout upstream/main -b fix/your-fix-description

Make your changes, then commit once:

    git add path/to/changed/file.py tests/path/to/test_file.py
    git commit -m "fix(scope): short description"

Push and open the PR:

    git push origin fix/your-fix-description

    gh pr create \
      --title "fix: short description" \
      --body "**Problem**: What is broken and how does a user experience it?

    **Root Cause**: What in the code causes the problem?

    **Fix**: What changed and why it is correct.

    **Tests**: What the new test verifies." \
      --repo NousResearch/hermes-agent \
      --head yourusername:fix/your-fix-description \
      --base main

### What makes a good PR description

- Problem: one paragraph, user-facing symptom, issue number
- Root Cause: specific -- file, function, what was missing or wrong
- Fix: what changed, why it is correct
- Tests: what the new test verifies

Do not over-explain. Do not apologize. Do not ask 'is this the right approach?' -- figure that out before opening.

---

## Part 9: After the PR

### CI failures

Check the CI results after opening. If tests fail, check if they failed on upstream too:

    git stash
    python3 -m pytest tests/the/failing/test.py -q -p no:xdist -o "addopts="
    git stash pop

If the test fails on upstream too -- it is a pre-existing failure, note it in the PR.

### The salvage pattern

Teknium occasionally cherry-picks multiple community PRs into a single consolidated PR, preserving original authorship in git. Your commit can land in main even if your PR itself gets closed.

To maximize the chance of being salvaged:
- Write clean, standalone commits
- Keep the scope tight -- one logical change
- Do not mix multiple fixes in a single commit

### Be patient

Teknium reviews in batches. A good PR that does not get merged in 24 hours is not rejected -- it is waiting. Keep the branch rebased if upstream moves.

---

## Summary

The workflow that gets contributions merged:

    # 1. Find the bug (use Hermes, read code, or read issues)
    # 2. Reproduce it -- confirm it is real
    # 3. Read the FULL code path before touching anything
    # 4. Find the root cause -- not just the symptom

    git fetch upstream && git checkout upstream/main -b fix/description

    # 5. Open an issue on GitHub
    # 6. Write the minimal fix following existing patterns

    python3 -c "import ast; ast.parse(open('file.py').read()); print('OK')"

    # 7. Write a test that fails before fix, passes after
    # 8. Run full test suite

    python3 -m pytest tests/ -q -p no:xdist -o "addopts=" \
      --ignore=tests/run_interrupt_test.py \
      --ignore=tests/test_interactive_interrupt.py \
      --ignore=tests/acp \
      --ignore=tests/test_real_interrupt_subagent.py \
      --ignore=tests/test_quick_commands.py 2>&1 | tail -15

    # 9. Check diff and commit
    git diff upstream/main..HEAD --stat
    git log --oneline upstream/main..HEAD

    # 10. Rebase and push
    git fetch upstream && git rebase upstream/main
    git push origin fix/description

    # 11. Open PR
    gh pr create --title "fix: ..." --body "..." \
      --repo NousResearch/hermes-agent \
      --head yourusername:fix/description --base main

The most important habit: before fixing anything, ask 'is this where the bug actually is, or just where the symptom appears?' That question alone separates fixes that get merged from fixes that do not.
