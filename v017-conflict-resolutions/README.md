# v0.17.0 conflict resolutions

When the 42 open PRs are pulled down onto v0.17.0 (2bd1977d8), 6 conflict because
v0.17.0 moved lines those PRs also touch. Each `*.v017.patch` here is the isolated,
compile-verified resolution of one PR's conflict on v0.17.0. Strategies:

- #49644 commands.py        : take-theirs (PR's superset /reasoning subcommand list)
- #50033 gemini_cloudcode   : take-theirs (additive google_user_agent import)
- #50056 test_kanban_db     : keep-BOTH imports (v0.17.0's sqlite3 + PR's subprocess)
- #50064 provider_attribution: take-theirs additive — PLUS drop the one test
  `test_routed_client_preserves_openai_sdk_default_headers`, which v0.17.0 itself
  removed (commit 8d59881a6 / #2647 deleted both the test and the routed-default_headers
  behavior). See V017-PER-PR-TEST-RESULTS.txt. The PR's feature (copilot identity) is
  intact: 61/62 tests pass on v0.17.0; this 1 test asserts pre-v0.17.0 internals.
- #50073 config.py          : keep v0.17.0's hygiene_hard_message_limit=400 + add the
  PR's 3 new keys (max_attempts/chunk_oversized_input/never_413). The PR's "5000" was
  unchanged context, not its change. Tests pass (9/9).
- #50296 agent_init.py      : take-theirs (additive session-isolation block)

Per-PR test results after resolution: V017-PER-PR-TEST-RESULTS.txt (5/6 fully green;
#50064 61/62 with 1 documented v0.17.0 forward-compat test removal).
