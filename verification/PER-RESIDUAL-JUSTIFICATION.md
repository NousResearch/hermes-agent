# Per-residual justification table — 34 deferred patches

Council round-14 item 3: for each residual, the blocking rule and the PR it would collide with.

| # | category | patch (file) | +lines | first hunk range | blocking rule / collision |
|---|----------|--------------|--------|------------------|----------------------------|
| 1 | cmx | `agent_conversation_loop.py.patch` | 319 | `-64,6 +64,34` | RULE-5 [id=92873]: CMX belongs in the single CMX PR (#50155) |
| 2 | cmx | `tests_test_context_engine_tool_wrap.py.patch` | 129 | `-0,0 +1,129` | RULE-5 [id=92873]: CMX belongs in the single CMX PR (#50155) |
| 3 | copilot-limits | `agent_model_metadata.py.patch` | 197 | `-149,6 +149,21` | ACCOUNT-SPECIFIC [id=63592]: caps ship-verbatim; generalized form already in #49449 |
| 4 | copilot-limits | `hermes_cli_models.py.patch` | 700 | `-12,6 +12,7` | ACCOUNT-SPECIFIC [id=63592]: caps ship-verbatim; generalized form already in #49449 |
| 5 | post-branch-drift | `agent_system_prompt_prelude.py.patch` | 272 | `-0,0 +1,272` | DRIFT-SUPERSESSION of #48101 |
| 6 | post-branch-drift | `cli.py.patch` | 171 | `-3349,6 +3349,24` | DRIFT-SUPERSESSION of #49917 |
| 7 | post-branch-drift | `gateway_run.py.patch` | 69 | `-1298,11 +1298,30` | DRIFT-SUPERSESSION of #50146 (already shipped there; duplicate) |
| 8 | post-branch-drift | `tests_agent_test_model_metadata.py.patch` | 9 | `-245,7 +245,9` | DRIFT-SUPERSESSION of ? |
| 9 | post-branch-drift | `tests_hermes_cli_test_model_switch_copilot_api_mode.py.patch` | 13 | `-56,7 +56,12` | DRIFT-SUPERSESSION of #50064 |
| 10 | post-branch-drift | `tests_run_agent_test_run_agent.py.patch` | 7 | `-3683,6 +3683,11` | DRIFT-SUPERSESSION of ? |
| 11 | post-branch-drift | `tools_mcp_tool.py.patch` | 80 | `-1125,6 +1125,7` | DRIFT-SUPERSESSION of #48069 |
| 12 | private-feature-mixed | `agent_agent_init.py.patch` | 98 | `-252,7 +252,18` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 13 | private-feature-mixed | `agent_agent_runtime_helpers.py.patch` | 41 | `-1069,6 +1069,23` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 14 | private-feature-mixed | `agent_models_dev.py.patch` | 306 | `-688,6 +688,265` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 15 | private-feature-mixed | `agent_system_prompt.py.patch` | 55 | `-38,6 +38,7` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 16 | private-feature-mixed | `gateway_platforms_api_server.py.patch` | 214 | `-53,6 +53,7` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 17 | private-feature-mixed | `tests_agent_test_copilot_opus_context_fix_2026_06_04.py.patch` | 714 | `-0,0 +1,714` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 18 | private-feature-mixed | `tests_probe_prelude_e2e.py.patch` | 128 | `-0,0 +1,128` | PRIVACY [id=92873]: residual private lines of files already in feature PRs |
| 19 | private-overlay | `agent_anthropic_adapter.py.patch` | 675 | `-77,34 +77,69` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 20 | private-overlay | `agent_auxiliary_client.py.patch` | 229 | `-438,16 +438,20` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 21 | private-overlay | `agent_chat_completion_helpers.py.patch` | 136 | `-122,6 +122,34` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 22 | private-overlay | `agent_gemini_cloudcode_adapter.py.patch` | 13 | `-39,6 +39,10` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 23 | private-overlay | `agent_gemini_native_adapter.py.patch` | 11 | `-28,6 +28,10` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 24 | private-overlay | `hermes_cli_main.py.patch` | 27 | `-2090,6 +2090,14` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 25 | private-overlay | `hermes_state.py.patch` | 128 | `-16,13 +16,68` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 26 | private-overlay | `run_agent.py.patch` | 158 | `-994,6 +994,8` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 27 | private-overlay | `tests_agent_test_anthropic_adapter.py.patch` | 204 | `-106,11 +106,13` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 28 | private-overlay | `tests_agent_test_auxiliary_client.py.patch` | 60 | `-1132,6 +1132,32` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 29 | private-overlay | `tui_gateway_server.py.patch` | 290 | `-300,6 +300,44` | PRIVACY [id=92873]/[id=40686]: v2026.6.5 overlay machinery (agy-cli/auto_router/accelerators), not contributable |
| 30 | private-overlay-phaseh | `hermes_cli_inventory.py.patch` | 6 | `-36,6 +36,8` | PRIVACY: private phase-h build machinery |
| 31 | private-overlay-phaseh | `tests_hermes_cli_test_copilot_catalog_oauth_fallback.py.patch` | 81 | `-16,142 +16,102` | PRIVACY: private phase-h build machinery |
| 32 | private-overlay-phaseh | `tests_hermes_cli_test_copilot_context.py.patch` | 136 | `-7,7 +7,10` | PRIVACY: private phase-h build machinery |
| 33 | private-overlay-phaseh | `tests_hermes_cli_test_inventory.py.patch` | 67 | `-173,6 +173,73` | PRIVACY: private phase-h build machinery |
| 34 | private-overlay-phaseh | `tests_hermes_cli_test_model_validation.py.patch` | 16 | `-334,14 +334,27` | PRIVACY: private phase-h build machinery |
| 35 | private-overlay-phaseh | `tools_skills_tool.py.patch` | 21 | `-976,10 +976,21` | PRIVACY: private phase-h build machinery |

**Total: 35 patches.** Every one is blocked from standalone graduation by a documented rule:
PRIVACY (private-overlay/phaseh/feature-mixed), ACCOUNT-SPECIFIC (copilot-limits), RULE-5 (cmx), or
DRIFT-SUPERSESSION (post-branch-drift — these UPDATE an existing owner PR, they are not new logical changes).
