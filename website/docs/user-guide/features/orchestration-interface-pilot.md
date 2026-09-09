---
title: Orchestration interface pilot
sidebar_label: Interface pilot evidence
---

# Orchestration interface pilot

The first controlled pilot exercised real parent and worker requests through
Hermes, with the same tasks presented through different tool vocabularies.
It found successful two-provider coordination and several incomplete workflows.
**No automatic interface entry was enabled.**

The tested source is [PR #106463](https://github.com/NousResearch/hermes-agent/pull/106463)
at `cbb0204203e6c43900cb4fa8c4f6d8b7cd40a071`, on September 9, 2026.
[CI](https://github.com/NousResearch/hermes-agent/actions/runs/34342527658),
[Docker](https://github.com/NousResearch/hermes-agent/actions/runs/34342526804)
and [Nix](https://github.com/NousResearch/hermes-agent/actions/runs/34342526878)
passed for that source. One independent source review and two targeted correction
reviews covered the interface candidate. These are separate from the live results.

## What was exercised

Each sample used four scenarios in fresh temporary Hermes homes, with no customer
data, external message recipients or worker tools. Two user-created profiles
selected OpenAI-Codex `gpt-5.6-luna` with requested effort `xhigh` and ZAI `glm-5.3`
with requested effort `high`. The parent used only its advertised orchestration
controls. Each scenario allowed at most 16 parent iterations, two workers,
three child iterations and four minutes; each child had a 90-second limit.

1. **Coordinate two workers:** discover profiles, use different providers, send
   guidance during execution, follow up with retained context and acknowledge
   the completed assignments.
2. **Reject an unavailable profile:** return an explicit error without launching
   another profile or silently changing the selected route.
3. **Protect another parent's worker:** deny inspection without changing or
   revealing that worker's conversation.
4. **Guide an idle worker:** queue guidance without waking it, explicitly start
   another assignment, retrieve the retained labels and acknowledge completion.

## Recorded results

Each cell is one attempt. Compare styles within a parent-model group; the parent
effort and model differ between the two groups. This is a usability pilot, not a
statistical ranking.

| Scenario | Luna / Hermes | Luna / Codex-style | Luna / Claude-style | Astra / Hermes | Astra / Codex-style |
|---|---|---|---|---|---|
| Two-provider coordination | ✗ | ✓ | ✓ | ✓ | ✗ |
| Missing-profile denial | ✓ | ✓ | ✓ | ✓ | ✓ |
| Foreign-worker denial | ✓ | ✓ | ✓ | ✓ | ✓ |
| Idle guidance and retained follow-up | ✗ | ✗ | ✗ | ✓ | ✓ |
| Passed scenarios | 2/4 | 3/4 | 3/4 | 4/4 | 3/4 |

The Luna parent was `gpt-5.6-luna` at `low`; the Astra parent was `gpt-6-astra`
at `high`, both through OpenAI-Codex. Claude-style here describes the Hermes tool
interface presented to these models. It is not an Anthropic model or Claude Code
runtime test. Astra with Claude-style and Anthropic-provider qualification were
not sampled.

Luna's canonical two-worker case lacked qualifying running guidance, a label-free
linked follow-up and complete acknowledgments. Its three idle-worker samples
retained both labels but failed the final acknowledgment assertion. Astra's
Codex-style two-worker case delivered running guidance and retained context but
failed the complete-acknowledgment assertion. The reports establish which checks
failed; they do not establish a single cause for every incomplete workflow.

## Read the evidence literally

The [preserved, sanitized reports](/evidence/orchestration-interface-pilot-cbb02042.json)
include source-receipt hashes, per-scenario assertions, selected interface,
observed tool names, guidance state, token counts and elapsed time. Failed samples
are retained alongside passing ones.

- All worker-request scenarios passed the runner's provider/model receipt and
  effective-tool checks. Requested effort settings are named above; these reports
  do not include transmitted effort values and cannot independently establish
  the actual model behind a provider router.
- Recorded structured invalid calls and observed changes to foreign worker state
  were zero. Malformed or unobserved calls and corrective-turn counts remain
  unknown. This does not replace the deterministic permission test suite.
- The duplicate metric compares run identities. It does not prove exactly-once
  execution of external actions; workers had no tools in this pilot.
- Cost aggregation is incomplete. A report's numeric cost must not be presented
  as the complete billed cost of the parent and workers.
- The runner's `qualified: true` means all four scenario assertions passed.
  Registry admission also requires the complete deterministic and evidence
  contract. It is not an automatic registry mutation.

The default automatic path therefore remains canonical Hermes. Explicit
Codex-style or Claude-style selections remain experimental. These results prove
the named live behaviors for the recorded source and test profiles; they do not
prove installation, release, customer readiness, vendor-runtime parity or that
one model family should be compulsory.
