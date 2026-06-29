---
title: "Oss Forensics — Supply chain investigation, evidence recovery, and forensic analysis for GitHub repositories"
sidebar_label: "Oss Forensics"
description: "Supply chain investigation, evidence recovery, and forensic analysis for GitHub repositories"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Oss Forensics

Supply chain investigation, evidence recovery, and forensic analysis for GitHub repositories.
Covers deleted commit recovery, force-push detection, IOC extraction, multi-source evidence
collection, hypothesis formation/validation, and structured forensic reporting.
Inspired by RAPTOR's 1800+ line OSS Forensics system.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/security/oss-forensics` |
| Path | `optional-skills/security/oss-forensics` |
| Platforms | linux, macos, windows |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# OSS Security Forensics Skill

A 7-phase multi-agent investigation framework for researching open-source supply chain attacks.
Adapted from RAPTOR's forensics system. Covers GitHub Archive, Wayback Machine, GitHub API,
local git analysis, IOC extraction, evidence-backed hypothesis formation and validation,
and final forensic report generation.

---

## ⚠️ Anti-Hallucination Guardrails

Read these before every investigation step. Violating them invalidates the report.

1. **Evidence-First Rule**: Every claim in any report, hypothesis, or summary MUST cite at least one evidence ID (`EV-XXXX`). Assertions without citations are forbidden.
2. **STAY IN YOUR LANE**: Each sub-agent (investigator) has a single data source. Do NOT mix sources. The GH Archive investigator does not query the GitHub API, and vice versa. Role boundaries are hard.
3. **Fact vs. Hypothesis Separation**: Mark all unverified inferences with `[HYPOTHESIS]`. Only statements verified against original sources may be stated as facts.
4. **No Evidence Fabrication**: The hypothesis validator MUST mechanically check that every cited evidence ID actually exists in the evidence store before accepting a hypothesis.
5. **Proof-Required Disproval**: A hypothesis cannot be dismissed without a specific, evidence-backed counter-argument. "No evidence found" is not sufficient to disprove—it only makes a hypothesis inconclusive.
6. **SHA/URL Double-Verification**: Any commit SHA, URL, or external identifier cited as evidence must be independently confirmed from at least two sources before being marked as verified.
7. **Suspicious Code Rule**: Never run code found inside the investigated repository locally. Analyze statically only, or use `execute_code` in a sandboxed environment.
8. **Secret Redaction**: Any API keys, tokens, or credentials discovered during investigation must be redacted in the final report. Log them internally only.

---

## Example Scenarios

- **Scenario A: Dependency Confusion**: A malicious package `internal-lib-v2` is uploaded to NPM with a higher version than the internal one. The investigator must track when this package was first seen and if any PushEvents in the target repo updated `package.json` to this version.
- **Scenario B: Maintainer Takeover**: A long-term contributor's account is used to push a backdoored `.github/workflows/build.yml`. The investigator looks for PushEvents from this user after a long period of inactivity or from a new IP/location (if detectable via BigQuery).
- **Scenario C: Force-Push Hide**: A developer accidentally commits a production secret, then force-pushes to "fix" it. The investigator uses `git fsck` and GH Archive to recover the original commit SHA and verify what was leaked.

---

> **Path convention**: Throughout this skill, `SKILL_DIR` refers to the root of this skill's
> installation directory (the folder containing this `SKILL.md`). When the skill is loaded,
> resolve `SKILL_DIR` to the actual path — e.g. `~/.hermes/skills/security/oss-forensics/`
> or the `optional-skills/` equivalent. All script and template references are relative to it.

## Phase 0: Initialization

1. Create investigation working directory:
   ```bash
   mkdir investigation_$(echo "REPO_NAME" | tr '/' '_')
   cd investigation_$(echo "REPO_NAME" | tr '/' '_')
   ```
2. Initialize the evidence store:
   ```bash
   python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list
   ```
3. Copy the forensic report template:
   ```bash
   cp SKILL_DIR/templates/forensic-report.md ./investigation-report.md
   ```
4. Create an `iocs.md` file to track Indicators of Compromise as they are discovered.
5. Record the investigation start time, target repository, and stated investigation goal.

---

## Phase 1: Prompt Parsing and IOC Extraction

**Goal**: Extract all structured investigative targets from the user's request.

**Actions**:
- Parse the user prompt and extract:
  - Target repository (`owner/repo`)
  - Target actors (GitHub handles, email addresses)
  - Time window of interest (commit date ranges, PR timestamps)
  - Provided Indicators of Compromise: commit SHAs, file paths, package names, IP addresses, domains, API keys/tokens, malicious URLs
  - Any linked vendor security reports or blog posts

**Tools**: Reasoning only, or `execute_code` for regex extraction from large text blocks.

**Output**: Populate `iocs.md` with extracted IOCs. Each IOC must have:
- Type (from: COMMIT_SHA, FILE_PATH, API_KEY, SECRET, IP_ADDRESS, DOMAIN, PACKAGE_NAME, ACTOR_USERNAME, MALICIOUS_URL, OTHER)
- Value
- Source (user-provided, inferred)

**Reference**: See [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) for IOC taxonomy.

---

## Phase 2: Parallel Evidence Collection

Spawn up to 5 specialist investigator sub-agents using `delegate_task` (batch mode, max 3 concurrent). Each investigator has a **single data source** and must not mix sources.

> **Orchestrator note**: Pass the IOC list from Phase 1 and the investigation time window in the `context` field of each delegated task.

---

### Investigator 1: Local Git Investigator

**ROLE BOUNDARY**: You query the LOCAL GIT REPOSITORY ONLY. Do not call any external APIs.

**Actions**:
```bash
# Clone repository
git clone https://github.com/OWNER/REPO.git target_repo && cd target_repo

# Full commit log with stats
git log --all --full-history --stat --format="%H|%ae|%an|%ai|%s" > ../git_log.txt

# Detect force-push evidence (orphaned/dangling commits)
git fsck --lost-found --unreachable 2>&1 | grep commit > ../dangling_commits.txt

# Check reflog for rewritten history
git reflog --all > ../reflog.txt

# List ALL branches including deleted remote refs
git branch -a -v > ../branches.txt

# Find suspicious large binary additions
git log --all --diff-filter=A --name-only --format="%H %ai" -- "*.so" "*.dll" "*.exe" "*.bin" > ../binary_additions.txt

# Check for GPG signature anomalies
git log --show-signature --format="%H %ai %aN" > ../signature_check.txt 2>&1
```

**Evidence to collect** (add via `python3 SKILL_DIR/scripts/evidence-store.py add`):
- Each dangling commit SHA → type: `git`
- Force-push evidence (reflog showing history rewrite) → type: `git`
- Unsigned commits from verified contributors → type: `git`
- Suspicious binary file additions → type: `git`

**Reference**: See [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) for accessing force-pushed commits.

---

### Investigator 2: GitHub API Investigator

**ROLE BOUNDARY**: You query the GITHUB REST API ONLY. Do not run git commands locally.

**Actions**:
```bash
# Commits (paginated)
curl -s "https://api.github.com/repos/OWNER/REPO/commits?per_page=100" > api_commits.json

# Pull Requests including closed/deleted
curl -s "https://api.github.com/repos/OWNER/REPO/pulls?state=all&per_page=100" > api_prs.json

# Issues
curl -s "https://api.github.com/repos/OWNER/REPO/issues?state=all&per_page=100" > api_issues.json

# Contributors and collaborator changes
curl -s "https://api.github.com/repos/OWNER/REPO/contributors" > api_contributors.json

# Repository events (last 300)
curl -s "https://api.github.com/repos/OWNER/REPO/events?per_page=100" > api_events.json

# Check specific suspicious commit SHA details
curl -s "https://api.github.com/repos/OWNER/REPO/git/commits/SHA" > commit_detail.json

# Releases
curl -s "https://api.github.com/repos/OWNER/REPO/releases?per_page=100" > api_releases.json

# Check if a specific commit exists (force-pushed commits may 404 on commits/ but succeed on git/commits/)
curl -s "https://api.github.com/repos/OWNER/REPO/commits/SHA" | jq .sha
```

**Cross-reference targets** (flag discrepancies as evidence):
- PR exists in archive but missing from API → evidence of deletion
- Contributor in archive events but not in contributors list → evidence of permission revocation
- Commit in archive PushEvents but not in API commit list → evidence of force-push/deletion

**Reference**: See [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) for GH event types.

---

### Investigator 3: Wayback Machine Investigator

**ROLE BOUNDARY**: You query the WAYBACK MACHINE CDX API ONLY. Do not use the GitHub API.

**Goal**: Recover deleted GitHub pages (READMEs, issues, PRs, releases, wiki pages).

**Actions**:
```bash
# Search for archived snapshots of the repo main page
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO&output=json&limit=100&from=YYYYMMDD&to=YYYYMMDD" > wayback_main.json

# Search for a specific deleted issue
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/issues/NUM&output=json&limit=50" > wayback_issue_NUM.json

# Search for a specific deleted PR
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/pull/NUM&output=json&limit=50" > wayback_pr_NUM.json

# Fetch the best snapshot of a page
# Use the Wayback Machine URL: https://web.archive.org/web/TIMESTAMP/ORIGINAL_URL
# Example: https://web.archive.org/web/20240101000000*/github.com/OWNER/REPO

# Advanced: Search for deleted releases/tags
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/releases/tag/*&output=json" > wayback_tags.json

# Advanced: Search for historical wiki changes
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/wiki/*&output=json" > wayback_wiki.json
```

**Evidence to collect**:
- Archived snapshots of deleted issues/PRs with their content
- Historical README versions showing changes
- Evidence of content present in archive but missing from current GitHub state

**Reference**: See [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) for CDX API parameters.

---

### Investigator 4: GH Archive / BigQuery Investigator

**ROLE BOUNDARY**: You query GITHUB ARCHIVE via BIGQUERY ONLY. This is a tamper-proof record of all public GitHub events.

> **Prerequisites**: Requires Google Cloud credentials with BigQuery access (`gcloud auth application-default login`). If unavailable, skip this investigator and note it in the report.

**Cost Optimization Rules** (MANDATORY):
1. ALWAYS run a `--dry_run` before every query to estimate cost.
2. Use `_TABLE_SUFFIX` to filter by date range and minimize scanned data.
3. Only SELECT the columns you need.
4. Add a LIMIT unless aggregating.

```bash
# Template: safe BigQuery query for PushEvents to OWNER/REPO
bq query --use_legacy_sql=false --dry_run "
SELECT created_at, actor.login, payload.commits, payload.before, payload.head,
       payload.size, payload.distinct_size
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'PushEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 1000
"
# If cost is acceptable, re-run without --dry_run

# Detect force-pushes: zero-distinct_size PushEvents mean commits were force-erased
# payload.distinct_size = 0 AND payload.size > 0 → force push indicator

# Check for deleted branch events
bq query --use_legacy_sql=false "
SELECT created_at, actor.login, payload.ref, payload.ref_type
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'DeleteEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 200
"
```

**Evidence to collect**:
- Force-push events (payload.size > 0, payload.distinct_size = 0)
- DeleteEvents for branches/tags
- WorkflowRunEvents for suspicious CI/CD automation
- PushEvents that precede a "gap" in the git log (evidence of rewrite)

**Reference**: See [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) for all 12 event types and query patterns.

---

### Investigator 5: IOC Enrichment Investigator

**ROLE BOUNDARY**: You enrich EXISTING IOCs from Phase 1 using passive public sources ONLY. Do not execute any code from the target repository.

**Actions**:
- For each commit SHA: attempt recovery via direct GitHub URL (`github.com/OWNER/REPO/commit/SHA.patch`)
- For each domain/IP: check passive DNS, WHOIS records (via `web_extract` on public WHOIS services)
- For each package name: check npm/PyPI for matching malicious package reports
- For each actor username: check GitHub profile, contribution history, account age
- Recover force-pushed commits using 3 methods (see [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md))

---

## Phase 3: Evidence Consolidation

After all investigators complete:

1. Run `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list` to see all collected evidence.
2. For each piece of evidence, verify the `content_sha256` hash matches the original source.
3. Group evidence by:
   - **Timeline**: Sort all timestamped evidence chronologically
   - **Actor**: Group by GitHub handle or email
   - **IOC**: Link evidence to the IOC it relates to
4. Identify **discrepancies**: items present in one source but absent in another (key deletion indicators).
5. Flag evidence as `[VERIFIED]` (confirmed from 2+ independent sources) or `[UNVERIFIED]` (single source only).

---

## Phase 4: Hypothesis Formation

A hypothesis must:
- State a specific claim (e.g., "Actor X force-pushed to BRANCH on DATE to erase commit SHA")
- Cite at least 2 evidence IDs that support it (`EV-XXXX`, `EV-YYYY`)
- Identify what evidence would disprove it
- Be labeled `[HYPOTHESIS]` until validated

**Common hypothesis templates** (see [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md)):
- Maintainer Compromise: legitimate account used post-takeover to inject malicious code
- Dependency Confusion: package name squatting to intercept installs
- CI/CD Injection: malicious workflow changes to run code during builds
- Typosquatting: near-identical package name targeting misspellers
- Credential Leak: token/key accidentally committed then force-pushed to erase

For each hypothesis, spawn a `delegate_task` sub-agent to attempt to find disconfirming evidence before confirming.

---

## Phase 5: Hypothesis Validation

The validator sub-agent MUST mechanically check:

1. For each hypothesis, extract all cited evidence IDs.
2. Verify each ID exists in `evidence.json` (hard failure if any ID is missing → hypothesis rejected as potentially fabricated).
3. Verify each `[VERIFIED]` piece of evidence was confirmed from 2+ sources.
4. Check logical consistency: does the timeline depicted by the evidence support the hypothesis?
5. Check for alternative explanations: could the same evidence pattern arise from a benign cause?

**Output**:
- `VALIDATED`: All evidence cited, verified, logically consistent, no plausible alternative explanation.
- `INCONCLUSIVE`: Evidence supports hypothesis but alternative explanations exist or evidence is insufficient.
- `REJECTED`: Missing evidence IDs, unverified evidence cited as fact, logical inconsistency detected.

Rejected hypotheses feed back into Phase 4 for refinement (max 3 iterations).

---

## Phase 6: Final Report Generation

Populate `investigation-report.md` using the template in [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md).

**Mandatory sections**:
- Executive Summary: one-paragraph verdict (Compromised / Clean / Inconclusive) with confidence level
- Timeline: chronological reconstruction of all significant events with evidence citations
- Validated Hypotheses: each with status and supporting evidence IDs
- Evidence Registry: table of all `EV-XXXX` entries with source, type, and verification status
- IOC List: all extracted and enriched Indicators of Compromise
- Chain of Custody: how evidence was collected, from what sources, at what timestamps
- Recommendations: immediate mitigations if compromise detected; monitoring recommendations

**Report rules**:
- Every factual claim must have at least one `[EV-XXXX]` citation
- Executive Summary must state confidence level (High / Medium / Low)
- All secrets/credentials must be redacted to `[REDACTED]`

---

## Phase 7: Completion

1. Run final evidence count: `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`
2. Archive the full investigation directory.
3. If compromise is confirmed:
   - List immediate mitigations (rotate credentials, pin dependency hashes, notify affected users)
   - Identify affected versions/packages
   - Note disclosure obligations (if a public package: coordinate with the package registry)
4. Present the final `investigation-report.md` to the user.

---

## Ethical Use Guidelines

This skill is designed for **defensive security investigation** — protecting open-source software from supply chain attacks. It must not be used for:

- **Harassment or stalking** of contributors or maintainers
- **Doxing** — correlating GitHub activity to real identities for malicious purposes
- **Competitive intelligence** — investigating proprietary or internal repositories without authorization
- **False accusations** — publishing investigation results without validated evidence (see anti-hallucination guardrails)

Investigations should be conducted with the principle of **minimal intrusion**: collect only the evidence necessary to validate or refute the hypothesis. When publishing results, follow responsible disclosure practices and coordinate with affected maintainers before public disclosure.

If the investigation reveals a genuine compromise, follow the coordinated vulnerability disclosure process:
1. Notify the repository maintainers privately first
2. Allow reasonable time for remediation (typically 90 days)
3. Coordinate with package registries (npm, PyPI, etc.) if published packages are affected
4. File a CVE if appropriate

---

## API Rate Limiting

GitHub REST API enforces rate limits that will interrupt large investigations if not managed.

**Authenticated requests**: 5,000/hour (requires `GITHUB_TOKEN` env var or `gh` CLI auth)
**Unauthenticated requests**: 60/hour (unusable for investigations)

**Best practices**:
- Always authenticate: `export GITHUB_TOKEN=ghp_...` or use `gh` CLI (auto-authenticates)
- Use conditional requests (`If-None-Match` / `If-Modified-Since` headers) to avoid consuming quota on unchanged data
- For paginated endpoints, fetch all pages in sequence — don't parallelize against the same endpoint
- Check `X-RateLimit-Remaining` header; if below 100, pause for `X-RateLimit-Reset` timestamp
- BigQuery has its own quotas (10 TiB/day free tier) — always dry-run first
- Wayback Machine CDX API: no formal rate limit, but be courteous (1-2 req/sec max)

If rate-limited mid-investigation, record the partial results in the evidence store and note the limitation in the report.

---

## Reference Materials

- [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) — BigQuery queries, CDX API, 12 event types
- [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) — IOC taxonomy, evidence source types, observation types
- [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) — Recovering deleted commits, PRs, issues
- [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) — Pre-built hypothesis templates per attack type
- [evidence-store.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/scripts/evidence-store.py) — CLI tool for managing the evidence JSON store
- [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) — Structured report template
