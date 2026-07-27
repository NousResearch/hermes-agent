---
title: "Unbroker — Autonomously remove your info from data-broker sites"
sidebar_label: "Unbroker"
description: "Autonomously remove your info from data-broker sites"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Unbroker

Autonomously remove your info from data-broker sites.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/security/unbroker` |
| Path | `optional-skills/security/unbroker` |
| Version | `1.0.0` |
| Author | SHL0MS (github.com/SHL0MS) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `privacy`, `data-broker`, `opt-out`, `ccpa`, `gdpr`, `security`, `doxxing` |
| Related skills | [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`agentmail`](/docs/user-guide/skills/optional/email/email-agentmail), [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya), [`scrapling`](/docs/user-guide/skills/optional/research/research-scrapling), [`osint-investigation`](/docs/user-guide/skills/optional/research/research-osint-investigation) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# unbroker

Find where a person's personal information (name, addresses, phone, email, relatives) is exposed on
data brokers and people-search sites, then remove it - automatically where possible, with guided
human steps only where a site demands a CAPTCHA, government ID, phone call, or fax. Manages multiple
people independently. It does **not** defeat anti-bot systems, does **not** act on anyone without
recorded consent, and does **not** remove public records (voter/property/court) or accounts the
person controls.

The Python CLI (`scripts/pdd.py`) owns the deterministic state - config, dossiers + consent, the
broker database, tier planning, the ledger, drafts, reports, **email sending (SMTP), verification-link
polling (IMAP), and the autonomous action queue (`next`)**. You (the agent) do the scanning and
form-driving with native tools: `web_extract` and `browser_navigate` for searching and web forms, and
`cronjob` for recurring re-scans.

## Autonomy contract

This skill is designed to run **hands-off**. After intake (+ recorded consent) there are exactly TWO
legitimate human touchpoints: (1) the intake conversation itself, and (2) ONE consolidated human-task
digest at the end of the run (`$PDD tasks`). Between those:

- **Never ask the operator to choose configuration.** `$PDD setup --auto` detects capabilities and
  picks the most autonomous valid config itself.
- **Never pause before individual submissions** when `autonomy=full` (the default): the consent
  recorded at intake is standing authorization for T0-T2 opt-outs. (`autonomy=assisted` restores
  per-submission confirmation for cautious operators - honor `confirm_first` flags in `next` output.)
- **Never interrupt the run for human-only work.** Record it (`record ... human_task_queued
  --reason "..."`) and keep going; it all surfaces once in the final digest.
- **Drive the whole run as a loop over `$PDD next <subject>`** - it returns the exact ordered actions
  to take right now (scan, poll verification, re-check, opt out parents-first, requeue blocked), plus
  the human digest. Execute every action, record outcomes, re-run `next`, repeat until
  `done_for_now`. Then present the digest, report, and schedule the cron.

The hard limits that autonomy never overrides: no acting without recorded consent, no disclosure
beyond `disclosure_fields`, no CAPTCHA/anti-bot bypass, and `confirmed_removed` only after a
verifying re-scan.

## When to Use

- "Remove my (or my family member's) data from data brokers / people-search sites."
- "Opt me out", "delete me from Spokeo/Whitepages/etc.", "clean up after a doxxing."
- "Set up recurring privacy monitoring" (brokers re-list people).
- Checking which brokers still expose someone and why.

## Prerequisites

- `python3` (stdlib only; no extra packages needed for the core engine).
- **Optional upgrades** (the skill works zero-config without these; `setup --auto` turns on every
  one it detects, reading credentials from the shell env **and from `$HERMES_HOME/.env`** so keys
  Hermes already loads for its own tools are picked up without re-exporting - each one converts a
  class of human tasks into agent actions):
  - **Cloud browser (recommended default): `BROWSERBASE_API_KEY`.** `setup --auto` selects it
    whenever the key is present, and it is the intended baseline: a real residential-IP cloud
    browser **clears soft/managed CAPTCHAs (Cloudflare Turnstile, hCaptcha/reCAPTCHA checkbox) as
    normal operation**, so those brokers stay automated (T1) instead of becoming human tasks. This
    is not CAPTCHA "solving" - no solver service, no fingerprint spoofing; only interactive/behavioral
    ("hard") challenges the browser genuinely cannot pass fall back to a human task. Without the key,
    the plain agent browser is used and soft-CAPTCHA brokers drop to T2 (human).
  - Email automation, two credential-free-or-not options:
    - **Browser mode (no password): `setup --email-mode browser`.** The agent sends opt-out/CCPA
      emails and opens verification links through the operator's **logged-in webmail** using
      `browser_*` tools. Nothing is stored. This requires Hermes to be pointed at the operator's own
      logged-in browser, **NOT** a cloud browser: a headless cloud browser (Browserbase) holds no
      webmail session and is itself Cloudflare/DataDome-gated on webmail and on session-bound broker
      gates (e.g. PeopleConnect guided-mode). Drive the operator's real Chrome over CDP - launch
      `chrome --remote-debugging-port=9222 --user-data-dir="$HOME/.hermes/chrome-debug"` (a dedicated
      debug profile signed into the webmail once, not the Default profile) and connect the browser
      tools to `127.0.0.1:9222`. **`$PDD cdp` launches this for you** (finds Chrome/Chromium/Brave/Edge,
      starts it detached on the dedicated profile, prints the CDP endpoint; `--check` to test, `--print`
      for the command). See `references/methods.md` -> "Browser backends: scan vs execute".
      Falls back to drafts for an email if the inbox isn't reachable.
    - **SMTP/IMAP (stored creds): `EMAIL_ADDRESS` + `EMAIL_PASSWORD`** (+ `EMAIL_SMTP_HOST` /
      `EMAIL_IMAP_HOST` for non-mainstream providers; gmail/outlook/yahoo/icloud/fastmail inferred).
      The CLI sends via `send-email` and reads verify links via `poll-verification`. The `agentmail`
      skill (per-broker aliases) also counts.
  - Google Sheets tracker: the `google-workspace` skill.
  - The `scrapling` skill for stealth/Cloudflare-protected pages.

## How to Run

Run everything through the `terminal` tool. From this skill's directory:

```bash
PDD="python3 scripts/pdd.py"
```

The engine stores data under `$PDD_DATA_DIR` (default `$HERMES_HOME/unbroker`), written
`0600`. Run via `terminal`, **not** `execute_code` (that sandbox scrubs env and redacts output, which
breaks reading the dossier).

## Quick Reference

| Command | Purpose |
|---|---|
| `$PDD setup --auto` | **Autonomous setup**: detect capabilities, pick the most autonomous valid config (no questions) |
| `$PDD doctor` | Readiness check: config, broker count, and which upgrades are on/available |
| `$PDD cdp [--check] [--print] [--port N]` | Launch/detect the operator's Chrome over CDP for Phase-2 browser + webmail (dedicated debug profile; the reliable way to send webmail and clear session-bound gates) |
| `$PDD intake --full-name "..." [--alias ...] [--email ... --phone ...] [--city --state] [--prior-location "City,ST"] --consent` | Create a consenting subject; captures aliases + multiple emails/phones + prior locations; prints `subject_id` |
| `$PDD next <subject>` | **The autonomous loop driver**: ordered agent actions right now + human digest + `next_wake_at` |
| `$PDD brokers [--priority crucial]` | List the people-search broker database (curated + live) |
| `$PDD refresh-brokers` | Pull the latest BADBOOL people-search list **and the CA Data Broker Registry** (`next` requeues this automatically when the cache is stale) |
| `$PDD registry [--search NAME]` | State registry coverage (CA ~545 ingested; VT/OR/TX portals surfaced); the DROP/email lane, not scanned |
| `$PDD drop <subject> [--filed]` | **The one-shot legal lever**: one CA DROP request deletes from ALL registered brokers; `--filed` records it |
| `$PDD plan <subject> [--priority crucial]` | Per-broker tier + method + `search_vectors` + the exact fields to disclose |
| `$PDD plan <subject> --batch` | **Reduce view**: overlays ledger state, groups brokers by next action (unscanned/found/indirect/blocked/in_progress/done), collapses ownership clusters, **orders `found` cluster-parents-first + emits a tailored `parent_playbook`**, prints `next_actions` |
| `$PDD fanout <subject> [--priority crucial] [--size 5]` | Batch brokers into parallel `delegate_task` subagents (auto for large runs; batches of 5 - 8+ time out) |
| `$PDD record <subject> <broker> <state> [--found true] [--evidence JSON] [--disclosed F --channel C] [--reason "..."]` | Update the ledger (validated state machine); **auto-stamps `next_recheck_at`** |
| `$PDD show <subject> <broker>` | Read back a case's recorded state + evidence + disclosure log (so the parent re-verifies a subagent's `found` without re-deriving the listing URL) |
| `$PDD send-email <subject> <broker> --listing <url> [--kind ccpa_indirect ...]` | Render + record the request (recipient locked to the broker's own address). **browser** mode returns a `compose` payload to send via webmail (no password); **programmatic** mode SMTP-sends |
| `$PDD verify-link <subject> <broker> --text '<body>'` | **browser mode**: extract a broker's verification link from webmail text you read (anti-phishing scored) |
| `$PDD poll-verification <subject> [--broker <id>]` | **programmatic mode**: poll IMAP for verification links (anti-phishing scored); auto-advances `submitted → verification_pending` |
| `$PDD render-email <subject> <broker> --listing <url>` | Draft only (fallback when no email mode is configured) |
| `$PDD due <subject>` | Cases whose recheck window arrived (the cron re-scan queue) |
| `$PDD tasks <subject>` | ONE consolidated human-task digest (present at END of run) |
| `$PDD status <subject>` | Markdown status report |
| `$PDD report <subject> --sheets` | Rows for the Google Sheets tracker |

## Batch operation (two-phase: crawl-all, then delete)

For anything past a couple of brokers, run this as **map → reduce → act**, not broker-by-broker:

- **Phase 1 - DISCOVER (read-only, parallel, idempotent).** Crawl *every* broker first and record a
  verdict for each (`found` / `not_found` / `indirect_exposure` / `blocked`). Scanning has no side
  effects, so it is safe to parallelize and retry. Getting the full exposure map *before* acting is
  what unlocks cluster dedup and prioritization below. **Default: the parent drives `web_extract`
  probes directly** - most people-search sites render name/phone/address results as static HTML that
  `web_extract` reads in seconds. Escalate to `browser_*` only for the few JS-only sites, and to
  `delegate_task` subagents only for genuinely *reasoning*-heavy work (large-scale namesake/relative
  disambiguation). **Do NOT hand a browser-toolset subagent a big list of brokers to crawl** - in the
  field this timed out repeatedly (600s, ~5-6 brokers each, no summary) because browser navigation is
  heavy; the ledger writes that survived came at 10x the cost of parent `web_extract`. A `blocked`
  (DataDome/Cloudflare/`antibot`) site is *not* a subagent job either: record `blocked` and requeue it
  for a stealth/cloud browser (Browserbase) pass. Subagent reports are self-reports - the parent
  re-fetches key URLs to confirm a `found` before trusting it (this cuts both ways: it caught a real
  listing the parent had wrongly assumed was a false positive).
- **REDUCE - `$PDD plan <subject> --batch`.** Collapses the crawl into a phase-oriented plan: groups by
  next action, **collapses ownership clusters** (a parent removal that clears children is ONE action,
  not N - e.g. one Intelius/PeopleConnect suppression covers Truthfinder/Instant Checkmate/US Search/…),
  and prints `next_actions`. `phase` is `discover` while anything is unscanned, else `delete`.
- **Phase 2 - DELETE (sequential, irreversible).** Work the reduced groups **parents first**:
  `plan --batch` orders the `found` group cluster-parents-first (most children first) and emits a
  `parent_playbook` with tailored, ordered steps per parent - follow that order and those steps
  (full recipes in `references/methods.md` → "Ownership clusters - DO PARENTS FIRST"). Do the
  cluster parents (skipping the covered children), **re-scan each parent's children after it confirms**
  (they usually drop out), then the standalone listings; send the `indirect_exposure` cases as
  CCPA/GDPR delete-my-PII emails (`send-email --kind ccpa_indirect`), and defer `blocked` to the
  stealth-browser pass. Opt-outs hit CAPTCHAs, email-verification loops, and session binding - work
  them **one at a time, carefully** (this is the opposite of fan-out), but do NOT stop to ask
  permission per submission in `autonomy=full`; in `assisted`, confirm each one. **Usually prefer
  deletion over suppression** where a broker offers both (Spokeo/BeenVerified) - but follow the
  record's `deletion.prefer`: **PeopleConnect is the exception** (`prefer: false`), where deleting
  your user data removes your suppressions and does not stop public-records re-listing, so you
  suppress-and-maintain instead.
- **Blind opt-out is the DEFAULT, not a fallback.** Submit an opt-out/deletion on **every site with an
  accessible removal channel, even when a listing was not first confirmed** - it discloses only the
  subject's own identifiers to the broker's own official channel, so it does not violate
  least-disclosure. Two corollaries: (1) a guided flow that matches email+DOB+name and says "no results"
  is a **stronger `not_found`** than any scrape - the opt-out flow doubles as the search; (2) when a form
  is automation-hostile (hard CAPTCHA, Cloudflare/DataDome, slide-to-verify slider), **default to the
  broker's cited rights-request email** (name+state+contact-email only) rather than recording `blocked`.
  CAPTCHA policy: never defeat behavioral/token/slider challenges; OK to read a static distorted-text or
  plain-arithmetic CAPTCHA on the subject's own opt-out, but stop if the site rejects the whole
  submission after a correct answer (it is fingerprinting the automation). Third-party/indirect records
  are the exception - still confirm those before acting. Per-site game plans + the meta-search no-op
  skip-list are in `references/site-playbooks.md`; the full policy is in `references/methods.md`.
- **PeopleConnect delete-wipes-suppression (permanent rule).** A PeopleConnect *deletion* wipes the
  suppression and the subject re-lists across the whole affiliate cluster. If a "Your deletion request
  for PeopleConnect.us is Complete" email ever appears, the suppression is gone -> **re-run suppression
  and re-verify** the Control step reads "suppressed". Never leave this cluster on a completed deletion
  (see `references/brokers/intelius.json`).

Subagent reports are self-reports: the parent re-verifies key claims (listing URLs, match basis) before
recording `found` and before any deletion.

## Procedure (the autonomous loop)

1. **Setup (once, no questions).** Run `$PDD setup --auto` - it detects capabilities and configures
   the most autonomous valid combination itself (programmatic email when `EMAIL_*` creds exist,
   Browserbase when its key exists, `age` encryption when the binary exists, `autonomy=full`). Then
   `$PDD doctor` and show the operator the readiness output **for information, not as a question** -
   proceed immediately. Mention what would unlock more automation (e.g. email creds) but do not wait.
2. **Intake + consent (the ONE human conversation).** `$PDD intake ...` with `--consent` (and
   `--consent-method`). Without consent the engine refuses to plan or act. Collect everything in one
   pass - names/aliases, current + prior cities, emails, phones - so you never have to come back with
   questions. For California subjects, also read `references/legal/drop.md`: `next` will surface a
   `drop_submit` one-shot that deletes from every registered broker (~545) at once, which is the
   single highest-leverage action. File it, then `drop <subject> --filed`. For non-CA subjects the
   registry is covered by targeted CCPA/GDPR emails (`registry --search`, then `send-email`); the
   people-search sites are worked directly in either case.
3. **Drain the queue.** Loop:

   ```
   while true:
     q = $PDD next <subject>
     if q.actions is empty: break
     execute EVERY action in order; record each outcome via $PDD record
   ```

   `next` emits, in order: `refresh_brokers` (stale cache), `fanout_scan`/`scan_inline` (Phase 1
   crawl - see step 4), `poll_verification` (in-flight email confirmations), `verify_removal` (due
   re-checks), `optout_web_form`/`optout_email_send` (Phase 2, parents-first with playbook steps),
   `indirect_email_send`, and `stealth_rescan`. Human-only work never appears as an action - it
   accumulates in `q.human_digest`. In `autonomy=full`, execute actions without pausing; honor
   `confirm_first` in `assisted` mode.
4. **Scanning (when `next` says so).** For `fanout_scan`: run `$PDD fanout <subject>` and **spawn one
   `delegate_task` subagent per `batch`, in parallel, passing that batch's ready-made `brief`** - do
   not scan all brokers yourself sequentially. For `scan_inline`: scan the few brokers yourself.
   Either way, each broker gets **every** `search_vectors` entry via the `references/methods.md`
   ladder (`web_extract` → `site:` probe → `browser_navigate` → `scrapling`), a 404 is INCONCLUSIVE
   (not `not_found`), `blocked` is recorded when `antibot` is set and no stealth browser is available,
   and subject vs namesake/relative is confirmed before recording:
   `$PDD record <subject> <broker> <found|not_found|indirect_exposure|blocked> --found <bool> --evidence '{"listing_urls":[...]}'`.
   The parent re-verifies key `found` claims from subagents before trusting them.
5. **Opt-outs (when `next` says so).** Actions come pre-ordered parents-first with `steps` from each
   broker record's own `optout.playbook` (field-verified; cluster parents like PeopleConnect,
   Whitepages, BeenVerified, Spokeo have exact, live-checked recipes). **Deletion usually beats
   suppression**: when an action carries `prefer_deletion`, complete the record's DELETION lane, not
   just the hide-my-listing flow. When it carries `prefer_suppression` instead (**PeopleConnect** -
   deleting removes your suppressions and does not stop re-listing), do the suppression flow and keep
   it maintained; use their Delete button only for a deliberate data-purge. Per method:
   - **web_form** → drive `optout_url` with `browser_navigate`/`browser_type`/`browser_click`, submit
     only `disclosure_fields`, screenshot the confirmation, then the action's `after` record command.
     Playbooks may end with a right-to-delete `send-email` follow-up - do it (full erasure, not just
     listing suppression).
    - **email** → `$PDD send-email <subject> <broker> --kind <ccpa|gdpr|generic> --to <addr>
      --listing <url>` records + discloses in one step (recipient locked to addresses the broker
      record declares; `next` picks the kind from residency - never claim CCPA/GDPR for someone who
      can't). In **browser** mode it returns a recipient-locked `compose` payload: compose a new
      message to `compose.to` with `compose.subject`/`compose.body` exactly in the operator's webmail
      via `browser_*` and send (no password); in **programmatic** mode it SMTP-sends. `next` also
      routes human-gated forms (phone-callback/gov-ID) through a broker's deletion email when one
      exists - the **rescue lane** (verified Whitepages pattern). Draft-only falls back to
      `render-email` + a digest entry.
   - **captcha** → soft/managed challenges clear automatically on the default cloud browser (proceed
     as normal); only a hard interactive/behavioral challenge it can't pass is recorded `blocked`
     (requeued for the stealth/operator-browser pass). Never a solver service.
   - **phone_callback / account / gov_id / fax / mail / voice (T3)** *without a deletion email* →
     never an agent action; `next` already routed these to the digest. Record them:
     `$PDD record <subject> <broker> human_task_queued --reason "..."`.
 6. **Verification (when `next` says so).** In **programmatic** mode `$PDD poll-verification <subject>`
    finds arrived confirmation links via IMAP (anti-phishing scored, auto-advances state). In
    **browser** mode, open the broker's confirmation email in the operator's webmail and run
    `$PDD verify-link <subject> <broker> --text '<body>'` to score the link. Either way **open the
    link in the same browser** (several brokers bind the verification session to the browser that
    opens it), finish the flow, then record `awaiting_processing`. `confirmed_removed` ONLY after a
    verifying re-scan shows the listing gone - never off the submission flow's own confirmation page.
7. **Wrap up (once per run).** When `next` returns no actions: present `$PDD tasks <subject>` (the
   consolidated human digest) if non-empty, then `$PDD status <subject>`; if the Sheets tracker is
   on, append `$PDD report <subject> --sheets` rows via the `google-workspace` skill.
8. **Schedule the next wake-up.** `next` returns `next_wake_at` (earliest due re-check). Create ONE
   `cronjob` that re-runs this skill's loop for the subject (a prompt like: *"run the
   unbroker loop for &lt;subject_id>: `$PDD next` and execute all actions"*). Processing
   windows, verification polls, and reappearance sweeps all flow through the same queue, so the case
   keeps advancing with zero human attention.

## Pitfalls

- **Never disclose more than the broker already shows.** Submit only `disclosure_fields`. The engine
  never volunteers SSN/ID numbers; you must not either.
- **No consent, no action.** The engine enforces this; do not work around it to "research" a third party.
- **`send-email` is idempotent + rate-limited.** It refuses to re-send a case already `submitted`
  or beyond (use `--force` only if a genuine re-send is needed), and SMTP sends are paced by
  `email_min_interval_seconds` (default 20s) with retry/backoff. Do not loop it to "make sure" -
  a successful SMTP handoff is not proof of delivery; the due-queue re-scan is the real confirmation.
- **Ledger writes are locked.** Concurrent runs (cron + manual) serialize safely; if you ever see a
  lock timeout, another run is mid-write - let it finish, don't delete the `.lock` by hand.
- **Autonomy ≠ improvisation.** Full autonomy means not *asking* between steps; it does not loosen any
  gate. If a broker demands MORE than the planned `disclosure_fields` mid-flow, stop that case and
  queue it (`human_task_queued --reason`) rather than deciding alone to disclose extra PII.
- **Don't interrupt the run with questions.** Config choices are `setup --auto`'s job; human-only work
  goes to the digest. The only mid-run question that's ever warranted is a missing-identity fact that
  blocks scanning (e.g. no city at all) - and that should have been collected at intake.
- **Use `terminal`, not `execute_code`** for `pdd.py` (secret scrubbing + output redaction break it).
- **Dossiers are plaintext by default** (JSON, `0600` under `HERMES_HOME`). For at-rest encryption run
  `$PDD setup --encryption age` - it generates a local `age` key and encrypts dossiers + ledgers (the
  audit log holds field names only and stays plaintext). It guards casual/backup/commit exposure, not
  a full-`HERMES_HOME` read; set `PDD_AGE_IDENTITY` to a separate volume for real key separation.
  `$PDD doctor` shows whether encryption is *actually* engaged (not just whether `age` is installed).
- **"Hidden from free search" ≠ deleted.** Only mark `confirmed_removed` after verifying the record is
  actually gone; note paid-tier retention in the report.
- **Soft CAPTCHAs clear by default; don't fight the hard ones.** The default cloud browser passes
  managed/soft challenges as normal operation (those brokers stay T1). For a hard interactive one it
  genuinely can't pass, record `blocked` and let the stealth/operator-browser pass take it - never a
  third-party solver service or fingerprint spoofing.
- **Broker pages change.** If a flow breaks, `$PDD record ... blocked` and flag the broker file in
  `references/brokers/` for re-verification instead of guessing.
- **Verify non-field-verified records before submitting.** `confidence: auto` records came from
  parsing BADBOOL (read `optout.notes`/`optout.links`, confirm the real opt-out URL). `confidence:
  documented` records (several people-search sites) carry the correct published opt-out URL but have
  **not** been field-verified (they 403 datacenter IPs), so confirm the live flow via the operator's
  residential browser on first use, then set `last_verified`. Field-verified curated records (no
  `confidence`, e.g. the cluster parents) have checked mechanics and take precedence.

## Verification

- `scripts/run_tests.sh tests/skills/test_unbroker_skill.py` (hermetic; no network), or the
  dependency-free runner `python3 tests/skills/test_unbroker_skill.py`.
- Dry run: `$PDD setup --auto && $PDD doctor && SID=$($PDD intake --full-name "Test Person"
  --email t@example.com --consent | python3 -c 'import sys,json;print(json.load(sys.stdin)["subject_id"])')
  && $PDD next "$SID"` and confirm a readiness summary plus an ordered action queue.
