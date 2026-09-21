# Runtime restore recipe

KenseiAgent is the source of truth for code, skills, and plugins. Two runtime
artefacts carry secrets and therefore cannot live in this public repo verbatim.
This file records how to restore them.

## config.yaml

- **Template (in repo):** `config.yaml.example` — full structure, every secret
  value replaced with `"REDACTED"`. The only redacted keys are
  `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`,
  `GITHUB_PERSONAL_ACCESS_TOKEN`, `GOOGLE_AI_API_KEY`, and one signing `secret`.
- **Real file (live):** `~/.hermes/config.yaml`.
- **Secret backup:** `~/.hermes/secret-vault/config-backup-<timestamp>/config.yaml`
  and `~/backups/hermes-config-crons-secret-<timestamp>.tar.gz`.
- **Restore:** copy the secret backup to `~/.hermes/config.yaml`, or copy
  `config.yaml.example` and fill the five redacted values from the secret store.

## cron jobs

- **Snapshot (in repo):** `cron/jobs.snapshot.json` — sanitised record of all
  50 scheduled jobs (definitions, schedules, delivery targets). Versioned so the
  cron fleet is reviewable in git.
- **Real file (live):** `~/.hermes/cron/jobs.json` (gateway-managed; do not edit
  by hand, use `hermes cron`).
- **Secret backup:** alongside config in the secret-vault timestamped dir.
- **Restore:** recreate jobs from the snapshot via `hermes cron`, or restore the
  real `jobs.json` from the secret backup.

## .env

Never in repo. Lives only at `~/.hermes/.env`; backed up in the secret-vault
timestamped dir. Restore by copying back.

## External skills

See `skills/EXTERNAL.md` for third-party skills (e.g. `avoid-ai-writing`) that
are cloned from upstream rather than committed here.

## Regenerating the snapshots

```bash
# config template (redacts secret-bearing keys, keeps structure)
python tools/make_config_example.py   # if added; otherwise see RESTORE history
# cron snapshot
python - <<'PY'
import json, re
SENS = re.compile(r'(key|secret|token|password|client_secret|credential|webhook|bot_token|api_key|bearer)', re.I)
d = json.load(open('~/.hermes/cron/jobs.json'))
def scrub(o):
    if isinstance(o, dict): return {k:("REDACTED" if SENS.search(str(k)) and isinstance(v,str) and v else scrub(v)) for k,v in o.items()}
    if isinstance(o, list): return [scrub(x) for x in o]
    return o
json.dump(scrub(d), open('cron/jobs.snapshot.json','w'), indent=2, sort_keys=True)
PY
```
