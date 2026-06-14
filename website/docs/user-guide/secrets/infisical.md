# Infisical

Pull API keys from [Infisical](https://infisical.com) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. Hermes uses Infisical Universal Auth for Machine Identities, then reads secrets from one project/environment/path and exports valid environment-variable names into `os.environ`.

## How it works

1. You create an Infisical **Machine Identity** with Universal Auth and grant it read access to a project.
2. Hermes stores only the bootstrap credentials in `~/.hermes/.env` as `INFISICAL_CLIENT_ID` and `INFISICAL_CLIENT_SECRET`.
3. Every time `hermes` starts, after `.env` has loaded, Hermes logs in through Universal Auth and calls the Infisical v4 secrets API.
4. Returned `secretKey` / `secretValue` pairs are written into `os.environ` before provider and gateway config is built.

Infisical's current API versions differ by surface: Universal Auth login is documented at `/api/v1/auth/universal-auth/login`, while secret listing is documented at `/api/v4/secrets`.

By default Hermes overwrites existing env vars with Infisical values so rotating a secret in Infisical takes effect on the next Hermes start. Set `override_existing: false` if local `.env` or shell exports should win.

## Setup

### 1. Create a Machine Identity

In Infisical:

1. Create or pick a project.
2. Add provider keys as secrets. The secret key becomes the environment variable name, e.g. `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or `SLACK_BOT_TOKEN`.
3. Create a Machine Identity with Universal Auth.
4. Grant that identity read access to the target project, environment, and path.
5. Copy the Universal Auth client ID and client secret.

### 2. Run the wizard

```bash
hermes secrets infisical setup
```

The wizard stores the bootstrap credentials in `.env`, asks for the project ID, environment, and secret path, test-fetches secrets, then enables `secrets.infisical.enabled: true`.

Non-interactive setup is also supported:

```bash
hermes secrets infisical setup \
  --client-id "$INFISICAL_CLIENT_ID" \
  --client-secret "$INFISICAL_CLIENT_SECRET" \
  --project-id <project-uuid> \
  --api-url https://us.infisical.com \
  --env prod \
  --path /
```

For Infisical EU Cloud, use `https://eu.infisical.com`. For self-hosted Infisical, set `--api-url` to your instance URL.

### 3. Confirm

```bash
hermes secrets infisical status
```

From now on, every `hermes` invocation pulls secrets at startup. You'll see a one-line summary on stderr the first time secrets are applied in a process.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets infisical setup` | Store Universal Auth credentials, configure project/path, test fetch |
| `hermes secrets infisical status` | Show config and bootstrap credential presence |
| `hermes secrets infisical sync` | Dry-run: pull secrets now and show what would be applied |
| `hermes secrets infisical sync --apply` | Pull and export into the current process environment |
| `hermes secrets infisical disable` | Flip `enabled: false`; leaves bootstrap credentials in place |

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  infisical:
    enabled: false
    api_url: https://us.infisical.com
    project_id: ""
    project_id_env: INFISICAL_PROJECT_ID
    env: prod
    path: /
    client_id_env: INFISICAL_CLIENT_ID
    client_secret_env: INFISICAL_CLIENT_SECRET
    organization_slug: ""
    override_existing: true
    cache_ttl_seconds: 300
    recursive: false
    include_imports: true
    expand_secret_references: true
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, Infisical is never contacted. |
| `api_url` | `https://us.infisical.com` | Infisical API base URL. Use `https://eu.infisical.com` for EU Cloud, or your self-hosted URL when applicable. |
| `project_id` | `""` | UUID of the project to sync from. If empty, Hermes falls back to `project_id_env`. |
| `project_id_env` | `INFISICAL_PROJECT_ID` | Env var fallback for project ID, useful for existing `infisical run` deployments. |
| `env` | `prod` | Infisical environment slug. |
| `path` | `/` | Secret path to sync. |
| `client_id_env` | `INFISICAL_CLIENT_ID` | Env var holding the Machine Identity client ID. |
| `client_secret_env` | `INFISICAL_CLIENT_SECRET` | Env var holding the Machine Identity client secret. |
| `organization_slug` | `""` | Optional organization slug for Universal Auth setups that require it. |
| `override_existing` | `true` | When true, Infisical values overwrite existing env vars. |
| `cache_ttl_seconds` | `300` | How long an in-process fetch result is reused. Set to `0` to disable caching. |
| `recursive` | `false` | Passed to Infisical's list-secrets API. |
| `include_imports` | `true` | Include imported secrets when the API returns them. |
| `expand_secret_references` | `true` | Ask Infisical to expand secret references in returned values. |

## Failure modes

Infisical never blocks Hermes startup. If anything goes wrong, Hermes prints a warning and continues with credentials from `.env` or the shell.

| Symptom | Cause | Fix |
|---|---|---|
| `INFISICAL_CLIENT_ID is not set` | Enabled in config but bootstrap ID is missing | Re-run setup or add it to `.env` |
| `INFISICAL_CLIENT_SECRET is not set` | Enabled in config but bootstrap secret is missing | Re-run setup or add it to `.env` |
| `project_id is empty` | No project ID in config or `INFISICAL_PROJECT_ID` | Set `secrets.infisical.project_id` or `INFISICAL_PROJECT_ID` |
| `HTTP 401` / `HTTP 403` | Machine Identity credentials revoked or missing access | Regenerate credentials or fix project permissions |
| `not a valid env-var name` | A secret key contains spaces, dashes, or starts with a digit | Rename the secret key to an env-var-safe name |

## Security notes

- The Universal Auth client secret is sensitive. Anyone with the client ID and client secret can read every secret allowed by that Machine Identity.
- Hermes refuses to let Infisical overwrite the bootstrap credential env vars themselves, even with `override_existing: true`.
- Secret values are cached only in process memory for `cache_ttl_seconds`; this backend does not write fetched Infisical secrets to disk.
- This integration helps secrets consumed by Hermes itself. Sibling containers or services that need secrets before Hermes starts still need their own Infisical integration, platform env injection, or wrapper.

## When NOT to use this

- Single-machine personal setups where `~/.hermes/.env` is enough.
- Air-gapped deployments that cannot reach your Infisical API.
- Services outside the Hermes process that need secrets at container startup.
