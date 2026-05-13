---
title: "Site deployment with Vercel and Supabase"
description: "How the Hermes Agent docs/site deployment uses Vercel deploy hooks, GitHub Pages, and optional Supabase backend migrations."
---

# Site deployment with Vercel and Supabase

Hermes Agent uses GitHub Actions to keep the public docs/site deployable from repository events while keeping production credentials out of the repository.

## Deployment paths

### Vercel production deploy hook

`.github/workflows/deploy-site.yml` triggers the existing Vercel Deploy Hook for:

- published releases,
- pushes to `main` that change the site, skills catalogs, Supabase deployment files, or the deployment workflow, and
- manual `workflow_dispatch` runs.

The Vercel job does not need GitHub token permissions. It only requires the `VERCEL_DEPLOY_HOOK` repository secret and calls the hook with:

```bash
curl --fail-with-body --show-error --silent -X POST "$VERCEL_DEPLOY_HOOK"
```

Deploy hooks are bearer-style URLs. Treat the hook URL like a password: anyone with it can trigger a deployment for the configured Vercel project/branch.

### GitHub Pages docs deployment

The same workflow still publishes the Docusaurus docs artifact to GitHub Pages for push and manual runs. It intentionally skips release events so a release can trigger Vercel without failing because the Pages job runs in the wrong event context.

The Pages job has the only GitHub token permissions it needs:

- `contents: read`
- `pages: write`
- `id-token: write`

## Supabase backend support

Supabase support is split into two concerns:

1. **Application runtime variables** live in Vercel Project Settings.
2. **Database migrations** live under `supabase/` and are applied by the `supabase-migrations` job in `.github/workflows/deploy-site.yml` only after migration deployment is explicitly enabled.

A Vercel deploy hook does not carry a secret payload to the build. Configure Supabase variables directly in Vercel so each new production deployment receives them. When Supabase migrations are enabled, the workflow applies committed migrations on `main` before triggering the Vercel deploy hook so application code does not deploy ahead of its database schema.

### Vercel environment variables

Configure these in Vercel for the project that the deploy hook targets.

Public/client-safe values:

- `SUPABASE_URL` — Supabase project URL.
- `SUPABASE_PUBLISHABLE_KEY` — current Supabase browser-safe publishable key.
- `SUPABASE_ANON_KEY` — legacy browser-safe anon key if the app still expects it.

Server-only values, only if a Vercel serverless/edge backend needs elevated access:

- `SUPABASE_SECRET_KEY` — preferred secret key for backend-only access.
- `SUPABASE_SERVICE_ROLE_KEY` — legacy elevated key; never expose in browser bundles.

Do not prefix elevated keys with a public/client build prefix. Static Docusaurus or Vite bundles should only receive values that are safe to expose to every browser visitor.

### Supabase migration workflow

The migration workflow is disabled by default. Enable it only after the Supabase project and migration review process are ready.

Required GitHub repository variable:

- `SUPABASE_MIGRATIONS_ENABLED=true`

Required GitHub repository secrets:

- `SUPABASE_ACCESS_TOKEN` — Supabase personal access token for the deploying account.
- `SUPABASE_PROJECT_ID` — project ref from the Supabase dashboard URL.
- `SUPABASE_DB_PASSWORD` — database password for `supabase link` / `supabase db push`.

Optional GitHub repository variable:

- `SUPABASE_CLI_VERSION` — pin a CLI version; defaults to `latest`.

The workflow is a no-op when there are no committed `supabase/migrations/*.sql` files. Once SQL migrations exist and deployment is enabled, it runs:

```bash
supabase link --project-ref "$SUPABASE_PROJECT_ID"
supabase db push
```

## First-time setup checklist

1. In Vercel, confirm the project is connected to `NousResearch/hermes-agent` and the production branch is `main`.
2. In Vercel, create or verify a Deploy Hook for `main`.
3. In GitHub repository secrets, set `VERCEL_DEPLOY_HOOK` to that hook URL.
4. In Vercel Project Settings, add the Supabase runtime variables for Production and Preview as needed.
5. In Supabase, create/confirm the project, enable Row Level Security where browser clients access data, and copy the project ref.
6. In GitHub repository secrets, add `SUPABASE_ACCESS_TOKEN`, `SUPABASE_PROJECT_ID`, and `SUPABASE_DB_PASSWORD`.
7. After at least one reviewed SQL migration exists under `supabase/migrations/`, set repository variable `SUPABASE_MIGRATIONS_ENABLED=true`.
8. Run `Deploy Site` manually once from `main` to verify Vercel hook deployment and, if migrations are ready/enabled, the Supabase migration step.

## Rotation and recovery

- If `VERCEL_DEPLOY_HOOK` leaks, revoke it in Vercel and replace the GitHub secret.
- If a Supabase secret key or service-role key leaks, rotate/delete it in Supabase and update every server-side consumer.
- Vercel environment-variable changes apply only to new deployments; trigger a fresh deployment after changing Supabase values.
