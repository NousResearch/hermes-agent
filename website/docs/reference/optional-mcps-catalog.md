---
sidebar_position: 10
title: "Optional MCPs Catalog"
description: "Nous-approved optional MCP servers shipped with hermes-agent — install via hermes mcp install <name>"
---

<!-- This page is auto-generated from optional-mcps/<name>/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifests, not this page. -->

# Optional MCPs Catalog

Optional MCP servers ship with hermes-agent under `optional-mcps/` but are **not active by default**. They are discovered through `hermes mcp catalog` and activated explicitly with `hermes mcp install <name>`.

Presence in `optional-mcps/` is the trust signal: an entry is in the catalog only because a maintainer merged a PR adding it. There is no community tier and no automatic refresh — the manifest you see is the manifest you get until you re-run `hermes mcp install` after a repo update.

## CLI usage

```bash
hermes mcp                  # interactive picker (TUI) — toggle entries on/off
hermes mcp catalog          # plain-text list of Nous-approved entries (scriptable)
hermes mcp install <name>   # install a catalog entry by name (prompts for env/OAuth)
hermes mcp uninstall <name> # remove the server's config block (.env credentials are preserved)
```

`hermes mcp install` writes a `mcp_servers.<name>` block into `~/.hermes/config.yaml` using the manifest's `transport:` keys, runs any `install:` bootstrap (e.g. `git clone` + `pip install`), and prompts for any `auth:` env vars defined by the manifest. Secrets go to `~/.hermes/.env`; non-secret env vars also land in `.env` to keep one credential store.

For the general MCP config shape (independent of the catalog), see the [MCP Config Reference](/reference/mcp-config-reference). For the conceptual overview, see [MCP (Model Context Protocol)](/user-guide/features/mcp).

## Catalog entries

| Name | Transport | Auth | Source | Description |
|------|-----------|------|--------|-------------|
| [**airtable**](/docs/user-guide/mcps/optional/airtable) | http | oauth | [https://support.airtable.com/articles/9897799762-using-the-airtable-mcp-server](https://support.airtable.com/articles/9897799762-using-the-airtable-mcp-server) | Bases, tables, and records from your Airtable workspace. |
| [**algolia**](/docs/user-guide/mcps/optional/algolia) | http | oauth | [https://www.algolia.com/doc/guides/model-context-protocol/productivity-mcp](https://www.algolia.com/doc/guides/model-context-protocol/productivity-mcp) | Algolia search: indices, analytics, and settings (read-only). |
| [**alltrails**](/docs/user-guide/mcps/optional/alltrails) | http | none | [https://www.alltrails.com/mcp](https://www.alltrails.com/mcp) | AllTrails: find hikes and trails with reviews and ratings. |
| [**amplitude**](/docs/user-guide/mcps/optional/amplitude) | http | oauth | [https://amplitude.com/docs/amplitude-ai/amplitude-mcp](https://amplitude.com/docs/amplitude-ai/amplitude-mcp) | Amplitude analytics: charts, dashboards, experiments, flags. |
| [**asana**](/docs/user-guide/mcps/optional/asana) | http | oauth | [https://developers.asana.com/docs/integrating-with-asanas-mcp-server](https://developers.asana.com/docs/integrating-with-asanas-mcp-server) | Tasks, projects, and goals from your Asana workspace. |
| [**atlassian**](/docs/user-guide/mcps/optional/atlassian) | http | oauth | [https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/](https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/) | Jira issues and Confluence pages via Atlassian's hosted remote MCP. |
| [**attio**](/docs/user-guide/mcps/optional/attio) | http | oauth | [https://attio.com/help/apps/other-apps/using-the-attio-mcp-server](https://attio.com/help/apps/other-apps/using-the-attio-mcp-server) | CRM records, lists, and notes in Attio. |
| [**aws-knowledge**](/docs/user-guide/mcps/optional/aws-knowledge) | http | none | [https://awslabs.github.io/mcp/servers/aws-knowledge-mcp-server/](https://awslabs.github.io/mcp/servers/aws-knowledge-mcp-server/) | Authoritative AWS docs, API references, and best practices. |
| [**betterstack**](/docs/user-guide/mcps/optional/betterstack) | http | oauth | [https://betterstack.com/docs/getting-started/integrations/mcp/](https://betterstack.com/docs/getting-started/integrations/mcp/) | Better Stack: logs, uptime monitors, incidents, and status pages. |
| [**buildkite**](/docs/user-guide/mcps/optional/buildkite) | http | oauth | [https://buildkite.com/docs/apis/mcp-server](https://buildkite.com/docs/apis/mcp-server) | CI/CD pipelines, builds, and test results from Buildkite. |
| [**calendly**](/docs/user-guide/mcps/optional/calendly) | http | oauth | [https://developer.calendly.com/calendly-mcp-server](https://developer.calendly.com/calendly-mcp-server) | Scheduling links, events, and invitees from Calendly. |
| [**canva**](/docs/user-guide/mcps/optional/canva) | http | oauth | [https://www.canva.dev/docs/mcp/](https://www.canva.dev/docs/mcp/) | Create, search, and manage Canva designs. |
| [**circleci**](/docs/user-guide/mcps/optional/circleci) | http | oauth | [https://circleci.com/docs/guides/toolkit/circleci-mcp-overview/](https://circleci.com/docs/guides/toolkit/circleci-mcp-overview/) | CircleCI: diagnose build failures, read logs, rerun workflows. |
| [**clickup**](/docs/user-guide/mcps/optional/clickup) | http | oauth | [https://developer.clickup.com/docs/connect-an-ai-assistant-to-clickups-mcp-server](https://developer.clickup.com/docs/connect-an-ai-assistant-to-clickups-mcp-server) | Tasks, docs, and workspaces in ClickUp. |
| [**close**](/docs/user-guide/mcps/optional/close) | http | oauth | [https://help.close.com/docs/mcp-server](https://help.close.com/docs/mcp-server) | Sales CRM: leads, opportunities, calls, and emails. |
| [**cloudflare**](/docs/user-guide/mcps/optional/cloudflare) | http | oauth | [https://developers.cloudflare.com/agents/model-context-protocol/cloudflare/servers-for-cloudflare/](https://developers.cloudflare.com/agents/model-context-protocol/cloudflare/servers-for-cloudflare/) | Full Cloudflare API access via the official remote MCP. |
| [**cloudinary**](/docs/user-guide/mcps/optional/cloudinary) | http | oauth | [https://cloudinary.com/documentation/cloudinary_llm_mcp](https://cloudinary.com/documentation/cloudinary_llm_mcp) | Upload, search, and transform media assets in Cloudinary. |
| [**comfy-cloud**](/docs/user-guide/mcps/optional/comfy-cloud) | http | oauth | [https://docs.comfy.org/agent-tools/cloud](https://docs.comfy.org/agent-tools/cloud) | Generate images, video, audio, and 3D on Comfy Cloud. |
| [**context7**](/docs/user-guide/mcps/optional/context7) | http | none | [https://context7.com/docs/resources/all-clients](https://context7.com/docs/resources/all-clients) | Up-to-date, version-specific library docs and code examples. |
| [**craft**](/docs/user-guide/mcps/optional/craft) | http | oauth | [https://support.craft.do/hc/en-us/articles/29455875123101](https://support.craft.do/hc/en-us/articles/29455875123101) | Craft: structured docs, tasks, and personal knowledge base. |
| [**datadog**](/docs/user-guide/mcps/optional/datadog) | http | oauth | [https://docs.datadoghq.com/bits_ai/mcp_server/](https://docs.datadoghq.com/bits_ai/mcp_server/) | Logs, monitors, dashboards, and incidents from Datadog. |
| [**deepwiki**](/docs/user-guide/mcps/optional/deepwiki) | http | none | [https://docs.devin.ai/work-with-devin/deepwiki-mcp](https://docs.devin.ai/work-with-devin/deepwiki-mcp) | Ask questions about any public GitHub repo (Devin's DeepWiki). |
| [**dropbox**](/docs/user-guide/mcps/optional/dropbox) | http | oauth | [https://help.dropbox.com/integrations/connect-dropbox-mcp-server](https://help.dropbox.com/integrations/connect-dropbox-mcp-server) | Search, read, and manage files in Dropbox. |
| [**figma**](/docs/user-guide/mcps/optional/figma) | http | oauth | [https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/](https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/) | Official Figma remote MCP — design context, Code Connect, and write-to-canvas via https://mcp.figma.com/mcp (OAuth). |
| [**fireflies**](/docs/user-guide/mcps/optional/fireflies) | http | oauth | [https://docs.fireflies.ai/getting-started/mcp-configuration](https://docs.fireflies.ai/getting-started/mcp-configuration) | Meeting transcripts, summaries, and action items. |
| [**gamma**](/docs/user-guide/mcps/optional/gamma) | http | oauth | [https://developers.gamma.app/docs/gamma-mcp-server](https://developers.gamma.app/docs/gamma-mcp-server) | Gamma: generate and edit AI presentations, docs, and sites. |
| [**gitlab**](/docs/user-guide/mcps/optional/gitlab) | http | oauth | [https://docs.gitlab.com/user/model_context_protocol/mcp_server/](https://docs.gitlab.com/user/model_context_protocol/mcp_server/) | GitLab: issues, merge requests, pipelines, and repo context. |
| [**globalping**](/docs/user-guide/mcps/optional/globalping) | http | oauth | [https://github.com/jsdelivr/globalping-mcp-server](https://github.com/jsdelivr/globalping-mcp-server) | Ping, traceroute, DNS, and HTTP tests from global probes. |
| [**grafana**](/docs/user-guide/mcps/optional/grafana) | http | oauth | [https://grafana.com/docs/grafana-cloud/ai-tools/mcp-servers/cloud-mcp/](https://grafana.com/docs/grafana-cloud/ai-tools/mcp-servers/cloud-mcp/) | Query metrics, logs, dashboards, alerts, and incidents from Grafana Cloud. |
| [**hugging_face**](/docs/user-guide/mcps/optional/hugging_face) | http | oauth | [https://huggingface.co/docs/hub/agents-mcp](https://huggingface.co/docs/hub/agents-mcp) | Models, datasets, Spaces, and papers from the Hugging Face Hub. |
| [**indeed**](/docs/user-guide/mcps/optional/indeed) | http | oauth | [https://docs.indeed.com/indeed-mcp](https://docs.indeed.com/indeed-mcp) | Search jobs and listings on Indeed. |
| [**intercom**](/docs/user-guide/mcps/optional/intercom) | http | oauth | [https://developers.intercom.com/docs/guides/mcp](https://developers.intercom.com/docs/guides/mcp) | Conversations, tickets, and customer data from Intercom. |
| [**kiwi**](/docs/user-guide/mcps/optional/kiwi) | http | none | [https://www.kiwi.com/stories/kiwi-mcp-connector/](https://www.kiwi.com/stories/kiwi-mcp-connector/) | Kiwi.com flight search: itineraries with direct booking links. |
| [**klaviyo**](/docs/user-guide/mcps/optional/klaviyo) | http | oauth | [https://developers.klaviyo.com/en/docs/klaviyo_mcp_server](https://developers.klaviyo.com/en/docs/klaviyo_mcp_server) | Klaviyo marketing: campaigns, flows, segments, and reporting. |
| [**linear**](/docs/user-guide/mcps/optional/linear) | http | oauth | [https://linear.app/docs/mcp](https://linear.app/docs/mcp) | Find, create, and update Linear issues, projects, and comments. |
| [**microsoft-learn**](/docs/user-guide/mcps/optional/microsoft-learn) | http | none | [https://learn.microsoft.com/en-us/training/support/mcp-get-started](https://learn.microsoft.com/en-us/training/support/mcp-get-started) | Official Microsoft, Azure, and .NET docs and code samples. |
| [**miro**](/docs/user-guide/mcps/optional/miro) | http | oauth | [https://developers.miro.com/docs/connecting-to-miro-mcp](https://developers.miro.com/docs/connecting-to-miro-mcp) | Read and edit Miro boards, diagrams, and frames. |
| [**mixpanel**](/docs/user-guide/mcps/optional/mixpanel) | http | oauth | [https://docs.mixpanel.com/docs/mcp](https://docs.mixpanel.com/docs/mcp) | Mixpanel analytics: events, funnels, retention, dashboards. |
| [**monday**](/docs/user-guide/mcps/optional/monday) | http | oauth | [https://developer.monday.com/apps/docs/mondaycom-mcp-integration](https://developer.monday.com/apps/docs/mondaycom-mcp-integration) | Boards, items, docs, and workflows in monday.com. |
| [**motherduck**](/docs/user-guide/mcps/optional/motherduck) | http | oauth | [https://motherduck.com/docs/key-tasks/ai-and-motherduck/mcp-setup/](https://motherduck.com/docs/key-tasks/ai-and-motherduck/mcp-setup/) | MotherDuck: query DuckDB cloud warehouses with SQL. |
| [**n8n-official**](/docs/user-guide/mcps/optional/n8n-official) | http | oauth | [https://docs.n8n.io/connect/connect-to-n8n-mcp-server/](https://docs.n8n.io/connect/connect-to-n8n-mcp-server/) | Connect to your n8n instance's official MCP server with browser OAuth. |
| [**neon**](/docs/user-guide/mcps/optional/neon) | http | oauth | [https://neon.com/docs/ai/neon-mcp-server](https://neon.com/docs/ai/neon-mcp-server) | Neon serverless Postgres: projects, branches, and SQL. |
| [**netlify**](/docs/user-guide/mcps/optional/netlify) | http | oauth | [https://docs.netlify.com/build/build-with-ai/agent-setup-guides/agent-setup-overview/](https://docs.netlify.com/build/build-with-ai/agent-setup-guides/agent-setup-overview/) | Sites, deploys, and env vars via Netlify's hosted MCP. |
| [**notion**](/docs/user-guide/mcps/optional/notion) | http | oauth | [https://developers.notion.com/docs/mcp](https://developers.notion.com/docs/mcp) | Pages and databases from your Notion workspace. |
| [**paypal**](/docs/user-guide/mcps/optional/paypal) | http | oauth | [https://developer.paypal.com/tools/mcp-server/](https://developer.paypal.com/tools/mcp-server/) | Payments, invoices, and subscriptions via PayPal's hosted MCP. |
| [**plaid**](/docs/user-guide/mcps/optional/plaid) | http | oauth | [https://plaid.com/docs/resources/mcp/](https://plaid.com/docs/resources/mcp/) | Plaid dashboard: integrations, Items, and usage debugging. |
| [**postman**](/docs/user-guide/mcps/optional/postman) | http | oauth | [https://learning.postman.com/docs/reference/postman-api/postman-mcp-server/postman-mcp-remote-server/](https://learning.postman.com/docs/reference/postman-api/postman-mcp-server/postman-mcp-remote-server/) | Postman workspaces, collections, environments, and APIs. |
| [**prisma-postgres**](/docs/user-guide/mcps/optional/prisma-postgres) | http | oauth | [https://www.prisma.io/docs/postgres/integrations/mcp-server](https://www.prisma.io/docs/postgres/integrations/mcp-server) | Create and manage Prisma Postgres databases. |
| [**railway**](/docs/user-guide/mcps/optional/railway) | http | oauth | [https://docs.railway.com/guides/mcp-server](https://docs.railway.com/guides/mcp-server) | Railway: projects, services, deployments, and environments. |
| [**robinhood**](/docs/user-guide/mcps/optional/robinhood) | http | oauth | [https://robinhood.com/us/en/support/articles/agentic-trading-overview/](https://robinhood.com/us/en/support/articles/agentic-trading-overview/) | Robinhood agentic trading: portfolio, balances, and orders. |
| [**semgrep**](/docs/user-guide/mcps/optional/semgrep) | http | oauth | [https://semgrep.dev/docs/mcp](https://semgrep.dev/docs/mcp) | Scan code for security vulnerabilities with Semgrep. |
| [**sentry**](/docs/user-guide/mcps/optional/sentry) | http | oauth | [https://docs.sentry.io/product/sentry-mcp/](https://docs.sentry.io/product/sentry-mcp/) | Issues, stack traces, and error context from Sentry. |
| [**square**](/docs/user-guide/mcps/optional/square) | http | oauth | [https://developer.squareup.com/docs/mcp](https://developer.squareup.com/docs/mcp) | Catalog, orders, and payments via Square's hosted MCP. |
| [**strava**](/docs/user-guide/mcps/optional/strava) | http | oauth | [https://support.strava.com/en-us/articles/15401531-strava-mcp-connector](https://support.strava.com/en-us/articles/15401531-strava-mcp-connector) | Strava: activities, fitness trends, training load (read-only). |
| [**stripe**](/docs/user-guide/mcps/optional/stripe) | http | oauth | [https://docs.stripe.com/mcp](https://docs.stripe.com/mcp) | Payments, customers, and invoices via Stripe's hosted MCP. |
| [**supabase**](/docs/user-guide/mcps/optional/supabase) | http | oauth | [https://supabase.com/docs/guides/ai-tools/mcp](https://supabase.com/docs/guides/ai-tools/mcp) | Database, auth, and storage from your Supabase projects. |
| [**todoist**](/docs/user-guide/mcps/optional/todoist) | http | oauth | [https://www.todoist.com/help/articles/todoist-mcp-server](https://www.todoist.com/help/articles/todoist-mcp-server) | Manage Todoist tasks and projects. |
| [**trivago**](/docs/user-guide/mcps/optional/trivago) | http | none | [https://mcp.trivago.com/mcp](https://mcp.trivago.com/mcp) | trivago hotel search: compare prices by city and dates. |
| [**twelve-data**](/docs/user-guide/mcps/optional/twelve-data) | http | oauth | [https://twelvedata.com/docs](https://twelvedata.com/docs) | Stocks, forex, and crypto market data from Twelve Data. |
| [**twilio-docs**](/docs/user-guide/mcps/optional/twilio-docs) | http | none | [https://www.twilio.com/docs/ai/mcp](https://www.twilio.com/docs/ai/mcp) | Twilio developer docs search (public beta, read-only). |
| [**unreal-engine**](/docs/user-guide/mcps/optional/unreal-engine) | http | none | [https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor](https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor) | Drive the Unreal Engine 5.8 editor over its local MCP server. |
| [**vercel**](/docs/user-guide/mcps/optional/vercel) | http | oauth | [https://vercel.com/docs/mcp](https://vercel.com/docs/mcp) | Deployments, logs, and projects via Vercel's hosted MCP. |
| [**webflow**](/docs/user-guide/mcps/optional/webflow) | http | oauth | [https://developers.webflow.com/mcp/reference/getting-started](https://developers.webflow.com/mcp/reference/getting-started) | Sites, CMS collections, and pages via Webflow's hosted MCP. |
| [**wolfram**](/docs/user-guide/mcps/optional/wolfram) | http | none | [https://www.wolfram.com/agent-tools/](https://www.wolfram.com/agent-tools/) | Wolfram\|Alpha computation, math, and curated knowledge. |
| [**wordpress-com**](/docs/user-guide/mcps/optional/wordpress-com) | http | oauth | [https://developer.wordpress.com/docs/mcp/](https://developer.wordpress.com/docs/mcp/) | WordPress.com: posts, pages, drafts, stats, and comments. |

## Trust model

The catalog policy is intentionally narrow, and is enforced at the directory level rather than via metadata:

- **Approval is a merged PR.** Entries are added only by merging a PR into hermes-agent. Presence in the `optional-mcps/` directory equals Nous approval. There is no community tier and no trust signals beyond "it's in the catalog".
- **Manifests pin transport details.** Each manifest fixes the command, args, install URL, and git ref. MCPs are never auto-updated — users re-run `hermes mcp install <name>` explicitly to pull a new manifest version after a repo update.
- **Secrets live in `~/.hermes/.env`.** Env vars prompted at install time go to `~/.hermes/.env` (the .env-is-for-secrets rule). Non-secret env vars also go to `.env` so there is one credential store.
- **Default tool surface is conservative.** When an entry specifies `tools.default_enabled`, the install-time checklist pre-prunes mutating or rarely-useful tools — users opt in to the full surface per their threat model.

## How to contribute an entry

To propose a new optional MCP:

1. Add a directory under `optional-mcps/<name>/` containing a `manifest.yaml`.
2. Use the existing entries (`optional-mcps/linear/manifest.yaml`, `optional-mcps/n8n/manifest.yaml`) as templates. They cover the two supported transports (HTTP with native MCP OAuth; stdio with git-clone install).
3. Set `manifest_version: 1` — the current schema version constant in `hermes_cli/mcp_catalog.py`. Manifests with a higher version than the running CLI are skipped, so bumping the version is a coordinated change with the catalog loader.
4. Submit a PR. Maintainers review transport, auth, default tool surface, and source provenance. Once merged, the entry appears in `hermes mcp catalog` and gets its own page in this catalog.

See [Contributing](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) for the general PR workflow.

## See also

- [MCP Config Reference](/reference/mcp-config-reference)
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
