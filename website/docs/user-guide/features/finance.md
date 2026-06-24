# Personal Finance (Plaid)

Hermes can read your real financial accounts — balances, transactions, spending by category, net worth, cashflow, and investment holdings — through [Plaid](https://plaid.com). Everything is stored **locally** in a SQLite database under your Hermes home; nothing about your finances leaves your machine unless you explicitly ask the agent to send it somewhere.

The integration is provider-agnostic by design (Plaid is the first backend), exposed two ways:

- A `hermes finance` CLI for connecting institutions and refreshing data.
- A gated `finance` toolset (`finance_*`) the agent uses to answer questions from the local cache.

The model tools are **off by default** and only appear once Plaid is configured and at least one account is linked — so users who don't use finance never ship finance tool schemas on every API call.

## Prerequisites

- A **Plaid account** with a client id and a secret. Sign up at [plaid.com](https://plaid.com); the free Sandbox tier is enough to try everything end-to-end with fake banks.
- Hermes Agent installed and running.
- For real bank data: a Plaid environment that allows production access. Plaid ships two environments — `sandbox` (fake test data) and `production` (your real banks). Start in sandbox.

## Setup

### 1. Add your Plaid keys (secrets)

Plaid credentials are secrets, so they live in `~/.hermes/.env`:

```
PLAID_CLIENT_ID=...
PLAID_SECRET=...
```

Get both from the [Plaid dashboard → Keys](https://dashboard.plaid.com/developers/keys). The secret is environment-specific — use your **sandbox** secret while testing.

### 2. Choose the environment (non-secret)

Everything that isn't a secret lives in `config.yaml` under `finance`:

```yaml
finance:
  provider: plaid
  privacy_mode: full        # full | summarized
  sync_interval: 6h
  plaid:
    environment: sandbox    # sandbox | production
    webhook_url: ""         # optional
  categorization:
    llm_fallback: false
```

### 3. Enable the toolset

```bash
hermes tools
```

Scroll to `💰 Personal Finance`, press space to toggle it on, then `s` to save. The finance tools stay opt-in; this is what surfaces them to the agent.

## Connecting an account

Hermes uses Plaid's [Hosted Link](https://plaid.com/docs/link/hosted-link/) flow: it creates a link token, hands you a URL to open in your browser, and polls until you finish — no local web server or redirect handling required.

```bash
hermes finance connect plaid
```

Hermes prints (and opens) a Plaid Link URL. Pick your institution and sign in there; when you finish, Hermes exchanges the result for an access token, stores it securely, and runs an initial sync.

### Try it with fake data first

The sandbox connects instantly with a test bank — no real credentials, no browser:

```bash
hermes finance connect plaid --sandbox
```

This links Plaid's "First Platypus Bank" test institution and syncs sample accounts and transactions so you can explore the tools immediately.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--sandbox` | Link an instant test item (sandbox data). |
| `--products a,b` | Comma-separated Plaid products (default: `transactions`). Add `liabilities`, `investments`, etc. |
| `--timeout N` | Seconds to wait for you to finish Link (default 300). |
| `--no-sync` | Skip the initial sync after linking. |
| `--no-browser` | Print the Link URL but don't auto-open a browser (SSH/headless). |

## Privacy

Local by default — the SQLite cache lives under your Hermes home and access tokens are stored separately in a `0600` file (`finance/plaid_tokens.json`), never in the database, never in `.env`, and **never returned to the model**.

`privacy_mode` controls what the agent's tools can see:

- `full` (default) — exact figures (`$2,431.18`).
- `summarized` — values are bucketed into ranges (`$2k-$5k`) *before* they ever enter a tool result, so the agent can reason about magnitude and trends without ingesting exact balances.

The `hermes finance` CLI always shows exact figures — it's you looking at your own data on your own terminal. Privacy buckets apply only to the model-facing tools.

## Using it

Once an account is linked, talk to the agent naturally — it picks the right tool:

```
> what's my current net worth
> how much did I spend on restaurants last month
> show my five largest transactions this week
> what's my monthly cashflow over the last 3 months
> list my investment holdings
> refresh my accounts
```

### Tool reference

All query tools read the **local cache only** — they never call Plaid. The one exception is `finance_sync`.

| Tool | Purpose |
|------|---------|
| `finance_accounts` | List linked accounts with balances and a net-worth summary. |
| `finance_transactions` | Query transactions (filter by account, date range, text search; sort by date or largest). |
| `finance_spending` | Spending by category over a window (default 30 days); excludes transfers. |
| `finance_net_worth` | Total assets, liabilities, and net worth by currency, optionally with a daily trend. |
| `finance_cashflow` | Monthly inflow / outflow / net for the trailing months. |
| `finance_holdings` | Investment positions (security, quantity, value). |
| `finance_sync` | Refresh data from the provider into the local cache (the only tool that contacts Plaid). |

## Categorization (rules-first)

Each transaction's category is resolved cheaply, with no LLM on the hot path, in this order:

1. **Your override** for that transaction or merchant (an override always wins).
2. **Plaid's category**.
3. **A local merchant rule** (substring / exact / regex match → category and/or a normalized display name).
4. **Optional LLM fallback** — off by default; only ever runs for otherwise-uncategorized transactions.
5. `Uncategorized`.

## CLI reference

```bash
hermes finance connect [plaid] [--sandbox] [--products ...] [--timeout N] [--no-sync] [--no-browser]
hermes finance sync [plaid] [--item-id ID]
hermes finance status [plaid]
hermes finance accounts [plaid]
hermes finance transactions [plaid] [--days N] [--limit N] [--account-id ID] [--search TEXT]
hermes finance disconnect [plaid] <item_id> [--purge]
```

`disconnect` revokes the item with Plaid and removes its access token. Cached data is kept for history unless you pass `--purge`.

## Incremental sync

Transactions use Plaid's cursor-based [`/transactions/sync`](https://plaid.com/docs/api/products/transactions/#transactionssync): the first sync pulls everything, later syncs pull only the delta. Hermes paginates fully and only writes after the final page, so a mid-pagination data mutation can't leave a partial write — on `TRANSACTIONS_SYNC_MUTATION_DURING_PAGINATION` the sync restarts from the cursor it began with. Removed transactions are **tombstoned**, never hard-deleted, so history and prior answers stay reproducible.

## Scheduling: finance + cron

Because `hermes finance sync` is a plain command, you can keep the cache fresh on a schedule:

```bash
hermes cron add --name "finance-refresh" "0 */6 * * *" \
  --script "hermes finance sync" --no-agent
```

Or let an agent session do it and summarize:

```bash
hermes cron add --name "weekly-finance" "0 8 * * 1" \
  "Sync my finances, then summarize last week's spending by category and flag anything unusual."
```

## Troubleshooting

**Tools don't appear to the agent** — finance is opt-in. Enable `💰 Personal Finance` in `hermes tools`, and make sure `PLAID_CLIENT_ID` / `PLAID_SECRET` are set and at least one account is linked (`hermes finance status`).

**`Plaid is not configured`** — add `PLAID_CLIENT_ID` and `PLAID_SECRET` to `~/.hermes/.env`. The secret must match the configured `finance.plaid.environment`.

**Link didn't return a hosted URL** — enable Hosted Link for your Plaid client (dashboard → Link → Hosted Link).

**An item needs re-authentication** — banks occasionally require you to log in again. Re-run `hermes finance connect plaid`; Plaid's update mode refreshes the existing item.

**No holdings or liabilities** — those products must be enabled for the item. Re-connect with `--products transactions,investments,liabilities`. Hermes treats missing products as "no data," not an error.

## Where things live

| Path | Contents |
|------|----------|
| `~/.hermes/finance/finance.db` | Accounts, transactions, holdings, liabilities, balance snapshots, sync state, merchant rules (SQLite). |
| `~/.hermes/finance/plaid_tokens.json` | Per-item Plaid access tokens (`0600`, never sent to the model). |
| `~/.hermes/.env` | `PLAID_CLIENT_ID`, `PLAID_SECRET`. |
| `~/.hermes/config.yaml` → `finance` | Provider, privacy mode, environment, categorization settings. |

(Paths are profile-aware — each Hermes profile gets its own finance database and tokens.)
