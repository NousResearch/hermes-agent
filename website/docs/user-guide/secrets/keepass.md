# KeePass (KeePassXC)

Fill website logins from a local [KeePass](https://keepass.info/) database (`.kdbx`) through the
[KeePassXC](https://keepassxc.org/) command-line tool, `keepassxc-cli`. KeePass entries show up in
`browser_vault action=list` next to your local vault items and any detected 1Password / Bitwarden
logins, and fill the same way: the agent gets an opaque handle (`kp:<group>/<title>`) and login
metadata, never the password. See [Credential vault](../features/credential-vault.md) for the
fill flow and its guarantees.

## How it works

1. You point Hermes at your database in `config.yaml` (`vault.keepass.database`).
2. The first time the agent needs a KeePass login, Hermes asks you to unlock it — a masked prompt
   in your surface (CLI panel, Desktop dialog), same as 1Password / Bitwarden. Optionally, the master
   password can come from the macOS Keychain instead (see below).
3. Hermes runs `keepassxc-cli export -f csv` in-process to list entries and keeps **metadata only**
   (title, group path, username, URL, whether a TOTP seed exists). Password and Notes columns are
   dropped before anything else sees the rows.
4. At fill time Hermes reads exactly one entry's password (`keepassxc-cli show -s -a Password <entry>`)
   and types it into the page, only on the origin from the entry's **URL** field.

Entries without a URL are never offered (there is no origin to bind them to), and entries in the
Recycle Bin are skipped. A URL without a scheme (`shop.example.org`) is treated as `https://`.

## Setup

### 1. Install `keepassxc-cli`

| Platform | Install |
|---|---|
| macOS | Install the [KeePassXC app](https://keepassxc.org/download/) (`brew install --cask keepassxc`). Hermes finds the bundled CLI at `/Applications/KeePassXC.app/Contents/MacOS/keepassxc-cli` automatically. |
| Linux | `apt install keepassxc` / `dnf install keepassxc` / Flatpak — anything that puts `keepassxc-cli` on `PATH`. |
| Other | Set `vault.keepass.binary_path` to the CLI's full path. |

Check it works: `keepassxc-cli db-info ~/path/to/file.kdbx` (it asks for your master password).

### 2. Configure the database

```yaml
vault:
  keepass:
    database: ~/Passwords/personal.kdbx   # required
    key_file: ~/Passwords/personal.keyx   # optional, if the database uses a key file
    # binary_path: /usr/local/bin/keepassxc-cli   # optional, overrides auto-detection
    # keychain_service: hermes-keepass   # optional (macOS): read the master password from the Keychain
    # keychain_account: hermes
    # enabled: false                    # opt out without removing the config
```

| Key | Default | What it does |
|---|---|---|
| `database` | — | Path to the `.kdbx` file. **Required**: without it KeePass is not a login source. |
| `key_file` | `""` | Key file passed to `keepassxc-cli -k`. |
| `binary_path` | auto | Explicit `keepassxc-cli` path. Otherwise the macOS app bundle, then `PATH`. |
| `keychain_service` | `""` | macOS only. Keychain generic-password service holding the master password. |
| `keychain_account` | `""` | Optional account name for that Keychain item. |
| `enabled` | on when configured | Set `false` to turn KeePass off (`hermes vault sources --disable keepass`). |

### 3. Confirm

```bash
hermes vault sources
```

KeePass shows as **detected** once `database` is set and the CLI is found.

## Optional: master password in the macOS Keychain

If you don't want to type the master password once per session, store it in your login Keychain
and tell Hermes the item's service name:

```bash
security add-generic-password -s hermes-keepass -a hermes -w   # prompts for the password
```

```yaml
vault:
  keepass:
    database: ~/Passwords/personal.kdbx
    keychain_service: hermes-keepass
    keychain_account: hermes
```

Hermes reads it with `security find-generic-password -w` when it needs to unlock. macOS may ask
you to allow access the first time. This trades the per-session prompt for "anyone who can run
code as your user while the Keychain is unlocked can open the database" — the same trust you
give any app you click **Always Allow** for. It also means headless sessions (cron, webhooks) can
fill KeePass logins, which an interactive-only unlock never allows.

## Headless sessions

Without a Keychain item, a locked KeePass database in a cron job, webhook or `hermes chat -q`
reports `unavailable_in_this_session` and `keepassxc-cli` is not run at all. Unlock from an
interactive session first, or use the Keychain option above.

## Security notes

- The master password goes to `keepassxc-cli` **on stdin only** — never in argv (visible to other
  processes) and never in any environment variable. It is held in process memory, scoped to the
  profile, for the vault's idle unlock window (30 minutes), and never written to disk.
  Locking the vault (Desktop **Lock**, session end) forgets it.
- Listing never exposes passwords or notes to the agent; only one entry's password is read, at
  fill time, and only for a page whose origin matches the entry's URL.
- The `.kdbx` file is opened read-only through the official CLI; Hermes never writes to it.
- TOTP: if an entry stores a TOTP seed, Hermes asks `keepassxc-cli show -t` for the current code
  when a site requests one, so 2FA fills without a prompt.
