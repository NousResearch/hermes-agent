#!/usr/bin/env python3
"""
Vault Tool Module — Password-protected AES-256 encrypted secrets management.

Provides Hermes-native tool access to the Secrets Vault (~/.hermes/vault/).
Backward-compatible with the Bash `vault` CLI — reads/writes the same file format.

Tools:
  vault_unlock       — Unlock vault with password → export secrets as env vars
  vault_lock         — Clear vault secrets from environment
  vault_get_secret   — Retrieve a specific secret by name
  vault_set_secret   — Store a new secret (encrypted)
  vault_list_secrets — List all stored secrets (names only, no unlock needed)
  vault_check_secret — Check if a value is already stored (by SHA-8 fingerprint)
  vault_status       — Show vault state (locked/unlocked, secret count)
"""

import json
import os
import subprocess
import hashlib
import base64
from pathlib import Path
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────
VAULT_DIR = Path.home() / ".hermes" / "vault"
SALT_FILE = VAULT_DIR / ".salt"
KEYWRAP_FILE = VAULT_DIR / ".keywrap"
INDEX_FILE = VAULT_DIR / ".index"
PBKDF2_ITERATIONS = 100000
ENV_CACHE_FILE = VAULT_DIR / ".unlocked"


# ─── Helpers ──────────────────────────────────────────────────────────

def _get_vault_dir() -> Path:
    """Get vault directory from env var or default."""
    return Path(os.environ.get("HERMES_VAULT_DIR", VAULT_DIR))


def _is_initialized() -> bool:
    """Check if vault exists and has been initialized."""
    d = _get_vault_dir()
    return d.exists() and (d / ".salt").exists() and (d / ".keywrap").exists()


def _derive_key(password: str, salt: str) -> Optional[str]:
    """Derive AES-256 key from password using PBKDF2."""
    try:
        result = subprocess.run(
            ["openssl", "kdf", "-keylen", "32",
             "-kdfopt", "digest:SHA-256",
             "-kdfopt", f"pass:{password}",
             "-kdfopt", f"salt:{salt}",
             "-kdfopt", f"iter:{PBKDF2_ITERATIONS}",
             "-binary", "PBKDF2"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return base64.b64encode(result.stdout.encode("latin-1")).decode()
    except Exception:
        pass
    return None


def _decrypt_vault_key(password: str) -> Optional[str]:
    """Decrypt the wrapped vault key with user password."""
    d = _get_vault_dir()
    salt = (d / ".salt").read_text().strip()
    keywrap = d / ".keywrap"
    
    result = subprocess.run(
        ["openssl", "enc", "-d", "-aes-256-cbc", "-base64", "-pbkdf2",
         "-iter", str(PBKDF2_ITERATIONS),
         "-pass", f"pass:{password}",
         "-in", str(keywrap)],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _decrypt_secret(vault_key: str, enc_path: Path) -> Optional[str]:
    """Decrypt a single .enc file with the vault key."""
    result = subprocess.run(
        ["openssl", "enc", "-d", "-aes-256-cbc", "-base64", "-pbkdf2",
         "-iter", str(PBKDF2_ITERATIONS),
         "-pass", f"pass:{vault_key}",
         "-in", str(enc_path)],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _encrypt_secret(vault_key: str, value: str, enc_path: Path) -> bool:
    """Encrypt a value and write to .enc file."""
    result = subprocess.run(
        ["openssl", "enc", "-aes-256-cbc", "-base64", "-pbkdf2",
         "-iter", str(PBKDF2_ITERATIONS),
         "-pass", f"pass:{vault_key}",
         "-out", str(enc_path)],
        input=value, capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        enc_path.chmod(0o600)
        return True
    return False


def _sha8(value: str) -> str:
    """Compute SHA-256 first 8 hex chars (fingerprint)."""
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def _get_env_name(name: str) -> str:
    """Convert kebab-case vault name to UPPER_SNAKE_CASE env var."""
    name_map = {
        "supabase-token": "SUPABASE_ACCESS_TOKEN",
        "n8n-token": "N8N_MCP_TOKEN",
        "stripe-key": "STRIPE_SECRET_KEY",
        "stripe-webhook": "STRIPE_WEBHOOK_SECRET",
        "openai-key": "OPENAI_API_KEY",
        "openrouter-key": "OPENROUTER_API_KEY",
        "resend-key": "RESEND_API_KEY",
        "hostinger-token": "HOSTINGER_API_TOKEN",
        "github-token": "GITHUB_TOKEN",
        "twilio-account-sid": "TWILIO_ACCOUNT_SID",
        "twilio-auth-token": "TWILIO_AUTH_TOKEN",
        "twilio-verify-sid": "TWILIO_VERIFY_SID",
    }
    if name in name_map:
        return name_map[name]
    return name.replace("-", "_").upper()


def _read_index() -> list:
    """Read the vault index file. Returns list of dicts."""
    d = _get_vault_dir()
    idx_file = d / ".index"
    if not idx_file.exists():
        return []
    
    entries = []
    for line in idx_file.read_text().strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            entries.append({
                "name": parts[0],
                "env_var": parts[1],
                "sha8": parts[2].replace("sha8:", ""),
                "description": parts[3] if len(parts) > 3 else "",
                "date": parts[4] if len(parts) > 4 else "",
                "location": parts[5] if len(parts) > 5 else "",
            })
    return entries


def _update_index(name: str, env_var: str, sha8: str, description: str = "") -> None:
    """Add or update an entry in the vault index."""
    d = _get_vault_dir()
    idx_file = d / ".index"
    today = __import__("datetime").date.today().isoformat()
    
    entries = _read_index()
    new_entry = f"{name} | {env_var} | sha8:{sha8} | {description} | {today} | "
    
    # Check if already exists
    for i, entry in enumerate(entries):
        if entry["name"] == name:
            entries[i] = {
                "name": name, "env_var": env_var, "sha8": sha8,
                "description": description, "date": today, "location": ""
            }
            break
    else:
        entries.append({
            "name": name, "env_var": env_var, "sha8": sha8,
            "description": description, "date": today, "location": ""
        })
    
    # Write sorted index
    lines = ["# Vault Index — format: name | env_var | sha8 | description | stored_on | used_in"]
    for e in sorted(entries, key=lambda x: x["name"]):
        lines.append(f"{e['name']} | {e['env_var']} | sha8:{e['sha8']} | {e['description']} | {e['date']} | {e['location']}")
    
    idx_file.write_text("\n".join(lines) + "\n")


# ─── Tool Handlers ────────────────────────────────────────────────────

def vault_status() -> str:
    """Check if vault is initialized and return its state."""
    if not _is_initialized():
        return json.dumps({
            "initialized": False,
            "locked": None,
            "secret_count": 0,
            "message": "Vault not initialized. Run 'hermes vault init' first, or use the vault CLI."
        }, ensure_ascii=False)
    
    d = _get_vault_dir()
    locked = not ENV_CACHE_FILE.exists()
    enc_count = len(list(d.glob("*.enc")))
    
    return json.dumps({
        "initialized": True,
        "locked": locked,
        "secret_count": enc_count,
        "message": f"Vault {'🔒 LOCKED' if locked else '✅ UNLOCKED'} — {enc_count} secret(s)"
    }, ensure_ascii=False)


def vault_unlock(password: str) -> str:
    """
    Unlock vault with master password.
    Decrypts the vault key, then exports all secrets as environment variables.
    """
    if not _is_initialized():
        return json.dumps({"error": "Vault not initialized."}, ensure_ascii=False)
    
    vault_key = _decrypt_vault_key(password)
    if not vault_key:
        return json.dumps({"error": "Wrong password or corrupted vault."}, ensure_ascii=False)
    
    d = _get_vault_dir()
    count = 0
    secrets = []
    
    for enc_file in d.glob("*.enc"):
        name = enc_file.stem
        value = _decrypt_secret(vault_key, enc_file)
        if value:
            env_var = _get_env_name(name)
            os.environ[env_var] = value
            os.environ["VAULT_UNLOCKED"] = "1"
            ENV_CACHE_FILE.touch()
            count += 1
            secrets.append({"name": name, "env_var": env_var})
    
    return json.dumps({
        "result": "ok",
        "unlocked": True,
        "secrets_loaded": count,
        "secrets": secrets,
        "message": f"✅ Vault unlocked — {count} secret(s) loaded"
    }, ensure_ascii=False)


def vault_lock() -> str:
    """
    Lock the vault — clear all vault-related environment variables.
    """
    d = _get_vault_dir()
    ENV_CACHE_FILE.unlink(missing_ok=True)
    
    # Clear known vault env vars
    for enc_file in d.glob("*.enc"):
        name = enc_file.stem
        env_var = _get_env_name(name)
        os.environ.pop(env_var, None)
    os.environ.pop("VAULT_UNLOCKED", None)
    
    return json.dumps({
        "result": "ok",
        "locked": True,
        "message": "🔒 Vault locked — all secrets cleared from environment"
    }, ensure_ascii=False)


def vault_get_secret(name: str) -> str:
    """
    Retrieve a secret by name. Vault must be unlocked first.
    """
    if not ENV_CACHE_FILE.exists():
        return json.dumps({"error": "Vault is locked. Run vault_unlock first."}, ensure_ascii=False)
    
    vault_key = os.environ.get("_VAULT_KEY_CACHED")
    if not vault_key:
        return json.dumps({"error": "Vault key not found in session. Re-lock and unlock."}, ensure_ascii=False)
    
    d = _get_vault_dir()
    enc_path = d / f"{name}.enc"
    
    if not enc_path.exists():
        return json.dumps({"error": f"Secret '{name}' not found."}, ensure_ascii=False)
    
    value = _decrypt_secret(vault_key, enc_path)
    if not value:
        return json.dumps({"error": "Failed to decrypt secret."}, ensure_ascii=False)
    
    return json.dumps({
        "name": name,
        "env_var": _get_env_name(name),
        "value": value
    }, ensure_ascii=False)


def vault_set_secret(name: str, value: str, description: str = "") -> str:
    """
    Store a new secret. Vault must be unlocked first.
    The secret is encrypted with AES-256-CBC and indexed with SHA-8 fingerprint.
    """
    if not ENV_CACHE_FILE.exists():
        return json.dumps({"error": "Vault is locked. Run vault_unlock first."}, ensure_ascii=False)
    
    vault_key = os.environ.get("_VAULT_KEY_CACHED")
    if not vault_key:
        return json.dumps({"error": "Vault key not found in session. Re-lock and unlock."}, ensure_ascii=False)
    
    d = _get_vault_dir()
    enc_path = d / f"{name}.enc"
    
    if not _encrypt_secret(vault_key, value, enc_path):
        return json.dumps({"error": "Failed to encrypt secret."}, ensure_ascii=False)
    
    # Compute fingerprint and update index
    sha8 = _sha8(value)
    env_var = _get_env_name(name)
    _update_index(name, env_var, sha8, description)
    
    # Export as env var
    os.environ[env_var] = value
    
    return json.dumps({
        "result": "ok",
        "name": name,
        "env_var": env_var,
        "sha8": sha8,
        "message": f"✅ Secret '{name}' stored → ${{{env_var}}} (sha8:{sha8})"
    }, ensure_ascii=False)


def vault_list_secrets() -> str:
    """
    List all stored secrets with their names and env vars. No unlock needed.
    """
    entries = _read_index()
    
    d = _get_vault_dir()
    enc_files = list(d.glob("*.enc"))
    
    # Check for unindexed secrets
    indexed_names = {e["name"] for e in entries}
    unindexed = [f.stem for f in enc_files if f.stem not in indexed_names]
    
    return json.dumps({
        "secrets": entries,
        "count": len(entries),
        "unindexed": unindexed,
        "locked": not ENV_CACHE_FILE.exists(),
    }, ensure_ascii=False)


def vault_check_secret(value: str) -> str:
    """
    Check if a value is already stored in the vault by comparing SHA-8 fingerprint.
    No unlock needed.
    """
    target_sha8 = _sha8(value)
    
    for entry in _read_index():
        if entry["sha8"] == target_sha8:
            return json.dumps({
                "found": True,
                "name": entry["name"],
                "sha8": target_sha8,
                "env_var": entry["env_var"],
                "message": f"✅ Match found: '{entry['name']}' (sha8:{target_sha8})"
            }, ensure_ascii=False)
    
    return json.dumps({
        "found": False,
        "sha8": target_sha8,
        "message": f"❌ No matching secret (sha8:{target_sha8})"
    }, ensure_ascii=False)


def check_vault_requirements() -> bool:
    """Vault requires openssl and an initialized vault directory."""
    # Check openssl is available
    try:
        subprocess.run(["openssl", "version"], capture_output=True, timeout=3)
    except Exception:
        return False
    return True


# ─── Schemas ──────────────────────────────────────────────────────────

VAULT_STATUS_SCHEMA = {
    "name": "vault_status",
    "description": "Check if the secrets vault is initialized and whether it's locked or unlocked. Use this before attempting to access any secret.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VAULT_UNLOCK_SCHEMA = {
    "name": "vault_unlock",
    "description": "Unlock the secrets vault with the user's master password. Decrypts all secrets and loads them as environment variables. After calling, vault_get_secret and vault_set_secret become available. IMPORTANT: Never log or display the password.",
    "parameters": {
        "type": "object",
        "properties": {
            "password": {
                "type": "string",
                "description": "The user's vault master password. NEVER store, log, or display this value.",
            },
        },
        "required": ["password"],
    },
}

VAULT_LOCK_SCHEMA = {
    "name": "vault_lock",
    "description": "Lock the vault and clear all secrets from the environment. Call this after you're done using secrets.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VAULT_GET_SCHEMA = {
    "name": "vault_get_secret",
    "description": "Retrieve a secret by its vault name (e.g. 'supabase-token', 'stripe-key'). The vault must be unlocked first via vault_unlock.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The vault name of the secret (e.g. 'supabase-token', 'stripe-key', 'github-token').",
            },
        },
        "required": ["name"],
    },
}

VAULT_SET_SCHEMA = {
    "name": "vault_set_secret",
    "description": "Store a new secret in the vault. The vault must be unlocked first. The secret is encrypted with AES-256-CBC. Auto-computes SHA-8 fingerprint for duplicate detection.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "A descriptive name for the secret (e.g. 'stripe-key', 'openai-key', 'twilio-account-sid'). Use kebab-case.",
            },
            "value": {
                "type": "string",
                "description": "The secret value to encrypt and store. NEVER log or display this value.",
            },
            "description": {
                "type": "string",
                "description": "Optional description of what this secret is for.",
                "default": "",
            },
        },
        "required": ["name", "value"],
    },
}

VAULT_LIST_SCHEMA = {
    "name": "vault_list_secrets",
    "description": "List all stored secret names and their environment variable mappings. Does NOT require the vault to be unlocked — safe to call anytime.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VAULT_CHECK_SCHEMA = {
    "name": "vault_check_secret",
    "description": "Check if a given value is already stored in the vault by comparing its SHA-256 fingerprint. Does NOT require the vault to be unlocked. Use this before storing a new credential to avoid duplicates.",
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "description": "The value to check against the vault index.",
            },
        },
        "required": ["value"],
    },
}


# ─── Registry ─────────────────────────────────────────────────────────
from tools.registry import registry

registry.register(
    name="vault_status",
    toolset="vault",
    schema=VAULT_STATUS_SCHEMA,
    handler=lambda args, **kw: vault_status(),
    check_fn=check_vault_requirements,
    emoji="🔒",
)

registry.register(
    name="vault_unlock",
    toolset="vault",
    schema=VAULT_UNLOCK_SCHEMA,
    handler=lambda args, **kw: vault_unlock(
        password=args.get("password", ""),
    ),
    check_fn=check_vault_requirements,
    emoji="🔓",
)

registry.register(
    name="vault_lock",
    toolset="vault",
    schema=VAULT_LOCK_SCHEMA,
    handler=lambda args, **kw: vault_lock(),
    check_fn=check_vault_requirements,
    emoji="🔒",
)

registry.register(
    name="vault_get_secret",
    toolset="vault",
    schema=VAULT_GET_SCHEMA,
    handler=lambda args, **kw: vault_get_secret(
        name=args.get("name", ""),
    ),
    check_fn=check_vault_requirements,
    emoji="🔑",
)

registry.register(
    name="vault_set_secret",
    toolset="vault",
    schema=VAULT_SET_SCHEMA,
    handler=lambda args, **kw: vault_set_secret(
        name=args.get("name", ""),
        value=args.get("value", ""),
        description=args.get("description", ""),
    ),
    check_fn=check_vault_requirements,
    emoji="💾",
)

registry.register(
    name="vault_list_secrets",
    toolset="vault",
    schema=VAULT_LIST_SCHEMA,
    handler=lambda args, **kw: vault_list_secrets(),
    check_fn=check_vault_requirements,
    emoji="📋",
)

registry.register(
    name="vault_check_secret",
    toolset="vault",
    schema=VAULT_CHECK_SCHEMA,
    handler=lambda args, **kw: vault_check_secret(
        value=args.get("value", ""),
    ),
    check_fn=check_vault_requirements,
    emoji="🔍",
)
