// Credential-vault API types + client for the dashboard Vault page.
// Moved out of api.ts so that facade stays under its code-health line cap
// (same split as api-files.ts / api-aux.ts).

export interface VaultItem {
  id: string;
  backend?: string;
  kind: string;
  label: string;
  origin?: string | null;
  identifier?: string | null;
  identifier_type?: string | null;
  has_otp?: boolean;
  created_at?: string;
}

/** One login source as the vault API describes it: the local store or an external manager. */
export interface VaultSource {
  name: string;
  display_name: string;
  enabled: boolean;
  needs_unlock: boolean;
  unlocked: boolean;
  installed: boolean;
}

type Fetch = <T>(path: string, init?: RequestInit) => Promise<T>;

const JSON_POST = { "Content-Type": "application/json" };

export function createVaultApi(fetchJSON: Fetch) {
  return {
    getVaultItems: () => fetchJSON<{ items: VaultItem[]; locked: string[] }>("/api/vault/items"),
    getVaultSources: () => fetchJSON<{ sources: VaultSource[] }>("/api/vault/sources"),
    addVaultItem: (
      kind: string,
      label: string,
      origin: string | undefined,
      secret: Record<string, string>,
    ) =>
      fetchJSON<{ id: string }>("/api/vault/items", {
        method: "POST",
        headers: JSON_POST,
        body: JSON.stringify({ kind, label, origin: origin || undefined, secret }),
      }),
    removeVaultItem: (id: string) =>
      fetchJSON<{ removed: boolean }>(`/api/vault/items/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    setVaultSourceEnabled: (name: string, enabled: boolean) =>
      fetchJSON<{ name: string; enabled: boolean }>(
        `/api/vault/sources/${encodeURIComponent(name)}`,
        { method: "PUT", headers: JSON_POST, body: JSON.stringify({ enabled }) },
      ),
    unlockVaultSource: (name: string, password: string) =>
      fetchJSON<{ name: string; unlocked: boolean }>("/api/vault/unlock", {
        method: "POST",
        headers: JSON_POST,
        body: JSON.stringify({ name, password }),
      }),
    lockVaultSources: (name?: string) =>
      fetchJSON<{ locked: boolean }>("/api/vault/lock", {
        method: "POST",
        headers: JSON_POST,
        body: JSON.stringify({ name: name || undefined }),
      }),
  };
}
