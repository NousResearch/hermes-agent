import { appendProfileParam, fetchJSON, getManagementProfile } from "./api";

export interface ConfigRevisionResponse {
  mtime_ns: number;
  path: string;
  size: number;
}

/** Profile-scoped configuration reads and writes share the dashboard transport. */
export const configApi = {
  getConfig: (profile = getManagementProfile()) =>
    fetchJSON<Record<string, unknown>>(appendProfileParam("/api/config", profile)),
  getConfigRevision: (profile = getManagementProfile()) =>
    fetchJSON<ConfigRevisionResponse>(appendProfileParam("/api/config/revision", profile)),
  getDefaults: () => fetchJSON<Record<string, unknown>>("/api/config/defaults"),
  getSchema: () => fetchJSON<{ fields: Record<string, unknown>; category_order: string[] }>("/api/config/schema"),
  saveConfig: (config: Record<string, unknown>, profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean }>(appendProfileParam("/api/config", profile), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }),
  getConfigRaw: (profile = getManagementProfile()) =>
    fetchJSON<{ yaml: string; path?: string }>(
      appendProfileParam("/api/config/raw", profile),
    ),
  saveConfigRaw: (yaml_text: string, profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean }>(appendProfileParam("/api/config/raw", profile), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_text }),
    }),
};
