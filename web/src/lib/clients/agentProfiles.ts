import { serviceFetchJSON } from "@/lib/clients/serviceFetch";
import type { DashboardServiceClientConfig } from "@/lib/clients/serviceFetch";

export const AGENT_PROFILES_API_BASE_URL_ENV = "VITE_AGENT_PROFILES_API_BASE_URL";

export interface AgentProfileInfo {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  has_env: boolean;
  skill_count: number;
}

export interface AgentProfilesResponse {
  profiles: AgentProfileInfo[];
}

export function listAgentProfiles(
  config?: DashboardServiceClientConfig,
): Promise<AgentProfilesResponse> {
  return serviceFetchJSON<AgentProfilesResponse>(
    AGENT_PROFILES_API_BASE_URL_ENV,
    "/api/profiles",
    undefined,
    config,
  );
}
