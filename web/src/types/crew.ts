export type CrewGatewayStatus = "running" | "stopped" | "starting" | "failed" | "unknown";
export type CrewLevel = "main" | "manager" | "worker" | "qa" | "unknown";
export type CrewMetadataStatus = "classified" | "inferred" | "missing";
export type CrewHealth = "green" | "yellow" | "red" | "gray";

export interface CrewTaskSummary {
  id: string;
  title?: string | null;
  status?: string | null;
  profile?: string | null;
}

export interface CrewProfileSnapshot {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  gateway_status: CrewGatewayStatus;
  has_env: boolean;
  has_soul: boolean;
  skill_count: number;
  toolsets?: string[];
  last_seen_at?: string | null;
  current_task?: CrewTaskSummary | null;
  recent_error_count?: number;
}

export interface CrewMetadata {
  display_name?: string;
  role?: string;
  level?: CrewLevel;
  department?: string;
  manager?: string | null;
  board?: string | null;
  lanes?: string[];
  telegram_bot?: string | null;
  telegram_topic?: string | null;
}

export interface CrewNode {
  profile: CrewProfileSnapshot;
  display_name: string;
  role: string;
  level: CrewLevel;
  department: string;
  manager: string | null;
  board?: string | null;
  lanes?: string[];
  telegram_bot?: string | null;
  telegram_topic?: string | null;
  metadata_status: CrewMetadataStatus;
  health: CrewHealth;
  health_reasons: string[];
}

export interface CrewDepartment {
  name: string;
  managers: string[];
  count: number;
  nodes: CrewNode[];
}

export interface CrewSummary {
  total: number;
  main: number;
  managers: number;
  workers: number;
  qa: number;
  unknown: number;
  running: number;
  stopped: number;
  green: number;
  yellow: number;
  red: number;
  gray: number;
  unassigned: number;
}

export interface CrewSource {
  profiles: string;
  metadata: string;
  metadata_exists: boolean;
  warnings: string[];
}

export interface CrewOrganizationResponse {
  generated_at: string;
  source: CrewSource;
  summary: CrewSummary;
  nodes: CrewNode[];
  departments: CrewDepartment[];
  unassigned: CrewNode[];
}

export interface CrewControlResponse {
  generated_at: string;
  source: CrewSource;
  summary: CrewSummary;
  profiles: CrewNode[];
}

export interface CrewProfileDetail {
  generated_at: string;
  node: CrewNode;
}

// --- Crew Usage / Token Monitor types ---

export interface CrewUsageTokens {
  sessions: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

export interface CrewUsageWorker extends CrewUsageTokens {
  source: string;
  last_active: number | null;
  model: string | null;
  provider: string | null;
}

export interface CrewUsageProfileTotal extends CrewUsageTokens {
  last_active: number | null;
  model: string | null;
  provider: string | null;
  error?: string;
}

export interface CrewUsageProfile {
  profile_name: string;
  profile: CrewProfileSnapshot;
  display_name: string;
  department: string;
  level: CrewLevel;
  manager: string | null;
  total: CrewUsageProfileTotal;
  tasks: { running: number; blocked: number };
  workers: CrewUsageWorker[];
}

export interface CrewUsageDepartment {
  department: string;
  sessions: number;
  total_tokens: number;
  estimated_cost_usd: number;
  profiles: CrewUsageProfile[];
}

export interface CrewUsageResponse {
  generated_at: string;
  period_days: number;
  totals: CrewUsageTokens;
  departments: CrewUsageDepartment[];
  profiles: CrewUsageProfile[];
}
