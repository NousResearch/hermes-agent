export type RuntimeName =
  | "codex"
  | "claude-code"
  | "ruflo"
  | "hermes"
  | "kanban"
  | "cron";

export interface HealthChip {
  name: RuntimeName | string;
  label: string;
  status: "online" | "degraded" | "offline" | "unknown";
  latencyMs?: number;
  lastChecked: string;
  port?: number;
  detail?: string;
}

export interface SwarmStatus {
  id: string;
  name: string;
  topology: string;
  workerCount: number;
  activeWorkers: number;
  queueDepth: number;
  lastActivity: string;
}

export interface SpendPoint {
  date: string;
  model: string;
  amountUsd: number;
  tokenCount: number;
}

export interface SpendHistory {
  range: string;
  points: SpendPoint[];
}

export interface SessionBrief {
  id: string;
  preview: string;
  createdAt: string;
  modelUsed?: string;
}

export interface CronBrief {
  name: string;
  nextRun: string;
  schedule: string;
}

export interface MissionSnapshot {
  model: string;
  spendToday: number;
  spendWeek: number;
  streakDays: number;
  runtimes: HealthChip[];
  swarm?: SwarmStatus;
  recentSessions: SessionBrief[];
  lastDream?: string;
  nextCron?: CronBrief;
}

export interface SSEHealthEvent extends HealthChip {
  eventType: "health";
}

export interface QueuePoint {
  date: string;
  count: number;
}

export interface QueueHistory {
  range: string;
  points: QueuePoint[];
  openNow: number;
}
