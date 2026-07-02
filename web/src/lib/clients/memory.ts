import { serviceFetchJSON } from "@/lib/clients/serviceFetch";
import type { DashboardServiceClientConfig } from "@/lib/clients/serviceFetch";

export const MEMORY_API_BASE_URL_ENV = "VITE_MEMORY_API_BASE_URL";

export type MemoryTarget = "memory" | "user";

export interface MemoryEntry {
  text: string;
}

export interface MemorySnapshot {
  content: string;
  entries: MemoryEntry[];
  char_count: number;
  char_limit: number;
  target: MemoryTarget;
}

export interface MemoryUpdateResponse {
  success: boolean;
  char_count: number;
  char_limit: number;
}

interface RawMemorySnapshot {
  content?: string;
  entries?: MemoryEntry[];
  char_count?: number;
  char_limit?: number;
  target?: string;
}

interface RawMemoryUpdateResponse {
  ok?: boolean;
  success?: boolean;
  char_count: number;
  char_limit: number;
}

function normalizeMemorySnapshot(snapshot: RawMemorySnapshot, fallbackTarget: MemoryTarget): MemorySnapshot {
  return {
    content: snapshot.content ?? "",
    entries: Array.isArray(snapshot.entries) ? snapshot.entries : [],
    char_count: typeof snapshot.char_count === "number" && Number.isFinite(snapshot.char_count) ? snapshot.char_count : 0,
    char_limit: typeof snapshot.char_limit === "number" && Number.isFinite(snapshot.char_limit) ? snapshot.char_limit : 0,
    target: snapshot.target === "memory" || snapshot.target === "user" ? snapshot.target : fallbackTarget,
  };
}

export async function fetchMemoryData(
  target: MemoryTarget = "memory",
  config?: DashboardServiceClientConfig,
): Promise<MemorySnapshot> {
  const raw = await serviceFetchJSON<RawMemorySnapshot>(
    MEMORY_API_BASE_URL_ENV,
    `/api/memory/content?target=${encodeURIComponent(target)}`,
    undefined,
    config,
  );
  return normalizeMemorySnapshot(raw, target);
}

export async function updateMemoryData(
  target: MemoryTarget,
  content: string,
  config?: DashboardServiceClientConfig,
): Promise<MemoryUpdateResponse> {
  const raw = await serviceFetchJSON<RawMemoryUpdateResponse>(
    MEMORY_API_BASE_URL_ENV,
    "/api/memory/content",
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target, content }),
    },
    config,
  );
  return {
    success: raw.ok ?? raw.success ?? false,
    char_count: raw.char_count,
    char_limit: raw.char_limit,
  };
}
