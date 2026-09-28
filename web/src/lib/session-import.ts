import type { SessionImportResponse } from "@/lib/api";

export type ImportableSession = Record<string, unknown>;

export type SessionImportParseErrorCode = "empty" | "invalid-format";

/** Stable parser error consumed by the localized Sessions page. */
export class SessionImportParseError extends Error {
  readonly code: SessionImportParseErrorCode;

  constructor(code: SessionImportParseErrorCode) {
    super(code);
    this.code = code;
    this.name = "SessionImportParseError";
  }
}

export interface ImportSummaryCopy {
  detached: string;
  imported: string;
  skipped: string;
}

function normalizeImportSessions(value: unknown): ImportableSession[] {
  const candidate =
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Array.isArray((value as { sessions?: unknown }).sessions)
      ? (value as { sessions: unknown[] }).sessions
      : Array.isArray(value)
        ? value
        : [value];

  const sessions = candidate.filter(
    (item): item is ImportableSession =>
      !!item && typeof item === "object" && !Array.isArray(item),
  );
  if (sessions.length !== candidate.length) {
    throw new SessionImportParseError("invalid-format");
  }
  return sessions;
}

export function parseImportSessions(text: string): ImportableSession[] {
  const trimmed = text.trim();
  if (!trimmed) throw new SessionImportParseError("empty");

  try {
    return normalizeImportSessions(JSON.parse(trimmed));
  } catch {
    const lines = trimmed.split(/\r?\n/).filter((line) => line.trim());
    if (lines.length <= 1) {
      throw new SessionImportParseError("invalid-format");
    }
    try {
      return normalizeImportSessions(lines.map((line) => JSON.parse(line)));
    } catch {
      throw new SessionImportParseError("invalid-format");
    }
  }
}

function formatCount(template: string, count: number): string {
  return template.replace("{count}", String(count));
}

export function importSummary(
  result: SessionImportResponse,
  copy: ImportSummaryCopy,
): string {
  const parts = [formatCount(copy.imported, result.imported)];
  if (result.skipped > 0) {
    parts.push(formatCount(copy.skipped, result.skipped));
  }
  if (result.detached > 0) {
    parts.push(formatCount(copy.detached, result.detached));
  }
  return parts.join("; ");
}
