const IPC_ERROR_RE = /Error invoking remote method '[^']+': Error: (.+)$/s;
const STATUS_JSON_RE = /^\d+:\s*(\{.*\})$/s;

function extractDetail(payload: string): string | null {
  const candidate =
    payload.match(STATUS_JSON_RE)?.[1] ??
    (payload.trim().startsWith("{") ? payload.trim() : null);
  if (!candidate) return null;
  try {
    const parsed = JSON.parse(candidate) as { detail?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail.trim();
    }
  } catch {
    return null;
  }
  return null;
}

export function formatBackendError(error: unknown): string {
  const raw =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : String(error ?? "");
  const withoutIpc = raw.match(IPC_ERROR_RE)?.[1]?.trim() ?? raw.trim();
  const detail = extractDetail(withoutIpc);
  return (detail ?? withoutIpc).replace(/^Error:\s*/, "").trim();
}
