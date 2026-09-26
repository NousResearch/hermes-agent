export function normalizeSessionTitle(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const title = raw.trim();
  return title ? title : null;
}

export function titleFromSessionInfoPayload(
  payload: unknown,
): string | null | undefined {
  if (!payload || typeof payload !== "object" || !("title" in payload)) {
    return undefined;
  }

  return normalizeSessionTitle((payload as { title?: unknown }).title);
}
