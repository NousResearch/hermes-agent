export interface RuntimeModelInfo {
  model: string;
  provider: string;
  fallbackActivated: boolean;
  primaryModel: string;
  primaryProvider: string;
}

function nonEmptyString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

/**
 * Normalizes the session.info payload emitted by the live PTY session.
 * The configured model is intentionally not consulted here: this data is
 * runtime-only and takes precedence in the chat sidebar once it exists.
 */
export function runtimeModelFromSessionInfo(payload: unknown): RuntimeModelInfo | null {
  if (!payload || typeof payload !== "object") return null;

  const raw = payload as Record<string, unknown>;
  const model = nonEmptyString(raw.model);
  if (!model) return null;

  return {
    model,
    provider: nonEmptyString(raw.provider),
    fallbackActivated: raw.fallback_activated === true,
    primaryModel: nonEmptyString(raw.primary_model),
    primaryProvider: nonEmptyString(raw.primary_provider),
  };
}

/** Returns the visible model, preferring the model that is actually answering. */
export function selectChatModel(
  configuredModel: string,
  runtime: RuntimeModelInfo | null,
): string {
  return runtime?.model || configuredModel || "—";
}
