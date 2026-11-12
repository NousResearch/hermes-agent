import type { EnvVarInfo } from "./api";

/** Reconcile a successful DELETE /api/env response with the Keys page state. */
export function removeDeletedEnvVarFromState(
  vars: Record<string, EnvVarInfo> | null,
  key: string,
): Record<string, EnvVarInfo> | null {
  const info = vars?.[key];
  if (!vars || !info) return vars;

  if (info.custom) {
    const updated = { ...vars };
    delete updated[key];
    return updated;
  }

  return {
    ...vars,
    [key]: { ...info, is_set: false, redacted_value: null },
  };
}
