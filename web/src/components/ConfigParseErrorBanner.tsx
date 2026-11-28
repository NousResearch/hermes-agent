import { useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import type { StatusResponse } from "@/lib/api";
import { useI18n } from "@/i18n";

/**
 * config.yaml failed to parse (duplicate key, bad indent), so every reader runs on defaults or
 * last-known-good settings and the user's edits are ignored (#93263). The only other signal is a
 * warning on the terminal that started Hermes. `config_parse_error` is set only while the active
 * file still fails, so presence == show. Dismissal is session-scoped and keyed on the error text
 * so a different failure re-surfaces.
 */
const STORAGE_KEY = "configParseErrorBannerDismissed";

export function ConfigParseErrorBanner({
  status,
}: {
  status: StatusResponse | null;
}) {
  const { t } = useI18n();
  const error = status?.config_parse_error ?? null;
  const [dismissed, setDismissed] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  });
  if (!error || dismissed === error) return null;

  const template =
    t.app.configParseErrorBanner ??
    "Your config.yaml {error}. Hermes is running on fallback settings, so your saved settings are not applied. Fix: hermes config edit, then hermes config check";
  const message = template.replace("{error}", error);

  const dismiss = () => {
    try {
      sessionStorage.setItem(STORAGE_KEY, error);
    } catch {
      /* ignore */
    }
    setDismissed(error);
  };

  return (
    <div
      role="alert"
      data-testid="config-parse-error-banner"
      className="flex items-center gap-2 border-b border-amber-500/40 bg-amber-500/10 px-4 py-1.5 text-xs text-amber-300"
    >
      <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 flex-1">{message}</span>
      <button
        type="button"
        aria-label={t.app.dismiss ?? "Dismiss"}
        onClick={dismiss}
        className="shrink-0 opacity-70 hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
