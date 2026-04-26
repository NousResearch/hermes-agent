import { useSyncExternalStore } from "react";
import { Loader2 } from "lucide-react";
import {
  getPluginComponent,
  getPluginLoadError,
  onPluginRegistered,
} from "./registry";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";
import type { Translations } from "@/i18n/types";

/** Renders a plugin tab once its bundle has called `register()`. */
export function PluginPage({ name }: { name: string }) {
  const { t } = useI18n();
  // Subscribe in render (via useSyncExternalStore) so we never miss
  // `register()` if the script loads before a useEffect would run.
  const Component = useSyncExternalStore(
    (onChange) => onPluginRegistered(onChange),
    () => getPluginComponent(name) ?? null,
    () => null,
  );
  const loadError = useSyncExternalStore(
    (onChange) => onPluginRegistered(onChange),
    () => getPluginLoadError(name) ?? null,
    () => null,
  );

  if (Component) {
    return <Component />;
  }

  if (loadError) {
    const message = formatPluginError(loadError, t);
    return (
      <div
        className={cn(
          "max-w-lg p-4",
          "font-medium text-sm tracking-[0.08em] text-muted-foreground",
        )}
        role="alert"
      >
        {message}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex items-center gap-2 p-4",
        "text-sm font-medium text-muted-foreground",
      )}
    >
      <Loader2 className="h-4 w-4 shrink-0 animate-spin" aria-hidden />
      <span>{t.common.loading}</span>
    </div>
  );
}

function formatPluginError(code: string, t: Translations): string {
  if (code === "LOAD_FAILED") return t.common.pluginLoadFailed;
  if (code === "NO_REGISTER") return t.common.pluginNotRegistered;
  return code;
}
