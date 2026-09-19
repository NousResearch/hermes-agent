import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import type { Translations } from "@/i18n/types";

interface PluginErrorBoundaryProps {
  children: ReactNode;
  /** Plugin manifest name — used in the fallback message and console tag. */
  name: string;
}

interface PluginErrorBoundaryState {
  error: Error | null;
}

/**
 * Isolates a single dashboard plugin's render tree so a bug in third-party
 * plugin code (undefined prop access, a bad manifest field, etc.) can't blank
 * the whole web dashboard. Desktop already wraps every window in
 * `ErrorBoundary` (apps/desktop/src/components/error-boundary.tsx); the web
 * dashboard had no equivalent around `<PluginPage>`, so the same class of
 * crash that desktop merely logs and recovers from used to take down every
 * route in the browser tab (React unmounts the whole tree above the nearest
 * boundary, and there wasn't one below the root).
 *
 * Scoped per plugin route rather than reusing one shared boundary instance:
 * each `<PluginErrorBoundary key={name}>` resets automatically when React
 * remounts the route element for a different plugin, so a crash in one
 * plugin's tab never bleeds into another plugin navigated to next.
 */
export class PluginErrorBoundary extends Component<
  PluginErrorBoundaryProps,
  PluginErrorBoundaryState
> {
  state: PluginErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): PluginErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[plugin-error-boundary:${this.props.name}]`, error, info.componentStack);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }
    return (
      <PluginErrorFallback error={error} name={this.props.name} reset={this.reset} />
    );
  }
}

function PluginErrorFallback({
  error,
  name,
  reset,
}: {
  error: Error;
  name: string;
  reset: () => void;
}) {
  const { t } = useI18n();
  return (
    <div
      className={cn("max-w-lg p-4", "font-mondwest text-sm tracking-[0.08em] text-text-secondary")}
      role="alert"
    >
      <p>{formatPluginCrashed(t, name)}</p>
      {error.message ? (
        <details className="mt-2 text-xs text-text-tertiary">
          <summary className="cursor-pointer select-none">
            {t.common.pluginCrashedDetails ?? "Error details"}
          </summary>
          <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words font-mono text-[0.6875rem]">
            {error.message}
          </pre>
        </details>
      ) : null}
      <Button className="mt-3" onClick={reset} size="sm">
        {t.common.retry}
      </Button>
    </div>
  );
}

function formatPluginCrashed(t: Translations, name: string): string {
  const message =
    t.common.pluginCrashed ??
    "This plugin ran into an error and stopped working. The rest of the dashboard is unaffected.";
  return `${name}: ${message}`;
}
