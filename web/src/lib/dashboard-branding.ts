export interface DashboardBranding {
  appName: string;
  assistantName: string;
  wordmarkLines: string[];
  title: string;
}

interface InjectedDashboardBranding {
  app_name?: unknown;
  assistant_name?: unknown;
  wordmark_lines?: unknown;
  title?: unknown;
}

declare global {
  interface Window {
    __HERMES_DASHBOARD_BRANDING__?: InjectedDashboardBranding;
  }
}

const DEFAULT_BRANDING: DashboardBranding = {
  appName: "Hermes Agent",
  assistantName: "Hermes",
  wordmarkLines: ["Hermes", "Agent"],
  title: "Hermes Agent - Dashboard",
};

function cleanString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

export function wordmarkLines(appName: string, explicit: unknown): string[] {
  if (Array.isArray(explicit)) {
    const lines = explicit
      .map((line) => cleanString(line))
      .filter((line): line is string => Boolean(line))
      .slice(0, 2);
    if (lines.length > 0) return lines;
  }

  const words = appName.split(/\s+/).filter(Boolean);
  if (words.length === 2) return words;
  return [appName];
}

export function getDashboardBranding(): DashboardBranding {
  if (typeof window === "undefined") return DEFAULT_BRANDING;

  const injected = window.__HERMES_DASHBOARD_BRANDING__ || {};
  const appName = cleanString(injected.app_name) || DEFAULT_BRANDING.appName;
  const assistantName =
    cleanString(injected.assistant_name) || DEFAULT_BRANDING.assistantName;
  const title = cleanString(injected.title) || `${appName} - Dashboard`;

  return {
    appName,
    assistantName,
    wordmarkLines: wordmarkLines(appName, injected.wordmark_lines),
    title,
  };
}
