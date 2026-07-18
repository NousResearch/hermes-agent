import type { McpOAuthFlow } from "./api";

type CompleteOptions = {
  serverName: string;
  start: (name: string) => Promise<McpOAuthFlow>;
  status: (flowId: string) => Promise<McpOAuthFlow>;
  open: (url?: string | URL, target?: string, features?: string) => unknown;
  sleep?: (milliseconds: number) => Promise<void>;
  maxPollFailures?: number;
};

export type McpDashboardOAuthErrorCode =
  | "popup-blocked"
  | "start-failed"
  | "authorization-url-missing"
  | "authorization-failed"
  | "authorization-window-closed";

const ERROR_MESSAGES: Record<McpDashboardOAuthErrorCode, string> = {
  "popup-blocked":
    "OAuth popup was blocked — allow popups for this dashboard and retry",
  "start-failed": "OAuth failed to start",
  "authorization-url-missing":
    "OAuth server did not provide an authorization URL",
  "authorization-failed": "OAuth authorization failed",
  "authorization-window-closed":
    "OAuth authorization window was closed before completion",
};

/** Stable frontend failures let each Dashboard locale own its presentation. */
export class McpDashboardOAuthError extends Error {
  constructor(public readonly code: McpDashboardOAuthErrorCode) {
    super(ERROR_MESSAGES[code]);
    this.name = "McpDashboardOAuthError";
  }
}

const defaultSleep = (milliseconds: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

export async function completeMcpDashboardOAuth({
  serverName,
  start,
  status,
  open,
  sleep = defaultSleep,
  maxPollFailures = 3,
}: CompleteOptions): Promise<McpOAuthFlow> {
  // Open synchronously from the click handler, before the first await. Browsers
  // otherwise classify the later OAuth popup as unsolicited and block it.
  const authWindow = open("about:blank", "_blank") as Window | null;
  if (!authWindow) {
    throw new McpDashboardOAuthError("popup-blocked");
  }
  authWindow.opener = null;
  let started: McpOAuthFlow;
  try {
    started = await start(serverName);
    if (started.status === "error") {
      if (started.error) throw new Error(started.error);
      throw new McpDashboardOAuthError("start-failed");
    }
    if (!started.authorization_url) {
      throw new McpDashboardOAuthError("authorization-url-missing");
    }
    authWindow.location.href = started.authorization_url;
  } catch (error) {
    authWindow.close();
    throw error;
  }

  let pollFailures = 0;
  for (;;) {
    let current: McpOAuthFlow;
    try {
      current = await status(started.flow_id);
      pollFailures = 0;
    } catch (error) {
      pollFailures += 1;
      if (pollFailures >= maxPollFailures) throw error;
      await sleep(1000);
      continue;
    }
    if (current.status === "approved") return current;
    if (current.status === "error") {
      if (current.error) throw new Error(current.error);
      throw new McpDashboardOAuthError("authorization-failed");
    }
    if (authWindow.closed) {
      throw new McpDashboardOAuthError("authorization-window-closed");
    }
    await sleep(1000);
  }
}
