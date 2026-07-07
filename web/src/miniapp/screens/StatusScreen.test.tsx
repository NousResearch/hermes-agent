import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

import { MiniAppContext, type MiniAppContextValue } from "../context";
import { formatPlatformName } from "../platform-names";
import { StatusScreen } from "./StatusScreen";

// renderToStaticMarkup never runs effects (React skips them on the server),
// so StatusScreen's own useEffect (which fetches GET /api/status) never
// fires here -- no fetch mock needed. That's fine for this test's purpose:
// statusExtra is a PROP, and the point is to prove the render itself never
// surfaces gateway_start_time for a non-admin tier, regardless of what any
// fetch would have returned.

const REAL_START_TIME = 1234567890;

function contextValue(isAdmin: boolean): MiniAppContextValue {
  return {
    tier: isAdmin ? "admin" : "paired",
    isAdmin,
    tab: "status",
    goTab: () => {},
    showToast: () => {},
    askConfirm: () => {},
    refreshStatus: () => {},
    askRestartGateway: () => {},
    askUpdateHermes: () => {},
    gwConnected: true,
    gwRestarting: false,
  };
}

function renderStatus(isAdmin: boolean): string {
  return renderToStaticMarkup(
    <MiniAppContext.Provider value={contextValue(isAdmin)}>
      <StatusScreen
        statusExtra={{
          gateway_start_time: REAL_START_TIME,
          telegram_allowlist_updated_at: REAL_START_TIME + 100,
        }}
        onOpenLog={() => {}}
      />
    </MiniAppContext.Provider>,
  );
}

describe("StatusScreen tier gating", () => {
  it("non-admin render never surfaces the Uptime card's admin-only value", () => {
    const html = renderStatus(false);
    // "admin only" is the static, non-derived placeholder the non-admin
    // branch renders instead -- its presence (and the absence of "since
    // start", the admin branch's own sub-label) confirms the isAdmin ?
    // ... : ... branch actually took the non-admin path, not just that the
    // literal epoch number is absent (which formatUptime's own output
    // wouldn't literally contain anyway).
    expect(html).toContain("admin only");
    expect(html).not.toContain("since start");
  });

  it("admin render does show the Uptime card derived from gateway_start_time", () => {
    const html = renderStatus(true);
    expect(html).toContain("since start");
    expect(html).not.toContain("admin only");
  });
});

describe("formatPlatformName", () => {
  it("uppercases known acronyms fully, not just the first letter", () => {
    // Was CSS text-transform:capitalize on the raw key, which only
    // uppercases the string's first letter -- "cli" rendered as "Cli".
    expect(formatPlatformName("cli")).toBe("CLI");
  });

  it("title-cases each underscore-separated word for known multi-word overrides", () => {
    // CSS capitalize doesn't treat "_" as a word boundary at all, so
    // "api_server" rendered as "Api_server" (literal underscore visible,
    // second word never capitalized).
    expect(formatPlatformName("api_server")).toBe("API Server");
  });

  it("falls back to generic snake_case -> Title Case for unlisted platforms", () => {
    expect(formatPlatformName("telegram")).toBe("Telegram");
    expect(formatPlatformName("telegram_inline")).toBe("Telegram Inline");
    expect(formatPlatformName("whatsapp_cloud")).toBe("Whatsapp Cloud");
  });
});
