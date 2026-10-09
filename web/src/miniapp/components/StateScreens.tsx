import type { ReactNode } from "react";
import { getInitData } from "../telegram";

function CenteredMessage({ palette, title, children }: { palette: string; title: string; children: ReactNode }) {
  return (
    <div className="miniapp-root" data-palette={palette}>
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          textAlign: "center",
          padding: 32,
        }}
      >
        <div style={{ fontSize: 15, fontWeight: 650, color: "var(--mid, #333)" }}>{title}</div>
        <div style={{ fontSize: 12.5, color: "var(--t2, #666)", lineHeight: 1.55, maxWidth: 260 }}>{children}</div>
      </div>
    </div>
  );
}

export function SessionExpiredScreen({ palette }: { palette: string }) {
  return (
    <CenteredMessage palette={palette} title="Session expired">
      This dashboard has been open a while. Close it and reopen from the bot's menu button to continue.
    </CenteredMessage>
  );
}

export function LoadingScreen({ palette }: { palette: string }) {
  return (
    <div className="miniapp-root" data-palette={palette}>
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--t3, #8a8f96)",
          fontFamily: "monospace",
          fontSize: 12,
        }}
      >
        Loading…
      </div>
    </div>
  );
}

export function NotAuthorizedScreen({ palette }: { palette: string }) {
  return (
    <CenteredMessage palette={palette} title="Not authorized">
      {getInitData()
        ? "Your Telegram account isn't paired with this Hermes instance yet. Message the bot to get started."
        : "This page only works when opened from Telegram."}
    </CenteredMessage>
  );
}
