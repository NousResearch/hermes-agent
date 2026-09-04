import type { ReactNode } from "react";
import type { MiniAppTab } from "../context";
import { closeMiniApp, isInsideTelegram, shouldShowCloseButton } from "../telegram";

export function Header({
  title,
  inDetail,
  onBack,
  onOpenPaletteSheet,
  gwConnected,
  gwRestarting,
}: {
  title: string;
  inDetail: boolean;
  onBack: () => void;
  onOpenPaletteSheet: () => void;
  gwConnected: boolean;
  gwRestarting: boolean;
}) {
  const gwOk = gwConnected && !gwRestarting;
  return (
    <div
      style={{
        padding: "calc(10px + var(--sat)) 10px 0",
        background: "var(--headbg)",
        borderBottom: "1px solid var(--line)",
        flexShrink: 0,
      }}
    >
      <div style={{ height: 48, display: "flex", alignItems: "center", gap: 8 }}>
        {/* Back: Telegram's docs confirm BackButton is genuine native chrome
            ("back button which can be displayed in the HEADER of the Mini
            App in the Telegram interface") -- MiniApp.tsx already wires
            WebApp.BackButton to the same goBack whenever inDetail is true,
            so showing THIS in-page arrow too is a confirmed duplicate on
            every platform Telegram's own docs describe. It's kept ONLY
            outside Telegram (browser preview / local dev), where there's no
            native chrome to fall back on. */}
        {inDetail && !isInsideTelegram() && (
          <button
            onClick={onBack}
            aria-label="Back"
            style={{
              width: 36,
              height: 36,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              color: "var(--mid)",
              borderRadius: 10,
            }}
          >
            <svg width="11" height="18" viewBox="0 0 11 18" fill="none">
              <path
                d="M9.5 1.5L2 9l7.5 7.5"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        )}
        {/* Close: UNLIKE BackButton, Telegram's docs don't document
            per-platform close-chrome behavior at all. Confirmed redundant
            by direct testing on Android and Windows Desktop (both already
            show their own close control) -- those two are excluded via
            shouldShowCloseButton(). Every other platform (iOS in
            particular, unconfirmed either way) still gets a REAL button
            wired to the documented WebApp.close() API, not the inert
            decoration this used to be. Only shown when !inDetail: a detail
            view's Back control is the right affordance there, not Close. */}
        {!inDetail && shouldShowCloseButton() && (
          <button
            onClick={closeMiniApp}
            aria-label="Close mini app"
            style={{
              width: 36,
              height: 36,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              color: "var(--t2)",
              borderRadius: 10,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M1.5 1.5l11 11M12.5 1.5l-11 11"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
        )}
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 1 }}>
          <div
            style={{
              fontSize: 15,
              fontWeight: 650,
              letterSpacing: "0.01em",
              color: "var(--mid)",
              lineHeight: 1.15,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {title}
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              fontFamily: "var(--mono)",
              fontSize: 10.5,
              color: "var(--t3)",
              letterSpacing: "0.06em",
              whiteSpace: "nowrap",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: 99,
                background: gwOk ? "var(--success)" : "var(--warning)",
                animation: "miniapp-hpulse 2.4s ease-out infinite",
              }}
            />
            <span>{gwRestarting ? "gateway restarting" : gwOk ? "gateway running" : "gateway stopped"}</span>
          </div>
        </div>
        <button
          onClick={onOpenPaletteSheet}
          aria-label="Switch palette"
          style={{
            width: 36,
            height: 36,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            borderRadius: 10,
          }}
        >
          {/* A two-tone circle drawn as actual vector shapes (base circle +
              a half-circle wedge on top), not a CSS border-radius clipping a
              linear-gradient background. The CSS version showed faint
              square corners bleeding through the circular clip on some
              WebViews -- a known Chromium quirk where the gradient
              background layer can composite before the border-radius clip
              applies, especially at small sizes. SVG paths have no such
              clipping step to get wrong: the wedge IS the shape, so there
              is no square to bleed through. */}
          <svg width="18" height="18" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" fill="var(--accent)" />
            <path d="M 10 10 L 15.657 4.343 A 8 8 0 0 1 4.343 15.657 Z" fill="var(--success)" />
            <circle cx="10" cy="10" r="8" fill="none" stroke="var(--line2)" strokeWidth="1.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}

const TAB_ICONS: Record<MiniAppTab, ReactNode> = {
  status: (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="12" cy="12" r="2.6" fill="currentColor" />
    </svg>
  ),
  skills: (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
      <path
        d="M4 8.5l8-4.5 8 4.5v7l-8 4.5-8-4.5z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path d="M4 8.5l8 4.5 8-4.5M12 13v7" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
    </svg>
  ),
  cron: (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="1.7" />
      <path d="M12 7.5V12l3.2 2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  ),
  sessions: (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
      <rect x="4" y="5" width="16" height="12" rx="3.5" stroke="currentColor" strokeWidth="1.7" />
      <path d="M9 17l-2.5 3.5V17" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
    </svg>
  ),
  users: (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="8.5" r="3.6" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M5 20c1.4-3.6 3.9-5.2 7-5.2s5.6 1.6 7 5.2"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  ),
};

const TAB_LABELS: Record<MiniAppTab, string> = {
  status: "Status",
  skills: "Skills",
  cron: "Cron",
  sessions: "Sessions",
  users: "Users",
};

export function TabBar({
  tab,
  isAdmin,
  onGo,
}: {
  tab: MiniAppTab;
  isAdmin: boolean;
  onGo: (tab: MiniAppTab) => void;
}) {
  const tabs: MiniAppTab[] = isAdmin
    ? ["status", "skills", "cron", "sessions", "users"]
    : ["status", "skills", "sessions"];

  return (
    <div
      style={{
        borderTop: "1px solid var(--line)",
        background: "var(--headbg)",
        display: "flex",
        padding: "6px 6px calc(6px + var(--sab))",
        flexShrink: 0,
      }}
    >
      {tabs.map((t) => (
        <button
          key={t}
          onClick={() => onGo(t)}
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 3,
            padding: "6px 0 2px",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            color: tab === t ? "var(--accent)" : "var(--t3)",
          }}
        >
          {TAB_ICONS[t]}
          <span
            style={{
              fontSize: 9.5,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              fontFamily: "var(--mono)",
            }}
          >
            {TAB_LABELS[t]}
          </span>
        </button>
      ))}
    </div>
  );
}
