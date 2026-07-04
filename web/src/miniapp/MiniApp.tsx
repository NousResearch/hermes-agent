import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./palettes.css";
import { MiniAppContext, type ConfirmSpec, type MiniAppTab } from "./context";
import { get, onAuthExpired, post } from "./api";
import { createAuthExpiryGate } from "./auth-expiry";
import {
  getInitData,
  hideBackButton,
  loadPersistedPalette,
  persistPalette,
  readyAndExpand,
  showBackButton,
} from "./telegram";
import type { MiniAppMeResponse, MiniAppStatusExtra } from "./types";
import { ConfirmSheet, DimBackdrop, LogPopup, PaletteSheet, Toast } from "./components/Overlays";
import { Header, TabBar } from "./components/Shell";
import { StatusScreen } from "./screens/StatusScreen";
import { SkillsScreen } from "./screens/SkillsScreen";
import { SkillDetailPane } from "./screens/SkillDetailPane";
import { CronScreen } from "./screens/CronScreen";
import { SessionsScreen } from "./screens/SessionsScreen";
import { SessionDetailPane } from "./screens/SessionDetailPane";
import { SessionNotFoundPane } from "./screens/SessionNotFoundPane";
import { UsersScreen } from "./screens/UsersScreen";

type DetailView =
  | { kind: "none" }
  | { kind: "skill"; name: string }
  | { kind: "session"; id: string }
  | { kind: "session-not-found" };

const TAB_TITLES: Record<MiniAppTab, string> = {
  status: "Hermes Agent",
  skills: "Hermes Agent",
  cron: "Hermes Agent",
  sessions: "Hermes Agent",
  users: "Hermes Agent",
};

export default function MiniApp() {
  const [tier, setTier] = useState<MiniAppMeResponse["tier"]>(null);
  const [tierLoaded, setTierLoaded] = useState(false);
  const [tab, setTab] = useState<MiniAppTab>("status");
  const [detail, setDetail] = useState<DetailView>({ kind: "none" });
  const [palette, setPalette] = useState("solarpunk");
  const [sheetOpen, setSheetOpen] = useState(false);
  const [confirm, setConfirm] = useState<ConfirmSpec | null>(null);
  const [logJob, setLogJob] = useState<{ title: string; text: string } | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [statusExtra, setStatusExtra] = useState<MiniAppStatusExtra>({
    gateway_start_time: null,
    telegram_allowlist_updated_at: null,
  });
  const [gwConnected, setGwConnected] = useState(true);
  const [gwRestarting, setGwRestarting] = useState(false);
  const [expired, setExpired] = useState(false);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // Held in a ref (not state): the onAuthExpired callback is registered
  // once on mount, and a ref read is always current inside that closure.
  // The expiry-vs-first-load decision itself lives in auth-expiry.ts.
  const authExpiryGate = useRef(createAuthExpiryGate());

  const isAdmin = tier === "admin";

  useEffect(() => {
    readyAndExpand();
    loadPersistedPalette().then((saved) => {
      if (saved) setPalette(saved);
    });
    // A 401 after a session was genuinely established means the reused
    // initData is no longer accepted -- flip to the one clear reopen screen
    // instead of every action surfacing its own misleading failure toast.
    // First-load 401s (unauthenticated browser, unpaired user) fall through
    // to the "not authorized"/"not paired" screens -- see auth-expiry.ts.
    onAuthExpired(() => {
      if (authExpiryGate.current.isExpiry401()) setExpired(true);
    });
  }, []);

  useEffect(() => {
    get<MiniAppMeResponse>("/api/miniapp/me")
      .then((me) => {
        if (me.tier) authExpiryGate.current.markSessionEstablished();
        setTier(me.tier);
      })
      .catch(() => setTier(null))
      .finally(() => setTierLoaded(true));
  }, []);

  const refreshStatus = useCallback(() => {
    get<{
      gateway_running?: boolean;
      gateway_start_time?: number | null;
      telegram_allowlist_updated_at?: number | null;
    }>("/api/status")
      .then((s) => {
        setGwConnected(s.gateway_running !== false);
        setStatusExtra({
          gateway_start_time: typeof s.gateway_start_time === "number" ? s.gateway_start_time : null,
          telegram_allowlist_updated_at:
            typeof s.telegram_allowlist_updated_at === "number" ? s.telegram_allowlist_updated_at : null,
        });
      })
      .catch(() => {
        /* the two admin fields simply stay null — fail-closed, no banner */
      });
  }, []);

  useEffect(() => {
    if (tierLoaded) refreshStatus();
  }, [tierLoaded, refreshStatus]);

  const showToast = useCallback((message: string) => {
    clearTimeout(toastTimer.current);
    setToast(message);
    toastTimer.current = setTimeout(() => setToast(null), 2200);
  }, []);

  const doRestart = useCallback(async () => {
    try {
      await post("/api/gateway/restart");
      setGwRestarting(true);
      showToast("Gateway restarting");
      setTimeout(() => {
        setGwRestarting(false);
        showToast("Gateway back online");
        refreshStatus();
      }, 2800);
    } catch {
      showToast("Restart failed to start");
    }
  }, [refreshStatus, showToast]);

  const askRestartGateway = useCallback(() => {
    setTab("status");
    setDetail({ kind: "none" });
    setConfirm({
      title: "Restart gateway?",
      body: "Platform connections (Telegram, Discord, etc.) will reconnect automatically. Any conversation currently mid-response will be interrupted, not resumed.",
      label: "Restart gateway",
      destructive: false,
      run: doRestart,
    });
  }, [doRestart]);

  const doUpdate = useCallback(async () => {
    try {
      const result = await post<{ ok: boolean; message?: string }>("/api/hermes/update");
      showToast(result.ok ? "Update started" : result.message || "Update unavailable");
      setTimeout(() => refreshStatus(), 3000);
    } catch {
      showToast("Update failed to start");
    }
  }, [refreshStatus, showToast]);

  const askUpdateHermes = useCallback(() => {
    setConfirm({
      title: "Update Hermes?",
      body: "Pulls the latest Hermes release and restarts the gateway into it. Same interruption behavior as Restart, plus whatever changed in the update.",
      label: "Update now",
      destructive: false,
      run: doUpdate,
    });
  }, [doUpdate]);

  const goTab = useCallback((next: MiniAppTab) => {
    setTab(next);
    setDetail({ kind: "none" });
  }, []);

  const inDetail = detail.kind !== "none";

  const goBack = useCallback(() => {
    if (confirm || sheetOpen || logJob) {
      setConfirm(null);
      setSheetOpen(false);
      setLogJob(null);
      return;
    }
    setDetail({ kind: "none" });
  }, [confirm, sheetOpen, logJob]);

  // BackButton: any overlay open, or a detail view open, both close on back.
  useEffect(() => {
    if (!inDetail && !confirm && !sheetOpen && !logJob) {
      hideBackButton(goBack);
      return;
    }
    showBackButton(goBack);
    return () => hideBackButton(goBack);
  }, [inDetail, confirm, sheetOpen, logJob, goBack]);

  const onPickPalette = (key: string) => {
    setPalette(key);
    setSheetOpen(false);
    persistPalette(key);
  };

  const openSkill = (name: string) => setDetail({ kind: "skill", name });
  const openSession = (id: string) => setDetail({ kind: "session", id });
  const openSessionNotFound = () => setDetail({ kind: "session-not-found" });

  const headerTitle = useMemo(() => {
    if (detail.kind === "skill") return detail.name;
    if (detail.kind === "session" || detail.kind === "session-not-found") return "Session";
    return TAB_TITLES[tab];
  }, [detail, tab]);

  const contextValue = useMemo(
    () => ({
      tier,
      isAdmin,
      tab,
      goTab,
      showToast,
      askConfirm: setConfirm,
      refreshStatus,
      askRestartGateway,
      askUpdateHermes,
      gwConnected,
      gwRestarting,
    }),
    [
      tier,
      isAdmin,
      tab,
      goTab,
      showToast,
      refreshStatus,
      askRestartGateway,
      askUpdateHermes,
      gwConnected,
      gwRestarting,
    ],
  );

  if (expired) {
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
          <div style={{ fontSize: 15, fontWeight: 650, color: "var(--mid)" }}>Session expired</div>
          <div style={{ fontSize: 12.5, color: "var(--t2)", lineHeight: 1.55, maxWidth: 260 }}>
            This dashboard has been open a while. Close it and reopen from the bot's menu button to
            continue.
          </div>
        </div>
      </div>
    );
  }

  if (!tierLoaded) {
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

  if (!tier) {
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
          <div style={{ fontSize: 15, fontWeight: 650, color: "var(--mid, #333)" }}>
            Not authorized
          </div>
          <div style={{ fontSize: 12.5, color: "var(--t2, #666)", lineHeight: 1.55, maxWidth: 260 }}>
            {getInitData()
              ? "Your Telegram account isn't paired with this Hermes instance yet. Message the bot to get started."
              : "This page only works when opened from Telegram."}
          </div>
        </div>
      </div>
    );
  }

  const overlayOpen = sheetOpen || !!confirm || !!logJob;

  return (
    <MiniAppContext.Provider value={contextValue}>
      <div className="miniapp-root" data-palette={palette}>
        <Header
          title={headerTitle}
          inDetail={inDetail}
          onBack={goBack}
          onOpenPaletteSheet={() => setSheetOpen(true)}
          gwConnected={gwConnected}
          gwRestarting={gwRestarting}
        />
        <Toast message={toast} />
        {/* min-height:0 is load-bearing: a flex child defaults to
            min-height:auto (won't shrink below content), so without it a
            long list makes this region grow instead of scroll, and the tab
            bar below gets pushed off-screen. */}
        <div
          className="miniapp-noscroll"
          style={{ flex: 1, minHeight: 0, overflowY: "auto", overscrollBehavior: "contain", WebkitOverflowScrolling: "touch" }}
        >
          {tab === "status" && !inDetail && <StatusScreen statusExtra={statusExtra} />}
          {tab === "skills" && detail.kind === "none" && <SkillsScreen onOpen={openSkill} />}
          {tab === "skills" && detail.kind === "skill" && <SkillDetailPane name={detail.name} />}
          {tab === "cron" && isAdmin && (
            <CronScreen onShowLog={(title, text) => setLogJob({ title, text })} />
          )}
          {tab === "sessions" && detail.kind === "none" && <SessionsScreen onOpen={openSession} />}
          {tab === "sessions" && detail.kind === "session" && (
            <SessionDetailPane key={detail.id} id={detail.id} onNotFound={openSessionNotFound} onBack={goBack} />
          )}
          {tab === "sessions" && detail.kind === "session-not-found" && (
            <SessionNotFoundPane onBack={goBack} />
          )}
          {tab === "users" && isAdmin && (
            <UsersScreen statusExtra={statusExtra} />
          )}
        </div>
        <TabBar tab={tab} isAdmin={isAdmin} onGo={goTab} />

        {overlayOpen && (
          <DimBackdrop
            onClick={() => {
              setSheetOpen(false);
              setConfirm(null);
              setLogJob(null);
            }}
          />
        )}
        {logJob && (
          <LogPopup
            title={logJob.title}
            text={logJob.text}
            onClose={() => setLogJob(null)}
            onCopy={() => {
              const done = () => showToast("Log copied");
              if (navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(logJob.text).then(done, done);
              } else {
                done();
              }
            }}
          />
        )}
        {sheetOpen && <PaletteSheet palette={palette} onPick={onPickPalette} />}
        {confirm && <ConfirmSheet confirm={confirm} onClose={() => setConfirm(null)} />}
      </div>
    </MiniAppContext.Provider>
  );
}
