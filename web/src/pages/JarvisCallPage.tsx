import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Mic, Bot, Music2, Globe, Sparkles, Disc3, Pause, SkipForward, Zap } from "lucide-react";
import { useLocation, useNavigate } from "react-router";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";
import { LiveVoiceCallWidget, type VoicePersona, type CallMessage, type CallLanguage } from "@/components/LiveVoiceCallWidget";
import { JarvisCoreWidget } from "@/components/JarvisCoreWidget";
import { MusicPlayerWidget } from "@/components/MusicPlayerWidget";
import { LiveWorldFeedWidget } from "@/components/LiveWorldFeedWidget";
import { GatewayClient } from "@/lib/gatewayClient";
import { api, type SessionInfo } from "@/lib/api";
import { parseMusicCommand } from "@/utils/musicCommander";

export type JarvisTab = "call" | "core" | "music" | "feed";

function safeGetStorage(key: string): string | null {
  try {
    if (typeof window !== "undefined" && typeof localStorage !== "undefined") {
      return localStorage.getItem(key);
    }
  } catch {
    // Ignore storage access issues
  }
  return null;
}

function safeSetStorage(key: string, value: string): void {
  try {
    if (typeof window !== "undefined" && typeof localStorage !== "undefined") {
      localStorage.setItem(key, value);
    }
  } catch {
    // Ignore storage access issues
  }
}

function safeRemoveStorage(key: string): void {
  try {
    if (typeof window !== "undefined" && typeof localStorage !== "undefined") {
      localStorage.removeItem(key);
    }
  } catch {
    // Ignore storage access issues
  }
}

export default function JarvisCallPage() {
  const { setEnd } = usePageHeader();
  const location = useLocation();
  const navigate = useNavigate();

  // Determine active tab from URL path or search query
  const getInitialTab = (): JarvisTab => {
    const path = location.pathname;
    if (path.includes("/jarvis-core")) return "core";
    if (path.includes("/jarvis-music")) return "music";
    if (path.includes("/jarvis-feed")) return "feed";
    if (path.includes("/jarvis-call")) return "call";

    const params = new URLSearchParams(location.search);
    const tabParam = params.get("tab");
    if (tabParam === "core" || tabParam === "music" || tabParam === "feed" || tabParam === "call") {
      return tabParam;
    }
    return "call";
  };

  const [activeTab, setActiveTab] = useState<JarvisTab>(getInitialTab);

  useEffect(() => {
    setActiveTab(getInitialTab());
  }, [location.pathname, location.search]);

  const handleTabChange = (tab: JarvisTab) => {
    setActiveTab(tab);
    navigate(`/jarvis?tab=${tab}`, { replace: true });
  };

  const gatewayRef = useRef<GatewayClient | null>(null);
  const liveSessionIdRef = useRef<string | null>(null);
  const storedSessionIdRef = useRef<string | null>(null);

  const [activeSessionId, setActiveSessionId] = useState<string | null>(() => {
    return safeGetStorage("JARVIS_ACTIVE_SESSION_ID");
  });
  const [activeSessionTitle, setActiveSessionTitle] = useState<string>("");
  const [callSessions, setCallSessions] = useState<SessionInfo[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [initialMessages, setInitialMessages] = useState<CallMessage[]>([]);
  const [musicPlayback, setMusicPlayback] = useState<{
    isPlaying: boolean;
    currentTrack: any;
  } | null>(null);

  useEffect(() => {
    const handleMusicState = (e: Event) => {
      const customEvent = e as CustomEvent<any>;
      if (customEvent.detail) {
        setMusicPlayback({
          isPlaying: !!customEvent.detail.isPlaying,
          currentTrack: customEvent.detail.currentTrack,
        });
      }
    };
    window.addEventListener("jarvis:music:state", handleMusicState);
    return () => {
      window.removeEventListener("jarvis:music:state", handleMusicState);
    };
  }, []);

  useLayoutEffect(() => {
    setEnd(
      <div className="flex items-center gap-2 text-xs font-mono text-[#00f0ff]">
        <Sparkles className="size-3.5 animate-pulse text-[#ffb700]" />
        <span>JARVIS COMMAND CENTER</span>
      </div>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd]);

  // Sync stored ref with state
  useEffect(() => {
    storedSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  const refreshCallSessions = useCallback(async () => {
    setIsHistoryLoading(true);
    try {
      const res = await api.getSessions(50, 0, undefined, "recent");
      if (res && Array.isArray(res.sessions)) {
        setCallSessions(res.sessions);
        const sid = storedSessionIdRef.current || activeSessionId;
        if (sid) {
          const current = res.sessions.find((s) => s.id === sid);
          if (current?.title) {
            setActiveSessionTitle(current.title);
          }
        }
      }
    } catch (err) {
      console.warn("[jarvis] Failed loading call sessions:", err);
    } finally {
      setIsHistoryLoading(false);
    }
  }, [activeSessionId]);

  useEffect(() => {
    void refreshCallSessions();
  }, [refreshCallSessions]);

  // Hydrate messages when activeSessionId changes or on load
  useEffect(() => {
    const sid = activeSessionId;
    if (!sid) return;

    api
      .getSessionMessages(sid)
      .then((res) => {
        if (res && Array.isArray(res.messages)) {
          const mapped: CallMessage[] = res.messages
            .filter((m) => m.role === "user" || m.role === "assistant")
            .map((m) => {
              const raw = m.content || "";
              const clean = raw
                .replace(/^\[(?:تعليمات المكالمة الصوتية الحية|VOICE CALL MODE)[^\]]*\]\s*/i, "")
                .trim();
              return {
                id: Math.random().toString(36).substring(2, 9),
                sender: (m.role === "user" ? "user" : "assistant") as "user" | "assistant",
                text: clean,
                persona: "jarvis" as VoicePersona,
                timestamp: m.timestamp
                  ? new Date(m.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                  : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
              };
            });
          if (mapped.length > 0) {
            setInitialMessages(mapped);
          }
        }
      })
      .catch((err) => {
        console.warn("[jarvis] Failed hydrating active session messages:", err);
      });
  }, [activeSessionId]);

  const handleSelectSession = useCallback(
    async (sid: string) => {
      storedSessionIdRef.current = sid;
      liveSessionIdRef.current = null;
      setActiveSessionId(sid);
      safeSetStorage("JARVIS_ACTIVE_SESSION_ID", sid);

      const found = callSessions.find((s) => s.id === sid);
      if (found?.title) {
        setActiveSessionTitle(found.title);
      }

      const gw = gatewayRef.current;
      if (gw) {
        if (gw.connectionState !== "open") {
          await gw.connect();
        }
        try {
          const resumed = await gw.request<{ session_id: string; stored_session_id?: string }>(
            "session.resume",
            { session_id: sid }
          );
          if (resumed?.session_id) {
            liveSessionIdRef.current = resumed.session_id;
            if (resumed.stored_session_id) {
              storedSessionIdRef.current = resumed.stored_session_id;
            }
          }
        } catch (err) {
          console.warn("[jarvis] session.resume via gateway deferred:", err);
        }
      }

      try {
        const res = await api.getSessionMessages(sid);
        if (res && Array.isArray(res.messages)) {
          const mapped: CallMessage[] = res.messages
            .filter((m) => m.role === "user" || m.role === "assistant")
            .map((m) => {
              const raw = m.content || "";
              const clean = raw
                .replace(/^\[(?:تعليمات المكالمة الصوتية الحية|VOICE CALL MODE)[^\]]*\]\s*/i, "")
                .trim();
              return {
                id: Math.random().toString(36).substring(2, 9),
                sender: (m.role === "user" ? "user" : "assistant") as "user" | "assistant",
                text: clean,
                persona: "jarvis" as VoicePersona,
                timestamp: m.timestamp
                  ? new Date(m.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                  : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
              };
            });
          if (mapped.length > 0) {
            setInitialMessages(mapped);
          }
        }
      } catch (err) {
        console.warn("[jarvis] Failed loading session messages:", err);
      }
    },
    [callSessions]
  );

  const handleNewSession = useCallback(() => {
    liveSessionIdRef.current = null;
    storedSessionIdRef.current = null;
    setActiveSessionId(null);
    setActiveSessionTitle("");
    safeRemoveStorage("JARVIS_ACTIVE_SESSION_ID");
    setInitialMessages([
      {
        id: Math.random().toString(36).substring(2, 9),
        sender: "system",
        text: "New Jarvis Voice Sentinel session ready. Memory & system capabilities synchronized. Click 'Start Call' to begin.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
  }, []);

  const handleDeleteSession = useCallback(
    async (sid: string) => {
      try {
        await api.deleteSession(sid);
        if (storedSessionIdRef.current === sid || activeSessionId === sid) {
          handleNewSession();
        }
        await refreshCallSessions();
      } catch (err) {
        console.warn("[jarvis] Failed deleting session:", err);
      }
    },
    [activeSessionId, handleNewSession, refreshCallSessions]
  );

  useEffect(() => {
    const gw = new GatewayClient();
    gatewayRef.current = gw;
    gw.connect().catch((err) => {
      console.warn("[jarvis] Gateway connection deferred or offline:", err);
    });

    return () => {
      gw.close();
      gatewayRef.current = null;
    };
  }, []);

  const ensureGatewaySession = useCallback(
    async (persona: VoicePersona): Promise<{ runtimeSid: string; storedSid: string }> => {
      const gw = gatewayRef.current;
      if (!gw) throw new Error("Gateway client not initialized");

      if (gw.connectionState !== "open") {
        await gw.connect();
      }

      // Check if existing live session is valid
      if (liveSessionIdRef.current) {
        return {
          runtimeSid: liveSessionIdRef.current,
          storedSid: storedSessionIdRef.current || liveSessionIdRef.current,
        };
      }

      // Try resuming stored session
      const candidateStored = storedSessionIdRef.current || activeSessionId;
      if (candidateStored) {
        try {
          const resumed = await gw.request<{ session_id: string; stored_session_id?: string }>(
            "session.resume",
            { session_id: candidateStored }
          );
          if (resumed?.session_id) {
            const runtimeSid = resumed.session_id;
            const persistentId = resumed.stored_session_id || candidateStored;
            liveSessionIdRef.current = runtimeSid;
            storedSessionIdRef.current = persistentId;
            setActiveSessionId(persistentId);
            safeSetStorage("JARVIS_ACTIVE_SESSION_ID", persistentId);
            return { runtimeSid, storedSid: persistentId };
          }
        } catch (err) {
          console.warn("[jarvis] session.resume failed, creating fresh session:", err);
        }
      }

      // Create a fresh session
      const nowStr = new Date().toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
      const title = `Jarvis Voice Sentinel (${persona === "gwen" ? "Gwen" : "Jarvis"}) - ${nowStr}`;
      const created = await gw.request<{ session_id: string; stored_session_id?: string }>(
        "session.create",
        { title }
      );
      const runtimeSid = created.session_id;
      const persistentId = created.stored_session_id || runtimeSid;
      liveSessionIdRef.current = runtimeSid;
      storedSessionIdRef.current = persistentId;
      setActiveSessionId(persistentId);
      setActiveSessionTitle(title);
      safeSetStorage("JARVIS_ACTIVE_SESSION_ID", persistentId);
      void refreshCallSessions();

      return { runtimeSid, storedSid: persistentId };
    },
    [activeSessionId, refreshCallSessions]
  );

  const handleSendMessage = useCallback(
    async (
      text: string,
      persona: VoicePersona = "jarvis",
      _language: CallLanguage = "Arabic",
      onDelta?: (partial: string) => void,
    ): Promise<string> => {
      const gw = gatewayRef.current;
      if (!gw) {
        throw new Error("Gateway client not initialized");
      }

      const sessionInfo = await ensureGatewaySession(persona);
      let currentRuntimeSid = sessionInfo.runtimeSid;
      const currentStoredSid = sessionInfo.storedSid;

      const musicCmd = parseMusicCommand(text);
      if (musicCmd) {
        window.dispatchEvent(
          new CustomEvent("jarvis:music:command", {
            detail: musicCmd,
          })
        );
      }

      // Hermes-chat mechanism: send the RAW user text with surface
      // "voice-live" and let the backend prepend VOICE_LIVE_TURN_NOTE to the
      // MODEL INPUT ONLY (tui_gateway/session_notifications._prepend_note).
      // The old code prepended a ~400-token personaInstruction to the
      // PERSISTED text on every turn, which broke per-conversation prompt
      // caching, polluted history, and added seconds of TTFB before the
      // model could "understand" the answer. Hermes chat never does that —
      // it sends what the user typed. Music side-effects are already
      // dispatched via the window event above; they don't belong in the prompt.

      return new Promise<string>((resolve, reject) => {
        let fullText = "";
        let settled = false;

        let timeout: any = null;

        const cleanup = () => {
          settled = true;
          if (timeout) clearTimeout(timeout);
          offDelta();
          offStatus();
          offTool();
          offComplete();
          offError();
        };

        const resetTimeout = (ms = 45000) => {
          if (settled) return;
          if (timeout) clearTimeout(timeout);
          timeout = setTimeout(() => {
            if (settled) return;
            cleanup();
            if (fullText.trim()) {
              resolve(fullText.trim());
            } else {
              reject(new Error("Voice response timed out"));
            }
          }, ms);
        };

        resetTimeout(45000);

        const matchSession = (evSid?: string) => {
          if (!evSid) return true;
          return evSid === currentRuntimeSid || evSid === currentStoredSid;
        };

        const offDelta = gw.on("message.delta", (ev) => {
          if (matchSession(ev.session_id) && ev.payload?.text) {
            fullText += ev.payload.text;
            // Hermes-chat mechanism: stream tokens immediately like the TUI
            // does, instead of waiting for message.complete. This is what
            // removes the perceived "takes more time to understand" delay.
            try {
              onDelta?.(fullText);
            } catch {
              /* streaming display is best-effort */
            }
            resetTimeout(35000);
          }
        });

        const offStatus = gw.on("status.update", (ev) => {
          if (matchSession(ev.session_id)) {
            // Extend timeout when Hermes is executing tools or reasoning
            resetTimeout(60000);
          }
        });

        const offTool = gw.on("reasoning.delta", (ev) => {
          if (matchSession(ev.session_id)) {
            // Extend timeout when reasoning / tool activity arrives
            resetTimeout(60000);
          }
        });

        const offComplete = gw.on("message.complete", (ev) => {
          if (matchSession(ev.session_id)) {
            cleanup();
            void refreshCallSessions();
            const finalReply = fullText.trim() || String(ev.payload?.text || "").trim();
            resolve(
              finalReply ||
                (persona === "gwen"
                  ? "تمام يا باشا، كل شيء جاهز وتحت السيطرة!"
                  : "Understood, sir. Systems operational and standing by.")
            );
          }
        });

        const offError = gw.on("error", (ev) => {
          if (matchSession(ev.session_id)) {
            cleanup();
            reject(new Error(String(ev.payload?.message || "Agent error received")));
          }
        });

        const voiceContext = `Voice Persona: ${persona === "gwen" ? "Gwen (Female AI)" : "Jarvis (Executive Male AI)"}. Spoken dialogue language: ${_language}. Keep responses concise, direct, and conversational for real-time speech. Output spoken dialogue immediately without preamble, without thinking tags (<think>), and without markdown lists so audio streams instantly.`;

        const submitPrompt = (sidToSubmit: string) => {
          gw.request("prompt.submit", {
            session_id: sidToSubmit,
            text,
            surface: "voice-live",
            voice_context: voiceContext,
          }).catch(
            async (err) => {
              if (settled) return;
              console.warn("[jarvis] prompt.submit error, checking recovery:", err);
              if (
                String(err).includes("session not found") ||
                (err && (err as any).code === 4001)
              ) {
                liveSessionIdRef.current = null;
                try {
                  const fresh = await ensureGatewaySession(persona);
                  currentRuntimeSid = fresh.runtimeSid;
                  await gw.request("prompt.submit", {
                    session_id: currentRuntimeSid,
                    text,
                    surface: "voice-live",
                    voice_context: voiceContext,
                  });
                  return;
                } catch (retryErr) {
                  cleanup();
                  reject(retryErr);
                  return;
                }
              }
              cleanup();
              reject(err);
            }
          );
        };

        submitPrompt(currentRuntimeSid);
      });
    },
    [ensureGatewaySession, refreshCallSessions]
  );

return (
  <div
    className={cn(
      "flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto p-4 sm:p-6 space-y-4 relative",
      "bg-[#030712] text-slate-100",
    )}
  >
    {/* Top Navigation Tabs Header */}
    <div className="bg-[#071526]/90 border border-[#00f0ff]/30 rounded-xl p-4 shadow-[0_0_20px_rgba(0,240,255,0.1)] flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div>
        <div className="flex flex-wrap items-center gap-2 text-[#00f0ff] font-bold text-lg font-mono">
          <Sparkles className="size-5 text-[#ffb700] animate-pulse" />
          <span>J.A.R.V.I.S. Executive OS & Autonomous Hub</span>
          <span className="px-2 py-0.5 rounded-full text-[10px] bg-emerald-500/20 border border-emerald-400/50 text-emerald-300 font-mono flex items-center gap-1 shadow-[0_0_8px_rgba(16,185,129,0.3)]">
            <span className="size-1.5 rounded-full bg-emerald-400 animate-ping" />
            <Zap className="size-3 text-amber-300" />
            24/7 AMBIENT LIVE
          </span>
        </div>
        <p className="text-xs text-[#80f7ff]/70 mt-1 max-w-2xl font-mono">
          Live Voice Sentinel, Chief of Staff AI Assistant, Holographic Audio Deck & Real-time Global Feeds integrated directly into Hermes Agent.
        </p>
      </div>

      {/* Tab Switcher Buttons */}
      <div className="flex items-center gap-1.5 bg-[#020b14] p-1.5 rounded-xl border border-[#00f0ff]/30 font-mono text-xs">
        <button
          onClick={() => handleTabChange("call")}
          className={cn(
            "px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all",
            activeTab === "call"
              ? "bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_12px_rgba(0,240,255,0.4)] border border-[#00f0ff]/40"
              : "text-[#80f7ff]/60 hover:text-[#00f0ff] border border-transparent",
          )}
        >
          <Mic className="size-3.5" />
          <span>Live Call</span>
        </button>

        <button
          onClick={() => handleTabChange("core")}
          className={cn(
            "px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all",
            activeTab === "core"
              ? "bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_12px_rgba(0,240,255,0.4)] border border-[#00f0ff]/40"
              : "text-[#80f7ff]/60 hover:text-[#00f0ff] border border-transparent",
          )}
        >
          <Bot className="size-3.5" />
          <span>Core Assistant</span>
        </button>

        <button
          onClick={() => handleTabChange("music")}
          className={cn(
            "px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all relative",
            activeTab === "music"
              ? "bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_12px_rgba(0,240,255,0.4)] border border-[#00f0ff]/40"
              : "text-[#80f7ff]/60 hover:text-[#00f0ff] border border-transparent",
          )}
        >
          <Music2 className={cn("size-3.5", musicPlayback?.isPlaying && "text-amber-400 animate-pulse")} />
          <span>Music Player</span>
          {musicPlayback?.isPlaying && (
            <span className="size-1.5 rounded-full bg-amber-400 animate-ping absolute top-1 right-1" />
          )}
        </button>

        <button
          onClick={() => handleTabChange("feed")}
          className={cn(
            "px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all",
            activeTab === "feed"
              ? "bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_12px_rgba(0,240,255,0.4)] border border-[#00f0ff]/40"
              : "text-[#80f7ff]/60 hover:text-[#00f0ff] border border-transparent",
          )}
        >
          <Globe className="size-3.5" />
          <span>World Feed</span>
        </button>
      </div>
    </div>

    {/* Active Tab Viewport - Always mounted to keep continuous audio playback & voice call active */}
    <div className="flex-1 min-h-0 flex flex-col relative">
      <div className={cn("flex-1 flex items-center justify-center", activeTab !== "call" && "hidden")}>
        <LiveVoiceCallWidget
          initialPersona="jarvis"
          activeSessionId={activeSessionId}
          activeSessionTitle={activeSessionTitle}
          callSessions={callSessions}
          isHistoryLoading={isHistoryLoading}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          onRefreshSessions={refreshCallSessions}
          initialMessages={initialMessages}
          onSendMessage={(txt, p, lang, onDelta) => handleSendMessage(txt, p, lang, onDelta)}
        />
      </div>

      <div className={cn("flex-1 min-h-[550px]", activeTab !== "core" && "hidden")}>
        <JarvisCoreWidget
          onSendMessage={(txt, onDelta) => handleSendMessage(txt, "jarvis", "Auto", onDelta)}
        />
      </div>

      <div className={cn("flex-1 min-h-[500px]", activeTab !== "music" && "hidden")}>
        <MusicPlayerWidget />
      </div>

      <div className={cn("flex-1 min-h-[550px]", activeTab !== "feed" && "hidden")}>
        <LiveWorldFeedWidget />
      </div>
    </div>

    {/* Floating Cyber Mini-Player Dock when music is active on another tab */}
    {musicPlayback?.isPlaying && activeTab !== "music" && (
      <div className="fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-[#040d1a]/95 border border-[#00f0ff]/50 shadow-[0_0_30px_rgba(0,240,255,0.35)] backdrop-blur-md font-mono text-xs animate-in fade-in slide-in-from-bottom-3 duration-300">
        <div className="size-8 rounded-lg bg-[#00f0ff]/10 border border-[#00f0ff]/40 flex items-center justify-center text-[#00f0ff] shrink-0">
          <Disc3 className="size-5 animate-spin" style={{ animationDuration: "3s" }} />
        </div>
        <div className="min-w-0 max-w-[200px] sm:max-w-[280px]">
          <p className="font-bold text-cyan-200 truncate text-[11px]">{musicPlayback.currentTrack?.title || "Playing Track"}</p>
          <p className="text-[10px] text-cyan-400/60 truncate">{musicPlayback.currentTrack?.artist || "JARVIS Audio"}</p>
        </div>
        <div className="flex items-center gap-1.5 ml-1">
          <button
            onClick={() => window.dispatchEvent(new CustomEvent("jarvis:music:command", { detail: { action: "pause" } }))}
            className="size-7 rounded-lg bg-[#00f0ff]/15 hover:bg-[#00f0ff]/25 text-[#00f0ff] flex items-center justify-center transition-all border border-[#00f0ff]/30"
            title="Pause"
          >
            <Pause className="size-3.5" />
          </button>
          <button
            onClick={() => window.dispatchEvent(new CustomEvent("jarvis:music:command", { detail: { action: "next" } }))}
            className="size-7 rounded-lg bg-[#00f0ff]/15 hover:bg-[#00f0ff]/25 text-[#00f0ff] flex items-center justify-center transition-all border border-[#00f0ff]/30"
            title="Next Track"
          >
            <SkipForward className="size-3.5" />
          </button>
          <button
            onClick={() => handleTabChange("music")}
            className="px-2.5 py-1 rounded-lg bg-[#00f0ff]/20 hover:bg-[#00f0ff]/30 text-[#00f0ff] text-[10px] font-bold border border-[#00f0ff]/40 transition-all ml-1"
            title="Open Full Player"
          >
            Open Deck
          </button>
        </div>
      </div>
    )}
  </div>
);
}
