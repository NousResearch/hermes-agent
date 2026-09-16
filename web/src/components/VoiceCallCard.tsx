/** Dashboard realtime voice call — own sidecar session, not the embedded TUI. */

import {
  isVoiceStopCommand,
  RealtimeVoiceClient,
  type RealtimeTokenGrant,
  VOICE_SUPERVISOR_GATEWAY_EVENTS,
  VoiceSupervisorController,
  voiceSupervisorSurfaceEvent,
  voiceSupervisorSurfaceRequest,
} from "@hermes/shared";
import { Mic, MicOff, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { useI18n } from "@/i18n";
import { GatewayClient } from "@/lib/gatewayClient";
import { cn } from "@/lib/utils";
import { voiceCallSessionCreateParams } from "@/lib/voiceCallSession";

type CallStatus = "idle" | "connecting" | "listening" | "speaking" | "thinking";

export function VoiceCallCard({ profile }: { profile?: string | null }) {
  const { t } = useI18n();
  const [status, setStatus] = useState<CallStatus>("idle");
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const gwRef = useRef<GatewayClient | null>(null);
  const clientRef = useRef<RealtimeVoiceClient | null>(null);
  const controllerRef = useRef<VoiceSupervisorController | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  // Turns still awaiting their terminal message.complete (FIFO — the gateway
  // emits one per accepted prompt.submit, interrupted/errored turns included).
  // Busy/queue state for the controller derives from this single counter.
  const pendingTurnsRef = useRef(0);

  const stop = useCallback(() => {
    controllerRef.current?.failActiveConsult("Voice session ended.");
    controllerRef.current = null;
    clientRef.current?.close();
    clientRef.current = null;
    gwRef.current?.close();
    gwRef.current = null;
    sessionIdRef.current = null;
    pendingTurnsRef.current = 0;
    setStatus("idle");
    setMuted(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    if (clientRef.current) {
      return;
    }
    setError(null);
    setStatus("connecting");
    try {
      const gw = new GatewayClient();
      gwRef.current = gw;
      await gw.connect();
      // Same profile as the sidecar session below — the token carries that
      // profile's voice.realtime config and xAI credentials.
      const grant = await gw.request<RealtimeTokenGrant>("voice.realtime_token", {
        ...(profile ? { profile } : {}),
      });
      // Invented ids 404 at prompt.submit — the gateway only routes sessions
      // it created, so mint a real sidecar session first.
      const created = await gw.request<{ session_id: string }>(
        "session.create",
        voiceCallSessionCreateParams(profile),
      );
      const sessionId = String(created.session_id || "").trim();
      if (!sessionId) {
        throw new Error("voice session.create returned no session_id");
      }
      sessionIdRef.current = sessionId;

      const client = new RealtimeVoiceClient();
      clientRef.current = client;
      const controller = new VoiceSupervisorController(client, {
        submit: async (task) => {
          const sid = sessionIdRef.current;
          const liveGw = gwRef.current;
          if (!sid || !liveGw) {
            return false;
          }
          try {
            await liveGw.request("prompt.submit", {
              session_id: sid,
              text: task,
              ...(profile ? { profile } : {}),
            });
            pendingTurnsRef.current += 1;
            return true;
          } catch {
            return false;
          }
        },
        interrupt: async () => {
          const sid = sessionIdRef.current;
          const liveGw = gwRef.current;
          if (!sid || !liveGw) {
            return;
          }
          await liveGw.request("session.interrupt", { session_id: sid }).catch(() => undefined);
        },
        isBusy: () => pendingTurnsRef.current > 0,
        isQueueEmpty: () => pendingTurnsRef.current === 0,
      });
      controllerRef.current = controller;

      gw.on("message.complete", (ev) => {
        if (ev.session_id !== sessionIdRef.current) {
          return;
        }
        pendingTurnsRef.current = Math.max(0, pendingTurnsRef.current - 1);
        const live = controllerRef.current;
        if (!live?.consultActive) {
          return;
        }
        if (pendingTurnsRef.current > 0) {
          // A steer interrupted this turn and resubmitted — the consult's
          // answer is the LAST submitted turn, not this partial one.
          return;
        }
        // Empty-fallback and truncation belong to the controller's
        // onTurnComplete — no local copy.
        const text = String((ev.payload as { text?: string } | undefined)?.text ?? "").trim();
        live.onTurnComplete(live.currentTask ?? "", text);
      });
      for (const eventName of VOICE_SUPERVISOR_GATEWAY_EVENTS) {
        gw.on(eventName, (ev) => {
          if (ev.session_id !== sessionIdRef.current) {
            return;
          }
          const surfaceEvent = voiceSupervisorSurfaceEvent(ev);
          if (surfaceEvent?.kind === "narrate-tool") {
            controllerRef.current?.narrateTool(surfaceEvent.name);
          } else if (surfaceEvent) {
            controllerRef.current?.notify(surfaceEvent.text);
          }
        });
      }
      // Blocking prompts arrive as server→client requests. Observe only
      // (decline with `false`) so the card never answers a prompt the voice
      // cannot resolve — it just tells the user to look at the app.
      gw.onRequest((request) => {
        if (!request.replayed && request.params.session_id === sessionIdRef.current) {
          const surfaceEvent = voiceSupervisorSurfaceRequest(request.method);
          if (surfaceEvent) {
            controllerRef.current?.notify(surfaceEvent.text);
          }
        }
        return false;
      });

      await client.connect(grant, {
        onFunctionCall: (call) => {
          void controllerRef.current?.onFunctionCall(call.name, call.callId, call.args);
        },
        // Sidecar ASR is kept solely for the spoken stop phrase — no captions.
        // Same matcher as the desktop composer — "stop please", "goodbye",
        // "hey hermes stop" all end the call, not just a bare "stop".
        onUserTranscript: (text) => {
          if (isVoiceStopCommand(text)) {
            stop();
          }
        },
        onStatus: (clientStatus, detail) => {
          if (!clientRef.current) {
            return;
          }
          if (clientStatus === "speaking") {
            setStatus("speaking");
          } else if (clientStatus === "listening") {
            setStatus(controllerRef.current?.consultActive ? "thinking" : "listening");
          } else if (clientStatus === "error") {
            setError(detail ?? "realtime error");
          } else if (clientStatus === "closed") {
            stop();
          }
        },
      });
      if (clientRef.current !== client) {
        return;
      }
      setStatus("listening");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      stop();
    }
  }, [profile, stop]);

  const toggleMute = useCallback(() => {
    setMuted((prev) => {
      const next = !prev;
      clientRef.current?.setMuted(next);
      return next;
    });
  }, []);

  const live = status !== "idle";
  const statusLabel =
    status === "connecting"
      ? t.voiceCall.connecting
      : status === "thinking"
        ? t.voiceCall.working
        : status === "speaking"
          ? t.voiceCall.speaking
          : status === "listening"
            ? t.voiceCall.listening
            : t.voiceCall.title;

  return (
    <div className="px-2 py-2">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => (live ? stop() : void start())}
          className={cn(
            "flex items-center gap-1.5 rounded border px-2 py-1 text-xs transition-colors",
            live
              ? "border-red-500/50 text-red-400 hover:bg-red-500/10"
              : "border-current/20 hover:bg-current/10",
          )}
          title={live ? t.voiceCall.end : t.voiceCall.start}
        >
          {live ? <Square className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
          <span>{live ? t.voiceCall.end : t.voiceCall.start}</span>
        </button>
        {live && (
          <button
            type="button"
            onClick={toggleMute}
            className="rounded border border-current/20 p-1 hover:bg-current/10"
            title={muted ? t.voiceCall.unmute : t.voiceCall.mute}
          >
            {muted ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
          </button>
        )}
        <span
          className={cn(
            "ml-auto text-[0.65rem] uppercase tracking-wide opacity-70",
            status === "speaking" && "text-emerald-400",
            status === "thinking" && "text-amber-400",
          )}
        >
          {statusLabel}
        </span>
      </div>
      {error && <div className="mt-1 text-xs text-red-400">{error}</div>}
    </div>
  );
}
