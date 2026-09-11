import { useEffect, useState } from "react";
import type { StatusResponse } from "@/lib/api";
import { get } from "../api";
import { useMiniApp } from "../context";
import { formatPlatformName } from "../platform-names";
import type { MiniAppStatusExtra } from "../types";

function formatUptime(startEpoch: number): string {
  const seconds = Math.max(0, Date.now() / 1000 - startEpoch);
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  if (days > 0) return `${days}d ${hours}h`;
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function platformDotColor(state: string): string {
  const s = state.toLowerCase();
  if (s.includes("connect") || s.includes("running")) return "var(--success)";
  if (s.includes("error") || s.includes("fail")) return "var(--destr)";
  return "var(--warning)";
}

const LOG_ITEMS: Array<{ key: string; label: string }> = [
  { key: "agent", label: "Agent Messages" },
  { key: "errors", label: "Errors and Warnings" },
  { key: "gateway", label: "Gateway Logs" },
  { key: "update", label: "Updates" },
];

export function StatusScreen({
  statusExtra,
  onOpenLog,
}: {
  statusExtra: MiniAppStatusExtra;
  onOpenLog: (key: string, label: string) => void;
}) {
  const { isAdmin, gwConnected, gwRestarting, askRestartGateway, askUpdateHermes } = useMiniApp();
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    get<StatusResponse>("/api/status").then(setStatus).catch(() => setStatus(null));
  }, [gwConnected]);

  const gwOk = gwConnected && !gwRestarting;
  const gwLabel = gwRestarting ? "Restarting…" : gwOk ? "Running" : "Stopped";
  const gwDotColor = gwOk ? "var(--success)" : "var(--warning)";

  const platforms = status ? Object.entries(status.gateway_platforms || {}) : [];

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--t3)",
              fontFamily: "var(--mono)",
            }}
          >
            Gateway
          </div>
          {status?.gateway_pid != null && (
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--t3)", whiteSpace: "nowrap" }}>
              pid {status.gateway_pid}
            </div>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 8 }}>
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: 99,
              background: gwDotColor,
              animation: "miniapp-hpulse 2.4s ease-out infinite",
            }}
          />
          <span style={{ fontSize: 21, fontWeight: 650, color: "var(--mid)" }}>{gwLabel}</span>
          {isAdmin && (
            <button
              onClick={askRestartGateway}
              disabled={gwRestarting}
              style={{
                marginLeft: "auto",
                fontFamily: "var(--mono)",
                fontSize: 11,
                letterSpacing: "0.05em",
                padding: "5px 11px",
                borderRadius: 8,
                border: "1px solid var(--line2)",
                background: "transparent",
                cursor: gwRestarting ? "default" : "pointer",
                color: "var(--t2)",
                whiteSpace: "nowrap",
              }}
            >
              Restart
            </button>
          )}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
        <StatCard label="Sessions" value={String(status?.active_sessions ?? "–")} sub="active now" />
        <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "12px 13px" }}>
          <div style={{ fontSize: 9.5, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)" }}>
            Version
          </div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 19, fontWeight: 600, marginTop: 5, color: "var(--mid)" }}>
            {status?.version ?? "–"}
          </div>
          <div style={{ fontSize: 10, color: "var(--t3)", marginTop: 1 }}>{status?.release_date ?? ""}</div>
          {isAdmin && status?.can_update_hermes !== false && (
            <button
              onClick={askUpdateHermes}
              style={{
                width: "100%",
                marginTop: 8,
                fontFamily: "var(--mono)",
                fontSize: 10.5,
                letterSpacing: "0.05em",
                padding: "5px 0",
                borderRadius: 8,
                border: "1px solid var(--line2)",
                background: "transparent",
                cursor: "pointer",
                color: "var(--t2)",
                whiteSpace: "nowrap",
              }}
            >
              Update
            </button>
          )}
        </div>
        {isAdmin ? (
          <StatCard
            label="Uptime"
            value={statusExtra.gateway_start_time != null ? formatUptime(statusExtra.gateway_start_time) : "–"}
            sub={statusExtra.gateway_start_time != null ? "since start" : "unavailable"}
          />
        ) : (
          // Never reference statusExtra.gateway_start_time on this branch at
          // all -- not even in a null-guard -- so a non-admin render can't
          // leak the field regardless of what the backend response
          // contains. A static placeholder, not derived from admin-only
          // data, keeps the 3-up grid layout intact for both tiers.
          <StatCard label="Uptime" value="–" sub="admin only" />
        )}
      </div>

      {platforms.length > 0 && (
        <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, overflow: "hidden" }}>
          <div style={{ padding: "11px 16px 8px", fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)" }}>
            Platforms
          </div>
          {platforms.map(([name, p]) => (
            <div key={name} style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 16px", borderTop: "1px solid var(--line)" }}>
              <span style={{ width: 7, height: 7, borderRadius: 99, background: platformDotColor(p.state) }} />
              <span style={{ fontSize: 13.5, fontWeight: 550, color: "var(--mid)" }}>{formatPlatformName(name)}</span>
              <span style={{ marginLeft: "auto", fontFamily: "var(--mono)", fontSize: 11, color: platformDotColor(p.state), whiteSpace: "nowrap" }}>
                {p.state}
              </span>
            </div>
          ))}
        </div>
      )}

      {isAdmin && (
        <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, overflow: "hidden" }}>
          <div style={{ padding: "11px 16px 8px", fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)" }}>
            Logs
          </div>
          {LOG_ITEMS.map(({ key, label }) => (
            <div
              key={key}
              onClick={() => onOpenLog(key, label)}
              style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 16px", borderTop: "1px solid var(--line)", cursor: "pointer" }}
            >
              <span style={{ fontSize: 13.5, fontWeight: 550, color: "var(--mid)" }}>{label}</span>
              <svg width="7" height="12" viewBox="0 0 7 12" style={{ marginLeft: "auto", flexShrink: 0 }}>
                <path
                  d="M1 1l5 5-5 5"
                  stroke="currentColor"
                  strokeOpacity={0.4}
                  strokeWidth="1.8"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          ))}
        </div>
      )}

      <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--t3)", padding: "0 4px", display: "flex", justifyContent: "space-between" }}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{status?.hermes_home ?? ""}</span>
        <span style={{ whiteSpace: "nowrap" }}>{isAdmin ? "admin" : "member · read-only"}</span>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "12px 13px" }}>
      <div style={{ fontSize: 9.5, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)" }}>
        {label}
      </div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 19, fontWeight: 600, marginTop: 5, color: "var(--mid)" }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--t3)", marginTop: 1 }}>{sub}</div>
    </div>
  );
}
