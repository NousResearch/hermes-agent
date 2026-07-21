import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Command,
  LayoutGrid,
  PanelTop,
  Sparkles,
} from "lucide-react";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";

const sidebarAgents = [
  { name: "Claude", accent: "from-[#ff9e72] to-[#f25d74]", status: "Online" },
  { name: "OpenClaw", accent: "from-[#ff74c7] to-[#c65bff]", status: "Ready" },
  { name: "Hermes", accent: "from-[#7e9bff] to-[#4d7dff]", status: "Online" },
  { name: "Gemini", accent: "from-[#ffb86b] to-[#ff6f7d]", status: "Syncing" },
  { name: "Antigravity", accent: "from-[#9b7cff] to-[#6a4fff]", status: "Ready" },
  { name: "Codex", accent: "from-[#50e3c2] to-[#23b7a2]", status: "Online" },
  { name: "Free Claude Code", accent: "from-[#36d399] to-[#0ea5a6]", status: "Offline" },
] as const;

type StatusCard = {
  label: string;
  value: string;
  detail: string;
  accent?: "amber" | "offline";
};

function buildStatusCards(status: StatusResponse | null): StatusCard[] {
  const gatewayPlatforms = Object.values(status?.gateway_platforms ?? {});
  const onlinePlatforms = gatewayPlatforms.filter((p) => p.state === "running" || p.state === "online").length;
  const totalPlatforms = gatewayPlatforms.length;
  const authLabel = status?.auth_required ? "Gated" : "Loopback";
  const authDetail = status?.auth_providers?.length
    ? status.auth_providers.join(", ")
    : status?.auth_required
      ? "No auth providers"
      : "Auth not required";

  return [
    {
      label: "Version",
      value: status?.version ?? "—",
      detail: status ? `Release ${status.release_date}` : "Waiting for live status",
    },
    {
      label: "Active sessions",
      value: status ? String(status.active_sessions) : "—",
      detail: status ? `Config v${status.config_version}` : "Waiting for live status",
    },
    {
      label: "Gateway",
      value: status?.gateway_running ? "Running" : status?.gateway_state ?? "Offline",
      detail: status?.gateway_exit_reason ?? "Live gateway state",
      accent: status?.gateway_running ? undefined : "offline",
    },
    {
      label: "Platforms",
      value: status ? `${onlinePlatforms}/${totalPlatforms}` : "—",
      detail: status
        ? `${Object.keys(status.gateway_platforms).length} known integrations`
        : "Waiting for live status",
    },
    {
      label: "Config",
      value: status ? `${status.config_version}/${status.latest_config_version}` : "—",
      detail: status?.config_path ?? "Live config path",
      accent: status && status.config_version !== status.latest_config_version ? "amber" : undefined,
    },
    {
      label: "Auth",
      value: authLabel,
      detail: authDetail,
      accent: status?.auth_required ? "amber" : undefined,
    },
  ];
}

const agentCards = [
  {
    name: "Claude",
    badge: "C",
    gradient: "from-[#ff9e72] to-[#f25d74]",
    status: "Online",
    description: "Direct streaming line to Claude Code. Full tool use, MCPs, plugins.",
    footer: "Live",
  },
  {
    name: "OpenClaw",
    badge: "O",
    gradient: "from-[#ff74c7] to-[#c65bff]",
    status: "Online",
    description: "Local agent gateway. Chat one-shot or open the control room.",
    footer: "Connected",
  },
  {
    name: "Hermes",
    badge: "H",
    gradient: "from-[#7e9bff] to-[#4d7dff]",
    status: "Online",
    description: "News research agent. Tool calls, kanban, skills, plugins.",
    footer: "Active",
  },
] as const;

const lowerPanels = [
  {
    name: "Memory",
    badge: "M",
    gradient: "from-[#8b5cf6] to-[#5b21b6]",
    status: "Synced",
    description: "CONTEXT.md, agent-comms-protocol, and recent operator decisions.",
  },
  {
    name: "Signals",
    badge: "S",
    gradient: "from-[#22c55e] to-[#0f766e]",
    status: "Listening",
    description: "Heartbeat checks, blockers, outbound task pings, and system alerts.",
  },
] as const;

function ShellCard({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={
        "rounded-[22px] border border-white/6 bg-[linear-gradient(180deg,rgba(40,30,52,0.98),rgba(24,17,32,0.98))] shadow-[0_18px_34px_rgba(0,0,0,0.32)] " +
        (className ?? "")
      }
    >
      {children}
    </div>
  );
}

function MissionControlShell({ tight }: { tight: boolean }) {
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api
        .getStatus()
        .then((resp) => {
          if (!cancelled) setStatus(resp);
        })
        .catch(() => {
          if (!cancelled) setStatus(null);
        });
    };

    load();
    const id = window.setInterval(load, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const statusCards = useMemo(() => buildStatusCards(status), [status]);
  const platformEntries = Object.entries(status?.gateway_platforms ?? {});
  const platformSummary =
    status && platformEntries.length > 0
      ? `${platformEntries.filter(([, p]) => p.state === "running" || p.state === "online").length}/${platformEntries.length} online`
      : "Waiting for live status";

  return (
    <div className="relative min-h-dvh overflow-hidden bg-[#0f0914] text-[#f5eefb]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_68%_14%,rgba(168,140,255,0.20),transparent_32%),radial-gradient(circle_at_18%_84%,rgba(124,240,216,0.10),transparent_28%),linear-gradient(180deg,#0f0914_0%,#120b18_100%)]" />
      <div className="absolute inset-0 opacity-[0.15] [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:32px_32px]" />

      <div
        className={
          "relative mx-auto grid min-h-dvh w-full max-w-[1550px] overflow-hidden border border-white/5 bg-[linear-gradient(180deg,rgba(18,11,24,0.96),rgba(13,8,16,0.98))] shadow-[0_30px_90px_rgba(0,0,0,0.42)] " +
          (tight
            ? "grid-cols-[272px_minmax(0,1fr)] rounded-none lg:rounded-[28px]"
            : "grid-cols-[290px_minmax(0,1fr)] rounded-none lg:m-4 lg:min-h-[calc(100dvh-2rem)] lg:rounded-[28px]")
        }
      >
        <aside className="flex flex-col gap-5 border-r border-white/5 bg-[linear-gradient(180deg,rgba(24,16,31,0.98),rgba(19,12,24,0.98))] px-4 py-5 lg:px-5">
          <div className="space-y-2 px-1 pt-1">
            <div className="text-[10px] uppercase tracking-[0.28em] text-white/40">Local · Bangkok</div>
            <div className="text-[30px] font-semibold leading-none tracking-[-0.05em]">Agentic OS</div>
          </div>

          <div className="space-y-2">
            <div className="px-2 text-[10px] uppercase tracking-[0.24em] text-white/38">Workspace</div>
            <div className="rounded-[14px] bg-[linear-gradient(90deg,rgba(168,140,255,0.16),rgba(255,255,255,0.02))] px-3 py-3 ring-1 ring-[rgba(168,140,255,0.12)]">
              <div className="flex items-center gap-3">
                <span className="grid h-8 w-8 place-items-center rounded-xl bg-white/8 text-sm text-white/90 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]">
                  <LayoutGrid className="h-4 w-4" />
                </span>
                <div className="min-w-0">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/45">Selected</div>
                  <div className="mt-0.5 truncate text-[15px] font-semibold text-white">Mission Control</div>
                </div>
                <span className="ml-auto h-2.5 w-2.5 rounded-full bg-[#7cf0d8] shadow-[0_0_0_4px_rgba(124,240,216,0.09),0_0_16px_rgba(124,240,216,0.8)]" />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="px-2 text-[10px] uppercase tracking-[0.24em] text-white/38">Agents</div>
            <div className="space-y-1">
              {sidebarAgents.map((agent, index) => (
                <div
                  key={agent.name}
                  className={
                    "flex min-h-11 items-center gap-3 rounded-[12px] px-3 text-white/70 transition-colors hover:bg-white/5 hover:text-white " +
                    (index === 0 ? "bg-white/5 text-white" : "")
                  }
                >
                  <span
                    className={
                      "grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-b text-[11px] font-semibold text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)] " +
                      agent.accent
                    }
                  >
                    {agent.name[0]}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-[14px] font-medium tracking-[-0.01em]">
                    {agent.name}
                  </span>
                  <span
                    className={
                      "h-2.5 w-2.5 rounded-full " +
                      (agent.status === "Offline"
                        ? "bg-[#ff758f] shadow-[0_0_0_4px_rgba(255,117,143,0.1)]"
                        : "bg-[#7cf0d8] shadow-[0_0_0_4px_rgba(124,240,216,0.08)]")
                    }
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[18px] border border-white/6 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))] p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[10px] uppercase tracking-[0.24em] text-white/38">Live status</div>
                <div className="mt-2 text-[14px] font-semibold text-white">{status?.gateway_running ? "Gateway online" : "Gateway offline"}</div>
              </div>
              <span className={"h-2.5 w-2.5 rounded-full " + (status?.gateway_running ? "bg-[#7cf0d8] shadow-[0_0_0_4px_rgba(124,240,216,0.09)]" : "bg-[#ff758f] shadow-[0_0_0_4px_rgba(255,117,143,0.1)]")} />
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 text-[12px] text-white/55">
              <div>
                <div className="uppercase tracking-[0.18em] text-white/30">Version</div>
                <div className="mt-1 text-[13px] text-white/80">{status?.version ?? "—"}</div>
              </div>
              <div>
                <div className="uppercase tracking-[0.18em] text-white/30">Sessions</div>
                <div className="mt-1 text-[13px] text-white/80">{status?.active_sessions ?? "—"}</div>
              </div>
              <div>
                <div className="uppercase tracking-[0.18em] text-white/30">Platforms</div>
                <div className="mt-1 text-[13px] text-white/80">{platformSummary}</div>
              </div>
              <div>
                <div className="uppercase tracking-[0.18em] text-white/30">Auth</div>
                <div className="mt-1 text-[13px] text-white/80">{status?.auth_required ? "Gated" : "Loopback"}</div>
              </div>
            </div>
          </div>

          <div className="mt-auto rounded-[18px] border border-white/6 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))] p-3">
            <div className="flex items-center gap-3">
              <div className="relative h-16 w-16 overflow-hidden rounded-[18px] bg-[linear-gradient(180deg,#ff987f,#f34e78)] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12),0_16px_28px_rgba(0,0,0,0.24)]">
                <div className="absolute left-6 top-4 h-4 w-4 rounded-full bg-white/30" />
                <div className="absolute bottom-4 left-4 h-4 w-9 rounded-full bg-black/12" />
              </div>
              <div>
                <div className="text-[14px] font-semibold leading-none">Richard</div>
                <div className="mt-2 text-[12px] text-white/45">Operator · Mission Control</div>
              </div>
            </div>
          </div>
        </aside>

        <main className={tight ? "px-6 py-6 lg:px-8 lg:py-7" : "px-6 py-6 lg:px-8 lg:py-8"}>
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-white/35">
                <span className="text-white/80">I</span>
                Mission Control
              </div>
              <h2 className={tight ? "text-[clamp(44px,5.2vw,78px)]" : "text-[clamp(48px,5.6vw,88px)]"}>
                Mission Control
              </h2>
              <p className="mt-3 max-w-2xl text-[16px] text-white/64">
                Status of every agent, every memory, every signal.
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] uppercase tracking-[0.22em] text-white/35">
                <span>07:52</span>
                <span>·</span>
                <span>Local</span>
                <span>·</span>
                <span>Bangkok</span>
              </div>
            </div>

            <div className="flex shrink-0 flex-wrap items-center gap-2 pt-2">
              <button className="inline-flex h-8 items-center gap-2 rounded-xl border border-white/6 bg-white/4 px-3 text-[12px] text-white/88 shadow-[0_10px_20px_rgba(0,0,0,0.12)] transition-colors hover:bg-white/6">
                <Command className="h-3.5 w-3.5" />
                <span className="text-white/45">Command palette</span>
              </button>
              <button className="inline-flex h-8 items-center gap-2 rounded-xl border border-white/6 bg-[linear-gradient(180deg,rgba(168,140,255,0.18),rgba(255,255,255,0.03))] px-3 text-[12px] font-semibold text-white shadow-[0_10px_20px_rgba(0,0,0,0.12)] transition-colors hover:bg-white/6">
                <PanelTop className="h-3.5 w-3.5" />
                All systems
              </button>
            </div>
          </div>

          <section className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-6">
            {statusCards.map((card) => (
              <ShellCard key={card.label} className="min-h-[94px] p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-white/60">
                    {card.label}
                  </div>
                  <span
                    className={
                      "h-2 w-2 rounded-full " +
                      (card.accent === "amber"
                        ? "bg-[#f4c76e] shadow-[0_0_0_4px_rgba(244,199,110,0.1)]"
                        : card.accent === "offline"
                          ? "bg-[#ff758f] shadow-[0_0_0_4px_rgba(255,117,143,0.1)]"
                          : "bg-[#7cf0d8] shadow-[0_0_0_4px_rgba(124,240,216,0.08)]")
                    }
                  />
                </div>
                <div className="mt-5 text-[22px] font-semibold leading-none tracking-[-0.03em]">
                  {card.value}
                </div>
                <div className="mt-3 text-[12px] leading-relaxed text-white/42">{card.detail}</div>
              </ShellCard>
            ))}
          </section>

          <div className="my-6 flex items-center gap-4 opacity-90">
            <div className="h-px flex-1 bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent)]" />
            <span className="text-[18px] text-[#ffd879]">✦</span>
            <div className="h-px flex-1 bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent)]" />
          </div>

          <div className="mb-3 flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-white/35">
            <Sparkles className="h-3.5 w-3.5 text-white/70" />
            Agents · click to open control room
          </div>

          <section className="grid gap-4 lg:grid-cols-3">
            {agentCards.map((agent) => (
              <ShellCard key={agent.name} className="min-h-[168px] p-[18px]">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className={"grid h-9 w-9 place-items-center rounded-[13px] bg-gradient-to-b text-sm font-semibold text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)] " + agent.gradient}>
                      {agent.badge}
                    </div>
                    <h3 className="text-[20px] font-semibold tracking-[-0.03em] text-white">{agent.name}</h3>
                  </div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-white/65">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#7cf0d8] shadow-[0_0_0_4px_rgba(124,240,216,0.08)]" />
                    {agent.status}
                  </span>
                </div>
                <p className="mt-4 max-w-[26ch] text-[14px] leading-[1.55] text-white/65">{agent.description}</p>
                <div className="mt-auto flex items-center justify-between pt-6 text-[12px] text-white/40">
                  <span className="inline-flex items-center gap-2">
                    <span className="text-[#7cf0d8]">●</span>
                    {agent.footer}
                  </span>
                  <span>Open room ↗</span>
                </div>
              </ShellCard>
            ))}
          </section>

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            {lowerPanels.map((panel) => (
              <ShellCard key={panel.name} className="min-h-[132px] p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className={"grid h-9 w-9 place-items-center rounded-[13px] bg-gradient-to-b text-sm font-semibold text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)] " + panel.gradient}>
                      {panel.badge}
                    </div>
                    <h3 className="text-[18px] font-semibold tracking-[-0.03em] text-white">{panel.name}</h3>
                  </div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-white/65">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#7cf0d8] shadow-[0_0_0_4px_rgba(124,240,216,0.08)]" />
                    {panel.status}
                  </span>
                </div>
                <p className="mt-4 max-w-[38ch] text-[14px] leading-[1.55] text-white/65">{panel.description}</p>
              </ShellCard>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}

export function MissionControlPage() {
  return <MissionControlShell tight={false} />;
}

export function MissionControlPageV2() {
  return <MissionControlShell tight={true} />;
}

export default MissionControlPageV2;
