/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type JarvisOverview } from "@/lib/api";
import "./jarvis-dashboard.css";
import {
  Activity,
  Bot,
  Box,
  CalendarDays,
  Check,
  ChevronRight,
  CircleGauge,
  Clock3,
  Cpu,
  Database,
  FileStack,
  FolderKanban,
  HardDrive,
  LayoutDashboard,
  Mail,
  MemoryStick,
  MessageSquareText,
  Network,
  Orbit,
  Radio,
  Settings2,
  ShieldCheck,
  Sparkles,
  UsersRound,
  Workflow,
  Zap,
} from "lucide-react";

const serviceNodes = [
  { id: "agents", label: "Agents", meta: "6 active", Icon: UsersRound, x: 50, y: 18 },
  { id: "messaging", label: "Messaging", meta: "12 channels", Icon: MessageSquareText, x: 25, y: 34 },
  { id: "automations", label: "Automations", meta: "28 flows", Icon: Workflow, x: 75, y: 34 },
  { id: "calendar", label: "Calendar", meta: "3 upcoming", Icon: CalendarDays, x: 25, y: 68 },
  { id: "files", label: "Files", meta: "42 recent", Icon: FileStack, x: 75, y: 68 },
  { id: "vault", label: "Vault", meta: "12.8k notes", Icon: Database, x: 50, y: 84 },
];

const agentOps = [
  { name: "Builder", task: "Rendering main dashboard", load: 86, tone: "green" },
  { name: "Researcher", task: "Reviewing voice mapping", load: 68, tone: "green" },
  { name: "Planner", task: "Sequencing agent hierarchy", load: 54, tone: "cyan" },
  { name: "Memory", task: "Indexing Obsidian context", load: 47, tone: "cyan" },
  { name: "Monitor", task: "Watching service health", load: 32, tone: "cyan" },
  { name: "Messenger", task: "Waiting for dispatch", load: 12, tone: "muted" },
];

const events = [
  ["10:42:08", "Memory vault synchronized", "green"],
  ["10:41:52", "Builder opened design context", "cyan"],
  ["10:40:16", "Voice core channel connected", "cyan"],
  ["10:38:03", "Approval requested: deployment", "amber"],
];

function formatCount(value, fallback = "0") {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : fallback;
}

function formatPercent(value, fallback = 0) {
  const number = Number(value);
  return Math.max(0, Math.min(100, Number.isFinite(number) ? Math.round(number) : fallback));
}

function overviewStatusTone(status) {
  if (status === "available" || status === "ok" || status === "healthy") return "green";
  if (status === "setup_needed" || status === "unknown") return "amber";
  return "red";
}

function liveServiceNodes(overview) {
  const agentStatus = overview?.agent_status;
  const memory = overview?.memory_vault?.obsidian;
  const cron = overview?.service_health?.cron;
  return serviceNodes.map((node) => {
    if (node.id === "agents") return { ...node, meta: `${formatCount(agentStatus?.active_agents)} active` };
    if (node.id === "messaging") return { ...node, meta: `${formatCount(agentStatus?.connected_platforms)}/${formatCount(agentStatus?.configured_platforms)} connected` };
    if (node.id === "automations") return { ...node, meta: `${formatCount(cron?.enabled)} enabled` };
    if (node.id === "files") return { ...node, meta: `${formatCount(memory?.recent_notes?.length)} recent` };
    if (node.id === "vault") return { ...node, meta: `${formatCount(memory?.note_count)} notes` };
    return node;
  });
}

function liveEvents(overview) {
  if (!overview) return events;
  const generated = overview.generated_at ? new Date(overview.generated_at) : null;
  const stamp = generated && !Number.isNaN(generated.getTime()) ? generated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "Live";
  const memory = overview.memory_vault?.obsidian;
  return [
    [stamp, memory?.message || "Jarvis overview synchronized", overviewStatusTone(memory?.status)],
    [stamp, `${formatCount(overview.todos?.length)} owner tasks loaded`, overview.todos?.length ? "amber" : "green"],
    [stamp, `${formatCount(overview.service_health?.cron?.enabled)} automations enabled`, "cyan"],
    [stamp, `${formatCount(overview.agent_status?.active_sessions)} active sessions`, "cyan"],
  ];
}

function liveAgentOps(overview) {
  const profiles = overview?.agent_status?.profiles || [];
  if (!profiles.length) return agentOps;
  return profiles.slice(0, 6).map((profile) => {
    const open = Number(profile.open_count || 0);
    const blocked = Number(profile.blocked_count || 0);
    const load = Math.max(8, Math.min(96, open * 12 + blocked * 18));
    return {
      name: profile.name,
      task: profile.needs_attention
        ? `${formatCount(blocked)} blocked / ${formatCount(open)} open`
        : profile.role || profile.state || "Monitoring",
      load,
      tone: profile.needs_attention ? "amber" : open ? "green" : "cyan",
    };
  });
}

function voiceEnvelope(time) {
  const seconds = time * 0.001;
  const phrase = Math.max(0, Math.sin(seconds * 0.92) * 0.72 + Math.sin(seconds * 2.3) * 0.22);
  const syllables = Math.abs(Math.sin(seconds * 7.1) * 0.62 + Math.sin(seconds * 11.8) * 0.25);
  const amplitude = Math.min(1, phrase * (0.42 + syllables));
  return {
    amplitude,
    low: Math.min(1, amplitude * 1.14 + Math.abs(Math.sin(seconds * 3.1)) * 0.12),
    mid: Math.min(1, amplitude * 0.9 + Math.abs(Math.sin(seconds * 8.7)) * 0.18),
    high: Math.min(1, amplitude * 0.68 + Math.abs(Math.sin(seconds * 15.3)) * 0.16),
  };
}

function VoiceCoreField({ speaking, activeNode, onSelectNode, onToggleSpeaking, nodes = serviceNodes }) {
  const canvasRef = useRef(null);
  const stageRef = useRef(null);
  const externalFrame = useRef(null);

  useEffect(() => {
    const handleFrame = (event) => {
      const frame = event.detail || {};
      externalFrame.current = {
        amplitude: Math.max(0, Math.min(1, Number(frame.amplitude) || 0)),
        low: Math.max(0, Math.min(1, Number(frame.low) || 0)),
        mid: Math.max(0, Math.min(1, Number(frame.mid) || 0)),
        high: Math.max(0, Math.min(1, Number(frame.high) || 0)),
        receivedAt: performance.now(),
      };
    };
    window.addEventListener("jarvis:voice-frame", handleFrame);
    return () => window.removeEventListener("jarvis:voice-frame", handleFrame);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return undefined;

    const context = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let ratio = 1;
    let frameId = 0;
    let lastDraw = 0;

    const resize = () => {
      const bounds = stage.getBoundingClientRect();
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, bounds.width);
      height = Math.max(1, bounds.height);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(stage);
    resize();

    const drawArc = (cx, cy, radius, start, length, alpha, lineWidth = 1) => {
      context.beginPath();
      context.arc(cx, cy, radius, start, start + length);
      context.strokeStyle = `rgba(87, 218, 255, ${alpha})`;
      context.lineWidth = lineWidth;
      context.stroke();
    };

    const draw = (time) => {
      if (time - lastDraw < 32) {
        frameId = window.requestAnimationFrame(draw);
        return;
      }
      lastDraw = time;
      context.clearRect(0, 0, width, height);

      const center = { x: width * 0.5, y: height * 0.475 };
      const base = Math.min(width, height);
      const realFrame = externalFrame.current;
      const realIsFresh = realFrame && time - realFrame.receivedAt < 260;
      const signal = realIsFresh
        ? realFrame
        : speaking && !reduceMotion
          ? voiceEnvelope(time)
          : { amplitude: 0.035, low: 0.03, mid: 0.025, high: 0.02 };

      const amplitude = signal.amplitude;
      const low = signal.low;
      const mid = signal.mid;
      const high = signal.high;

      context.save();
      context.strokeStyle = "rgba(83, 206, 242, .09)";
      context.lineWidth = 1;
      for (let ring = 0; ring < 5; ring += 1) {
        context.beginPath();
        context.ellipse(
          center.x,
          center.y,
          base * (0.19 + ring * 0.078),
          base * (0.14 + ring * 0.055),
          0,
          0,
          Math.PI * 2,
        );
        context.stroke();
      }

      nodes.forEach((node, index) => {
        const point = { x: width * node.x * 0.01, y: height * node.y * 0.01 };
        const selected = node.id === activeNode;
        const curve = (index % 2 ? -1 : 1) * height * 0.055;
        context.beginPath();
        context.moveTo(center.x, center.y);
        context.bezierCurveTo(
          center.x + (point.x - center.x) * 0.3,
          center.y + curve,
          center.x + (point.x - center.x) * 0.72,
          point.y - curve,
          point.x,
          point.y,
        );
        context.strokeStyle = selected ? "rgba(144, 236, 255, .58)" : "rgba(71, 194, 231, .16)";
        context.lineWidth = selected ? 1.6 : 0.85;
        context.stroke();

        const packet = reduceMotion ? 0.54 : (time * 0.00005 + index * 0.158) % 1;
        const inv = 1 - packet;
        const p1x = center.x + (point.x - center.x) * 0.3;
        const p1y = center.y + curve;
        const p2x = center.x + (point.x - center.x) * 0.72;
        const p2y = point.y - curve;
        const px =
          inv ** 3 * center.x +
          3 * inv ** 2 * packet * p1x +
          3 * inv * packet ** 2 * p2x +
          packet ** 3 * point.x;
        const py =
          inv ** 3 * center.y +
          3 * inv ** 2 * packet * p1y +
          3 * inv * packet ** 2 * p2y +
          packet ** 3 * point.y;
        context.beginPath();
        context.arc(px, py, selected ? 2.7 : 1.5, 0, Math.PI * 2);
        context.fillStyle = selected ? "rgba(198, 246, 255, .92)" : "rgba(88, 217, 255, .52)";
        context.shadowColor = "#56d8ff";
        context.shadowBlur = selected ? 12 : 5;
        context.fill();
        context.shadowBlur = 0;
      });
      context.restore();

      context.save();
      context.shadowColor = "#50d8ff";
      context.shadowBlur = 9 + amplitude * 16;
      for (let orbit = 0; orbit < 3; orbit += 1) {
        context.beginPath();
        context.ellipse(
          center.x,
          center.y,
          base * (0.28 + orbit * 0.085),
          base * (0.19 + orbit * 0.055),
          orbit === 1 ? 0.03 : -0.03,
          0.24 + orbit * 0.46,
          Math.PI * (1.62 + orbit * 0.08),
        );
        context.strokeStyle = `rgba(75, 210, 250, ${0.17 + mid * 0.15})`;
        context.lineWidth = orbit === 1 ? 1.35 : 0.9;
        context.stroke();
      }
      context.restore();

      const glow = context.createRadialGradient(
        center.x,
        center.y,
        0,
        center.x,
        center.y,
        base * (0.19 + amplitude * 0.032),
      );
      glow.addColorStop(0, `rgba(223, 251, 255, ${0.92 + amplitude * 0.08})`);
      glow.addColorStop(0.12, `rgba(100, 224, 255, ${0.68 + amplitude * 0.2})`);
      glow.addColorStop(0.45, `rgba(34, 158, 215, ${0.18 + low * 0.22})`);
      glow.addColorStop(1, "rgba(10, 74, 104, 0)");
      context.fillStyle = glow;
      context.beginPath();
      context.arc(center.x, center.y, base * (0.19 + amplitude * 0.032), 0, Math.PI * 2);
      context.fill();

      context.save();
      context.translate(center.x, center.y);
      for (let ring = 0; ring < 7; ring += 1) {
        const radius = base * (0.085 + ring * 0.022) + low * (ring + 1) * 1.8;
        const segments = 8 + ring * 2;
        for (let segment = 0; segment < segments; segment += 1) {
          if ((segment + ring) % 4 === 0) continue;
          const start =
            (Math.PI * 2 * segment) / segments +
            (reduceMotion ? 0 : time * 0.000035 * (ring % 2 ? -1 : 1));
          const length = (Math.PI * 2) / segments * (0.5 + mid * 0.25);
          drawArc(0, 0, radius, start, length, 0.13 + ring * 0.025 + mid * 0.16, ring % 3 === 0 ? 1.4 : 0.8);
        }
      }
      context.restore();

      context.save();
      const coreRadius = base * (0.058 + low * 0.012);
      context.shadowColor = "#8beaff";
      context.shadowBlur = 22 + amplitude * 26;
      const core = context.createRadialGradient(
        center.x - coreRadius * 0.22,
        center.y - coreRadius * 0.26,
        1,
        center.x,
        center.y,
        coreRadius,
      );
      core.addColorStop(0, "#f5feff");
      core.addColorStop(0.2, "#b7f4ff");
      core.addColorStop(0.62, "#46caef");
      core.addColorStop(1, "rgba(9, 92, 133, .45)");
      context.fillStyle = core;
      context.beginPath();
      context.arc(center.x, center.y, coreRadius, 0, Math.PI * 2);
      context.fill();
      context.restore();

      const particleCount = 78;
      for (let index = 0; index < particleCount; index += 1) {
        const angle = index * 2.39996 + (reduceMotion ? 0 : time * 0.000015 * (index % 2 ? -1 : 1));
        const band = (index % 9) / 9;
        const radius = base * (0.086 + band * 0.145) + high * (10 + (index % 7) * 2.4);
        const x = center.x + Math.cos(angle) * radius;
        const y = center.y + Math.sin(angle) * radius * 0.72;
        context.beginPath();
        context.arc(x, y, index % 11 === 0 ? 2.1 : 1.05, 0, Math.PI * 2);
        context.fillStyle = `rgba(129, 231, 255, ${0.16 + high * 0.48})`;
        context.fill();
      }

      frameId = window.requestAnimationFrame(draw);
    };

    frameId = window.requestAnimationFrame(draw);
    return () => {
      window.cancelAnimationFrame(frameId);
      observer.disconnect();
    };
  }, [activeNode, speaking, nodes]);

  return (
    <div className="core-stage" ref={stageRef}>
      <canvas ref={canvasRef} aria-hidden="true" />
      <button
        className={`voice-core ${speaking ? "is-speaking" : ""}`}
        type="button"
        onClick={onToggleSpeaking}
        aria-pressed={speaking}
        aria-label="Toggle voice-reactive core demonstration"
      >
        <span>Hermes Core</span>
        <strong>{speaking ? "Speaking" : "Active"}</strong>
        <small>{speaking ? "Voice synchronized" : "Voice channel ready"}</small>
      </button>

      {nodes.map(({ id, label, meta, Icon, x, y }) => (
        <button
          key={id}
          type="button"
          className={`service-node ${activeNode === id ? "is-active" : ""}`}
          style={{ left: `${x}%`, top: `${y}%` }}
          onClick={() => onSelectNode(id)}
          aria-pressed={activeNode === id}
        >
          <span>
            <Icon aria-hidden="true" />
          </span>
          <strong>{label}</strong>
          <small>{meta}</small>
        </button>
      ))}
    </div>
  );
}

function DashboardNav() {
  const items = [
    ["Dashboard", LayoutDashboard, "/jarvis"],
    ["Memory", Network, "/jarvis/memory"],
    ["Agents", UsersRound, "/jarvis/agents"],
    ["Automations", Zap, "#automations"],
    ["Vault", Box, "/jarvis/memory"],
    ["Settings", Settings2, "#settings"],
  ];

  return (
    <nav className="dashboard-nav" aria-label="Primary navigation">
      {items.map(([label, Icon, href]) => (
        <Link key={label} to={href} className={label === "Dashboard" ? "is-current" : ""}>
          <Icon aria-hidden="true" />
          <span>{label}</span>
        </Link>
      ))}
    </nav>
  );
}

export default function JarvisPage() {
  const [speaking, setSpeaking] = useState(false);
  const [activeNode, setActiveNode] = useState("agents");
  const [approvalResolved, setApprovalResolved] = useState(false);
  const [synced, setSynced] = useState(false);
  const [overview, setOverview] = useState<JarvisOverview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadOverview = async () => {
      try {
        const nextOverview = await api.getJarvisOverview();
        if (!cancelled) {
          setOverview(nextOverview);
          setOverviewError(null);
        }
      } catch (error) {
        if (!cancelled) setOverviewError(error instanceof Error ? error.message : "Overview unavailable");
      }
    };
    void loadOverview();
    const interval = window.setInterval(loadOverview, Math.max(15, overview?.refresh_after_seconds || 15) * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [overview?.refresh_after_seconds]);

  const memoryVault = overview?.memory_vault?.obsidian;
  const serviceHealth = overview?.service_health;
  const agentStatus = overview?.agent_status;
  const diskPercent = formatPercent(serviceHealth?.system?.disk_percent, memoryVault?.configured ? 45 : 0);
  const memoryStatus = overviewError ? "Unavailable" : memoryVault?.status === "setup_needed" ? "Setup needed" : memoryVault?.status === "available" ? "Healthy" : "Loading";
  const memoryTone = overviewError || memoryVault?.status === "unavailable" ? "red" : memoryVault?.status === "setup_needed" ? "amber" : "green";
  const activeAgentCount = agentStatus?.active_agents ?? 0;
  const queuedTaskCount = overview?.todos?.length ?? 0;
  const approvalCount = (overview?.todos || []).filter((task) => task.attention_action || task.block_kind === "needs_input").length;
  const liveNodes = liveServiceNodes(overview);
  const dashboardEvents = liveEvents(overview);
  const dashboardAgents = liveAgentOps(overview);

  const sync = () => {
    setSynced(true);
    window.setTimeout(() => setSynced(false), 1000);
  };

  return (
    <div className="dashboard-shell">
      <header className="dashboard-topbar">
        <div className="dashboard-brand">
          <Orbit aria-hidden="true" />
          <span>Hermes OS</span>
          <i>/</i>
          <strong>Jarvis</strong>
        </div>
        <div className="dashboard-security">
          <span>
            <ShieldCheck aria-hidden="true" />
            Local Secure
          </span>
          <span>
            <i aria-hidden="true" />
            Live
          </span>
        </div>
        <div className="dashboard-time">
          <button type="button" onClick={sync}>
            <CircleGauge className={synced ? "is-spinning" : ""} aria-hidden="true" />
            {synced ? "Syncing" : "Synced"}
          </button>
          <time dateTime={new Date().toISOString()}>{new Date().toLocaleString([], { month: "short", day: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" })}</time>
        </div>
      </header>

      <main className="dashboard-grid">
        <aside className="agent-ops-panel hud-panel" id="agent-ops">
          <div className="panel-heading">
            <span>
              <UsersRound aria-hidden="true" />
              Agent Ops
            </span>
            <b>{formatCount(activeAgentCount)} active</b>
          </div>
          <div className="agent-summary">
            <div>
              <strong>{formatCount(activeAgentCount)}</strong>
              <span>Tasks running</span>
            </div>
            <div>
              <strong>{formatCount(queuedTaskCount)}</strong>
              <span>Queued</span>
            </div>
            <div>
              <strong className="green">98%</strong>
              <span>Success</span>
            </div>
          </div>
          <div className="agent-list">
            {dashboardAgents.map((agent) => (
              <button
                key={agent.name}
                type="button"
                className={agent.name.toLowerCase() === activeNode ? "is-active" : ""}
                onClick={() => setActiveNode("agents")}
              >
                <span className={`agent-status agent-status--${agent.tone}`} aria-hidden="true" />
                <span>
                  <strong>{agent.name}</strong>
                  <small>{agent.task}</small>
                </span>
                <em>{agent.load}%</em>
                <i style={{ "--load": `${agent.load}%` }} aria-hidden="true" />
              </button>
            ))}
          </div>
          <Link className="panel-link" to="/jarvis/agents">
            View hierarchy
            <ChevronRight aria-hidden="true" />
          </Link>
        </aside>

        <section className="command-center" aria-label="Jarvis command center">
          <div className="command-center__label">
            <span>
              <Radio aria-hidden="true" />
              Neural command field
            </span>
            <small>{speaking ? "Speech frame input active" : "Core standing by"}</small>
          </div>
          <VoiceCoreField
            speaking={speaking}
            activeNode={activeNode}
            nodes={liveNodes}
            onSelectNode={setActiveNode}
            onToggleSpeaking={() => setSpeaking((value) => !value)}
          />
        </section>

        <aside className="vault-panel hud-panel">
          <div className="panel-heading">
            <span>
              <Database aria-hidden="true" />
              Data Vault
            </span>
            <b className={memoryTone}>{memoryStatus}</b>
          </div>
          <div className="vault-capacity">
            <span>
              <HardDrive aria-hidden="true" />
            </span>
            <div>
              <strong>{formatCount(memoryVault?.note_count)}</strong>
              <small>{memoryVault?.configured ? "Obsidian notes indexed" : "Vault not configured"}</small>
            </div>
            <em>{diskPercent}%</em>
          </div>
          <div className="vault-meter" aria-label={`Vault storage ${diskPercent} percent`}>
            <i style={{ width: `${diskPercent}%` }} />
          </div>
          <div className="vault-stats">
            <div>
              <FolderKanban aria-hidden="true" />
              <span>Projects</span>
              <strong>{formatCount(memoryVault?.product_note_count)}</strong>
            </div>
            <div>
              <MemoryStick aria-hidden="true" />
              <span>Memories</span>
              <strong>{formatCount(memoryVault?.note_count)}</strong>
            </div>
            <div>
              <Mail aria-hidden="true" />
              <span>Recent</span>
              <strong>{formatCount(memoryVault?.recent_notes?.length)}</strong>
            </div>
            <div>
              <Sparkles aria-hidden="true" />
              <span>Decisions</span>
              <strong>{formatCount(memoryVault?.decision_count)}</strong>
            </div>
          </div>
          <Link className="panel-link" to="/jarvis/memory">
            Open memory atlas
            <ChevronRight aria-hidden="true" />
          </Link>
        </aside>

        <section className="approval-strip hud-panel">
          <div className={approvalResolved ? "approval-icon is-resolved" : "approval-icon"}>
            {approvalResolved ? <Check aria-hidden="true" /> : <Clock3 aria-hidden="true" />}
          </div>
          <div>
            <span>{approvalResolved ? "Approval resolved" : `${formatCount(approvalCount)} approval${approvalCount === 1 ? "" : "s"} waiting`}</span>
            <strong>
              {approvalResolved
                ? "Deployment request acknowledged"
                : overview?.todos?.[0]?.title || "Production deployment requires operator review"}
            </strong>
          </div>
          <button type="button" onClick={() => setApprovalResolved((value) => !value)}>
            {approvalResolved ? "Reopen" : "Review"}
          </button>
          <div className="health-metrics">
            <span>
              <Cpu aria-hidden="true" />
              CPU <strong>{formatPercent(serviceHealth?.system?.cpu_percent, 0)}%</strong>
            </span>
            <span>
              <MemoryStick aria-hidden="true" />
              Memory <strong>{formatPercent(serviceHealth?.system?.memory_percent, 0)}%</strong>
            </span>
            <span>
              <Activity aria-hidden="true" />
              Sessions <strong>{formatCount(agentStatus?.active_sessions)}</strong>
            </span>
            <span>
              <Bot aria-hidden="true" />
              Agents <strong>{formatCount(activeAgentCount)}</strong>
            </span>
          </div>
        </section>

        <section className="event-panel hud-panel">
          <div className="panel-heading">
            <span>
              <Activity aria-hidden="true" />
              System Events
            </span>
            <b>Live feed</b>
          </div>
          <div className="event-list">
            {dashboardEvents.map(([time, label, tone]) => (
              <div key={`${time}-${label}`}>
                <time>{time}</time>
                <i className={`event-dot event-dot--${tone}`} aria-hidden="true" />
                <span>{label}</span>
              </div>
            ))}
          </div>
        </section>
      </main>

      <DashboardNav />
    </div>
  );
}
