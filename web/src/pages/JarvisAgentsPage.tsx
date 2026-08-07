/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "./jarvis-agents-city.css";
import {
  Activity,
  Bot,
  Box,
  Building2,
  CalendarDays,
  ChevronRight,
  Clock3,
  Cpu,
  Database,
  GitBranch,
  HardDrive,
  LayoutDashboard,
  MessageSquareText,
  Network,
  Orbit,
  Search,
  Settings2,
  ShieldCheck,
  UsersRound,
  Workflow,
  Zap,
} from "lucide-react";

const agents = [
  {
    id: "benson",
    name: "Benson",
    role: "Chief Agent",
    description: "Executive Orchestrator",
    parent: "Hermes Core / Jarvis",
    district: "core",
    status: "active",
    load: 92,
    icon: Bot,
    x: 50,
    y: 31,
    task: "Coordinating product and personal operations",
    delegation: "Delegated voice-frame validation to Researcher",
    memory: "Executive Context",
    memorySize: "2.31 GB",
  },
  {
    id: "builder",
    name: "Builder",
    role: "Lead Agent",
    description: "Product engineering and delivery",
    parent: "Benson",
    district: "product-a",
    status: "working",
    load: 86,
    icon: Building2,
    x: 21,
    y: 38,
    task: "Rendering the Jarvis application surfaces",
    delegation: "Frontend is refining responsive hierarchy",
    memory: "Product A Build Context",
    memorySize: "812 MB",
    children: ["frontend", "backend"],
  },
  {
    id: "researcher",
    name: "Researcher",
    role: "Lead Agent",
    description: "Research, validation, and synthesis",
    parent: "Benson",
    district: "product-a",
    status: "working",
    load: 68,
    icon: Search,
    x: 33,
    y: 38,
    task: "Validating voice-frame mapping and design evidence",
    delegation: "Scout is collecting source material",
    memory: "Product A Research Context",
    memorySize: "624 MB",
    children: ["scout", "analyst"],
  },
  {
    id: "memory",
    name: "Memory",
    role: "Lead Agent",
    description: "Knowledge indexing and context retrieval",
    parent: "Benson",
    district: "product-b",
    status: "working",
    load: 47,
    icon: Database,
    x: 43,
    y: 68,
    task: "Indexing Obsidian context for active work",
    delegation: "Archivist is reconciling recent decisions",
    memory: "Product B Memory Context",
    memorySize: "1.74 GB",
    children: ["indexer", "archivist"],
  },
  {
    id: "monitor",
    name: "Monitor",
    role: "Lead Agent",
    description: "Service health and operational telemetry",
    parent: "Benson",
    district: "product-b",
    status: "working",
    load: 32,
    icon: Activity,
    x: 58,
    y: 68,
    task: "Watching service health and task throughput",
    delegation: "Watcher is tracing two slow operations",
    memory: "Product B Runtime Context",
    memorySize: "438 MB",
    children: ["watcher", "alerts"],
  },
  {
    id: "calendar",
    name: "Calendar",
    role: "Personal Agent",
    description: "Schedule planning and time protection",
    parent: "Benson",
    district: "personal",
    status: "active",
    load: 18,
    icon: CalendarDays,
    x: 71,
    y: 38,
    task: "Preparing the next-day schedule brief",
    delegation: "Scheduler is checking focus-time conflicts",
    memory: "Personal Schedule Context",
    memorySize: "214 MB",
    children: ["scheduler"],
  },
  {
    id: "messenger",
    name: "Messenger",
    role: "Personal Agent",
    description: "Communication triage and dispatch",
    parent: "Benson",
    district: "personal",
    status: "waiting",
    load: 8,
    icon: MessageSquareText,
    x: 82,
    y: 38,
    task: "Waiting for approval to dispatch the daily brief",
    delegation: "Comms has a response draft ready",
    memory: "Personal Communications Context",
    memorySize: "198 MB",
    children: ["comms"],
  },
  {
    id: "frontend",
    name: "Frontend",
    role: "Sub-agent",
    description: "Interface implementation",
    parent: "Builder",
    district: "product-a",
    status: "working",
    load: 72,
    icon: Cpu,
    x: 17,
    y: 58,
    task: "Refining city overlays",
    delegation: "No current delegation",
    memory: "UI Working Context",
    memorySize: "126 MB",
  },
  {
    id: "backend",
    name: "Backend",
    role: "Sub-agent",
    description: "Services and integration",
    parent: "Builder",
    district: "product-a",
    status: "working",
    load: 61,
    icon: HardDrive,
    x: 26,
    y: 58,
    task: "Mapping runtime agent events",
    delegation: "No current delegation",
    memory: "Runtime Working Context",
    memorySize: "142 MB",
  },
  {
    id: "scout",
    name: "Scout",
    role: "Sub-agent",
    description: "Source discovery",
    parent: "Researcher",
    district: "product-a",
    status: "working",
    load: 54,
    icon: Search,
    x: 31,
    y: 58,
    task: "Collecting current source evidence",
    delegation: "No current delegation",
    memory: "Research Working Context",
    memorySize: "96 MB",
  },
  {
    id: "analyst",
    name: "Analyst",
    role: "Sub-agent",
    description: "Evidence synthesis",
    parent: "Researcher",
    district: "product-a",
    status: "working",
    load: 58,
    icon: GitBranch,
    x: 38,
    y: 58,
    task: "Comparing implementation evidence",
    delegation: "No current delegation",
    memory: "Analysis Working Context",
    memorySize: "118 MB",
  },
  {
    id: "indexer",
    name: "Indexer",
    role: "Sub-agent",
    description: "Semantic indexing",
    parent: "Memory",
    district: "product-b",
    status: "working",
    load: 46,
    icon: Database,
    x: 39,
    y: 84,
    task: "Indexing recently changed notes",
    delegation: "No current delegation",
    memory: "Index Queue",
    memorySize: "327 MB",
  },
  {
    id: "archivist",
    name: "Archivist",
    role: "Sub-agent",
    description: "Decision provenance",
    parent: "Memory",
    district: "product-b",
    status: "working",
    load: 40,
    icon: Box,
    x: 46,
    y: 84,
    task: "Reconciling design decisions",
    delegation: "No current delegation",
    memory: "Archive Queue",
    memorySize: "289 MB",
  },
  {
    id: "watcher",
    name: "Watcher",
    role: "Sub-agent",
    description: "Runtime observation",
    parent: "Monitor",
    district: "product-b",
    status: "working",
    load: 35,
    icon: Activity,
    x: 55,
    y: 84,
    task: "Tracing slow operations",
    delegation: "No current delegation",
    memory: "Telemetry Window",
    memorySize: "84 MB",
  },
  {
    id: "alerts",
    name: "Alerts",
    role: "Sub-agent",
    description: "Exception routing",
    parent: "Monitor",
    district: "product-b",
    status: "working",
    load: 22,
    icon: Zap,
    x: 62,
    y: 84,
    task: "Routing operational exceptions",
    delegation: "No current delegation",
    memory: "Alert Context",
    memorySize: "74 MB",
  },
  {
    id: "scheduler",
    name: "Scheduler",
    role: "Sub-agent",
    description: "Calendar optimization",
    parent: "Calendar",
    district: "personal",
    status: "active",
    load: 21,
    icon: CalendarDays,
    x: 72,
    y: 60,
    task: "Checking focus-time conflicts",
    delegation: "No current delegation",
    memory: "Schedule Working Context",
    memorySize: "88 MB",
  },
  {
    id: "comms",
    name: "Comms",
    role: "Sub-agent",
    description: "Response preparation",
    parent: "Messenger",
    district: "personal",
    status: "waiting",
    load: 10,
    icon: MessageSquareText,
    x: 82,
    y: 60,
    task: "Holding a response draft for approval",
    delegation: "No current delegation",
    memory: "Draft Context",
    memorySize: "62 MB",
  },
];

const districtMeta = {
  "product-a": { label: "Product A", count: 6, health: "Optimal" },
  "product-b": { label: "Product B", count: 6, health: "Optimal" },
  personal: { label: "Personal", count: 4, health: "Optimal" },
};

const trafficLinks = [
  ["benson", "builder"],
  ["benson", "researcher"],
  ["benson", "memory"],
  ["benson", "monitor"],
  ["benson", "calendar"],
  ["benson", "messenger"],
  ["builder", "frontend"],
  ["builder", "backend"],
  ["researcher", "scout"],
  ["researcher", "analyst"],
  ["memory", "indexer"],
  ["memory", "archivist"],
  ["monitor", "watcher"],
  ["monitor", "alerts"],
  ["calendar", "scheduler"],
  ["messenger", "comms"],
];

function useLocalClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

function TrafficField({ enabled, district }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = canvas?.parentElement;
    if (!canvas || !stage) return undefined;

    const context = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let ratio = 1;
    let frame = 0;
    let lastDraw = 0;

    const resize = () => {
      const bounds = stage.getBoundingClientRect();
      width = Math.max(1, bounds.width);
      height = Math.max(1, bounds.height);
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(stage);
    resize();

    const draw = (time) => {
      if (time - lastDraw < 32) {
        frame = window.requestAnimationFrame(draw);
        return;
      }
      lastDraw = time;
      context.clearRect(0, 0, width, height);

      if (enabled) {
        trafficLinks.forEach(([sourceId, targetId], index) => {
          const source = agents.find((agent) => agent.id === sourceId);
          const target = agents.find((agent) => agent.id === targetId);
          const visible = district === "all" || target.district === district;
          if (!source || !target || !visible) return;

          const x1 = width * source.x * 0.01;
          const y1 = height * source.y * 0.01;
          const x2 = width * target.x * 0.01;
          const y2 = height * target.y * 0.01;
          const isWaiting = target.status === "waiting";
          const progress = reduceMotion ? 0.62 : (time * 0.00011 + index * 0.19) % 1;
          const packetX = x1 + (x2 - x1) * progress;
          const packetY = y1 + (y2 - y1) * progress;

          context.save();
          context.beginPath();
          context.moveTo(x1, y1);
          context.lineTo(x2, y2);
          context.strokeStyle = isWaiting
            ? "rgba(255, 178, 55, .12)"
            : "rgba(72, 215, 255, .1)";
          context.lineWidth = 1;
          context.stroke();
          context.beginPath();
          context.arc(packetX, packetY, isWaiting ? 2.8 : 2.3, 0, Math.PI * 2);
          context.fillStyle = isWaiting ? "#ffb237" : "#8be8ff";
          context.shadowColor = isWaiting ? "#ff9f1a" : "#47d8ff";
          context.shadowBlur = 12;
          context.fill();
          context.restore();
        });
      }
      frame = window.requestAnimationFrame(draw);
    };

    frame = window.requestAnimationFrame(draw);
    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [district, enabled]);

  return <canvas className="city-traffic" ref={canvasRef} aria-hidden="true" />;
}

function AgentMarker({ agent, selected, showSubAgents, district, onSelect }) {
  const Icon = agent.icon;
  const isSubAgent = agent.role === "Sub-agent";
  const filtered = district !== "all" && agent.district !== district && agent.district !== "core";
  if (isSubAgent && !showSubAgents) return null;

  return (
    <button
      className={`agent-marker ${isSubAgent ? "agent-marker--sub" : ""} ${
        selected ? "is-selected" : ""
      } ${filtered ? "is-filtered" : ""} agent-marker--${agent.status}`}
      style={{ left: `${agent.x}%`, top: `${agent.y}%` }}
      type="button"
      aria-pressed={selected}
      aria-label={`Select ${agent.name}, ${agent.status}`}
      onClick={() => onSelect(agent.id)}
    >
      <span className="agent-marker__icon">
        <Icon aria-hidden="true" />
      </span>
      <span className="agent-marker__label">
        <strong>{agent.name}</strong>
        <small>{agent.status}</small>
      </span>
    </button>
  );
}

function CityStage({ selectedId, district, showSubAgents, showTraffic, onSelect }) {
  const stageRef = useRef(null);

  return (
    <div className="city-stage" ref={stageRef}>
      <img src={`${import.meta.env.BASE_URL}static/agents-command-city.png`} alt="" className="city-stage__art" />
      <div className="city-stage__vignette" aria-hidden="true" />
      <TrafficField enabled={showTraffic} district={district} />

      <div className="system-root-label">
        <Orbit aria-hidden="true" />
        <span>
          Hermes Core / Jarvis
          <small>System root</small>
        </span>
      </div>

      <div className="district-label district-label--a">
        <strong>Product A District</strong>
        <span>6 agents · health <b>optimal</b></span>
      </div>
      <div className="district-label district-label--b">
        <strong>Product B District</strong>
        <span>6 agents · health <b>optimal</b></span>
      </div>
      <div className="district-label district-label--personal">
        <strong>Personal District</strong>
        <span>4 agents · health <b>optimal</b></span>
      </div>

      {agents.map((agent) => (
        <AgentMarker
          key={agent.id}
          agent={agent}
          selected={selectedId === agent.id}
          showSubAgents={showSubAgents}
          district={district}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

function TreeView({ selectedId, district, showSubAgents, onSelect }) {
  const visibleDistricts = Object.keys(districtMeta).filter(
    (districtId) => district === "all" || district === districtId,
  );

  return (
    <div className="hierarchy-view" aria-label="Agent hierarchy tree">
      <button
        className={`hierarchy-root ${selectedId === "benson" ? "is-selected" : ""}`}
        type="button"
        onClick={() => onSelect("benson")}
      >
        <Bot aria-hidden="true" />
        <span>
          <strong>Benson</strong>
          <small>Chief Agent · Hermes Core / Jarvis</small>
        </span>
      </button>
      <div className="hierarchy-districts">
        {visibleDistricts.map((districtId) => {
          const meta = districtMeta[districtId];
          const leads = agents.filter(
            (agent) => agent.district === districtId && agent.role !== "Sub-agent",
          );
          return (
            <section key={districtId} className="hierarchy-district">
              <div className="hierarchy-district__heading">
                <span>{meta.label}</span>
                <b>{meta.health}</b>
              </div>
              {leads.map((lead) => {
                const Icon = lead.icon;
                const children = agents.filter((agent) => lead.children?.includes(agent.id));
                return (
                  <div className="hierarchy-branch" key={lead.id}>
                    <button
                      type="button"
                      className={selectedId === lead.id ? "is-selected" : ""}
                      onClick={() => onSelect(lead.id)}
                    >
                      <Icon aria-hidden="true" />
                      <span>
                        <strong>{lead.name}</strong>
                        <small>{lead.status} · {lead.load}% load</small>
                      </span>
                    </button>
                    {showSubAgents ? (
                      <div className="hierarchy-children">
                        {children.map((child) => (
                          <button
                            key={child.id}
                            type="button"
                            className={selectedId === child.id ? "is-selected" : ""}
                            onClick={() => onSelect(child.id)}
                          >
                            <span className={`status-dot status-dot--${child.status}`} />
                            <span>
                              <strong>{child.name}</strong>
                              <small>{child.task}</small>
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </section>
          );
        })}
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="city-toggle">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <i aria-hidden="true" />
    </label>
  );
}

function AgentsNavigation() {
  const items = [
    ["Dashboard", LayoutDashboard, "/jarvis"],
    ["Memory", Network, "/jarvis/memory"],
    ["Agents", UsersRound, "/jarvis/agents"],
    ["Automations", Workflow, "/cron"],
    ["Vault", Box, "/jarvis/memory"],
    ["Settings", Settings2, "/config"],
  ];

  return (
    <nav className="agents-nav" aria-label="Primary navigation">
      {items.map(([label, Icon, href]) => (
        <Link key={label} to={href} className={label === "Agents" ? "is-current" : ""}>
          <Icon aria-hidden="true" />
          <span>{label}</span>
        </Link>
      ))}
    </nav>
  );
}

export default function JarvisAgentsPage() {
  const [selectedId, setSelectedId] = useState("benson");
  const [district, setDistrict] = useState("all");
  const [view, setView] = useState("city");
  const [showSubAgents, setShowSubAgents] = useState(true);
  const [showTraffic, setShowTraffic] = useState(true);
  const [briefOpen, setBriefOpen] = useState(false);
  const now = useLocalClock();

  const selected = useMemo(
    () => agents.find((agent) => agent.id === selectedId) || agents[0],
    [selectedId],
  );
  const SelectedIcon = selected.icon;
  const childrenCount = selected.children?.length || 0;

  return (
    <div className="agents-shell">
      <header className="agents-topbar">
        <div className="agents-brand">
          <Orbit aria-hidden="true" />
          <span>Hermes OS</span>
          <i>/</i>
          <strong>Jarvis</strong>
        </div>
        <div className="agents-security">
          <span>
            <ShieldCheck aria-hidden="true" />
            Local Secure
          </span>
          <span>
            <i aria-hidden="true" />
            Live
          </span>
        </div>
        <div className="agents-time">
          <time dateTime={now.toISOString()}>
            {now.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" })}
            <b>
              {now.toLocaleTimeString("en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </b>
          </time>
          <small>System time (local)</small>
        </div>
      </header>

      <main className="agents-layout">
        <aside className="city-controls">
          <div className="agents-panel-title">
            <span>City Overview</span>
          </div>
          <div className="city-count">
            <span>Total agents</span>
            <strong>17</strong>
          </div>
          <div className="city-status-list">
            <span>
              <i className="status-dot status-dot--working" />
              Working
              <b>12</b>
            </span>
            <span>
              <i className="status-dot status-dot--active" />
              Active
              <b>3</b>
            </span>
            <span>
              <i className="status-dot status-dot--waiting" />
              Waiting
              <b>2</b>
            </span>
          </div>

          <div className="control-section">
            <span className="control-label">Districts</span>
            {[
              ["all", "All Districts", 17],
              ["product-a", "Product A", 6],
              ["product-b", "Product B", 6],
              ["personal", "Personal", 4],
            ].map(([id, label, count]) => (
              <button
                key={id}
                type="button"
                className={`district-filter ${district === id ? "is-selected" : ""}`}
                aria-pressed={district === id}
                onClick={() => setDistrict(id)}
              >
                <span className={`district-dot district-dot--${id}`} />
                {label}
                <b>{count}</b>
              </button>
            ))}
          </div>

          <div className="control-section">
            <span className="control-label">View options</span>
            <div className="view-switch">
              <button
                type="button"
                className={view === "city" ? "is-selected" : ""}
                aria-pressed={view === "city"}
                onClick={() => setView("city")}
              >
                <Building2 aria-hidden="true" />
                City
              </button>
              <button
                type="button"
                className={view === "tree" ? "is-selected" : ""}
                aria-pressed={view === "tree"}
                onClick={() => setView("tree")}
              >
                <GitBranch aria-hidden="true" />
                Tree
              </button>
            </div>
            <Toggle label="Show sub-agents" checked={showSubAgents} onChange={setShowSubAgents} />
            <Toggle label="Show task traffic" checked={showTraffic} onChange={setShowTraffic} />
          </div>

          <div className="control-section traffic-legend">
            <span className="control-label">Traffic legend</span>
            <span>
              <i className="traffic-line traffic-line--active" />
              <b>Active traffic</b>
              <small>Task in progress</small>
            </span>
            <span>
              <i className="traffic-line traffic-line--waiting" />
              <b>Waiting traffic</b>
              <small>Awaiting action</small>
            </span>
          </div>
        </aside>

        <section className="agents-command-surface" aria-label="Agent command city">
          <div className="surface-heading">
            <span>
              <Building2 aria-hidden="true" />
              Command City
            </span>
            <small>{view === "city" ? "Live operational topology" : "Organizational hierarchy"}</small>
          </div>
          {view === "city" ? (
            <CityStage
              selectedId={selectedId}
              district={district}
              showSubAgents={showSubAgents}
              showTraffic={showTraffic}
              onSelect={setSelectedId}
            />
          ) : (
            <TreeView
              selectedId={selectedId}
              district={district}
              showSubAgents={showSubAgents}
              onSelect={setSelectedId}
            />
          )}
        </section>

        <aside className="agent-details">
          <div className="agents-panel-title">
            <span>Agent Details</span>
          </div>
          <span className="detail-eyebrow">Selected agent</span>
          <div className="selected-agent">
            <span className={`selected-agent__icon selected-agent__icon--${selected.status}`}>
              <SelectedIcon aria-hidden="true" />
            </span>
            <span>
              <strong>{selected.name}</strong>
              <small>{selected.role}</small>
              <b>{selected.status}</b>
            </span>
          </div>

          <div className="detail-card">
            <span>Role</span>
            <strong>{selected.description}</strong>
            <span>Parent</span>
            <strong>{selected.parent}</strong>
          </div>

          <div className="detail-stats">
            <span>
              District
              <b>{selected.district === "core" ? "Core" : districtMeta[selected.district]?.label}</b>
            </span>
            <span>
              Direct agents
              <b>{selected.id === "benson" ? 6 : childrenCount}</b>
            </span>
            <span>
              Sub-agents
              <b>{selected.id === "benson" ? 10 : childrenCount}</b>
            </span>
            <span>
              Operational load
              <b>{selected.load}%</b>
            </span>
          </div>

          <div className="detail-section">
            <span className="detail-eyebrow">Current task</span>
            <p>
              <i className={`status-dot status-dot--${selected.status}`} />
              {selected.task}
            </p>
          </div>

          <div className="detail-section">
            <span className="detail-eyebrow">Recent delegation</span>
            <p>
              <Clock3 aria-hidden="true" />
              {selected.delegation}
            </p>
          </div>

          <div className="detail-section">
            <span className="detail-eyebrow">Memory context</span>
            <div className="memory-context">
              <Database aria-hidden="true" />
              <span>
                <strong>{selected.memory}</strong>
                <small>In use · {selected.memorySize}</small>
              </span>
              <Link to="/jarvis/memory">View</Link>
            </div>
          </div>

          {briefOpen ? (
            <div className="command-brief" role="status">
              <span>Command brief open</span>
              <strong>{selected.name} · {selected.load}% load</strong>
              <p>{selected.task}. Next checkpoint in 18 minutes.</p>
            </div>
          ) : null}

          <button
            className="command-brief-button"
            type="button"
            aria-pressed={briefOpen}
            onClick={() => setBriefOpen((open) => !open)}
          >
            {briefOpen ? "Close command brief" : "Open command brief"}
            <ChevronRight aria-hidden="true" />
          </button>
        </aside>
      </main>

      <AgentsNavigation />
    </div>
  );
}
