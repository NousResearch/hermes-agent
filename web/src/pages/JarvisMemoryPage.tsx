/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type JarvisOverview } from "@/lib/api";
import "./jarvis-memory-vault.css";
import {
  Bot,
  Box,
  CalendarDays,
  CircleGauge,
  Database,
  ExternalLink,
  FileCheck2,
  FileText,
  FolderOpen,
  LayoutDashboard,
  Network,
  Orbit,
  Package,
  Scale,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  UserRound,
  UsersRound,
  Zap,
} from "lucide-react";

const clusters = [
  {
    id: "decisions",
    label: "Decisions",
    count: 184,
    icon: Scale,
    x: 50,
    y: 10,
    notes: [
      { title: "Technical Direction", date: "JUL 25" },
      { title: "Security Posture", date: "JUL 24" },
      { title: "Build vs Buy", date: "JUL 20" },
    ],
  },
  {
    id: "products",
    label: "Products",
    count: 7,
    icon: Package,
    x: 22,
    y: 34,
    notes: [
      { title: "Hermes OS", date: "JUL 26" },
      { title: "Builder", date: "JUL 24" },
      { title: "Relay", date: "JUL 18" },
    ],
  },
  {
    id: "projects",
    label: "Projects",
    count: 236,
    icon: FolderOpen,
    x: 78,
    y: 34,
    notes: [
      { title: "Jarvis", date: "JUL 26" },
      { title: "Hermes Core", date: "JUL 23" },
      { title: "Voice Engine", date: "JUL 21" },
    ],
  },
  {
    id: "people",
    label: "People",
    count: 128,
    icon: UserRound,
    x: 25,
    y: 76,
    notes: [
      { title: "Aaron Olson", date: "JUL 26" },
      { title: "Design Guild", date: "JUL 25" },
      { title: "Engineering", date: "JUL 19" },
    ],
  },
  {
    id: "daily",
    label: "Daily Notes",
    count: 12287,
    icon: CalendarDays,
    x: 75,
    y: 76,
    notes: [
      { title: "Today", date: "JUL 26" },
      { title: "Yesterday", date: "JUL 25" },
      { title: "Jul 24, 2026", date: "JUL 24" },
    ],
  },
];

const relatedNotes = [
  {
    title: "Dashboard Architecture",
    path: "/Projects/Jarvis/Dashboard Architecture.md",
    updated: "JUL 26 2026 · 09:58",
  },
  {
    title: "Voice Core Mapping",
    path: "/Projects/Jarvis/Voice Core Mapping.md",
    updated: "JUL 26 2026 · 09:41",
  },
  {
    title: "Implementation Checklist",
    path: "/Projects/Jarvis/Implementation Checklist.md",
    updated: "JUL 25 2026 · 16:32",
  },
];


function formatCount(value, fallback = "0") {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : fallback;
}

function formatDate(value) {
  if (!value) return "LIVE";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "LIVE";
  return date.toLocaleDateString([], { month: "short", day: "2-digit" }).toUpperCase();
}

function memoryTone(status) {
  if (status === "available") return "green";
  if (status === "setup_needed") return "amber";
  return "cyan";
}

function buildLiveClusters(memoryVault) {
  const memory = memoryVault?.obsidian;
  const recentNotes = memory?.recent_notes || [];
  const recentClusterNotes = recentNotes.slice(0, 3).map((note) => ({ title: note.title, date: formatDate(note.updated_at), href: note.href }));
  const quickLinkNotes = (memory?.quick_links || []).slice(0, 3).map((link) => ({ title: link.label, date: "PIN", href: link.href }));
  return clusters.map((cluster) => {
    if (cluster.id === "decisions") {
      return { ...cluster, count: memory?.decision_count ?? cluster.count, notes: quickLinkNotes.length ? quickLinkNotes : cluster.notes };
    }
    if (cluster.id === "products" || cluster.id === "projects") {
      return { ...cluster, count: memory?.product_note_count ?? cluster.count, notes: recentClusterNotes.length ? recentClusterNotes : cluster.notes };
    }
    if (cluster.id === "daily") {
      return { ...cluster, count: memory?.note_count ?? cluster.count, notes: recentClusterNotes.length ? recentClusterNotes : cluster.notes };
    }
    if (cluster.id === "people") {
      return { ...cluster, count: recentNotes.length || cluster.count, notes: recentClusterNotes.length ? recentClusterNotes : cluster.notes };
    }
    return cluster;
  });
}

function MemoryField({ activeCluster, query, onSelectCluster, clusters: memoryClusters = clusters, vaultLabel = "Obsidian Vault", noteCountLabel = "0 indexed notes" }) {
  const canvasRef = useRef(null);
  const stageRef = useRef(null);

  const matchingClusters = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return new Set(memoryClusters.map((cluster) => cluster.id));
    return new Set(
      memoryClusters
        .filter(
          (cluster) =>
            cluster.label.toLowerCase().includes(normalized) ||
            cluster.notes.some((note) => note.title.toLowerCase().includes(normalized)),
        )
        .map((cluster) => cluster.id),
    );
  }, [query, memoryClusters]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return undefined;

    const context = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    let width = 0;
    let height = 0;
    let ratio = 1;
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

    const draw = (time) => {
      if (time - lastDraw < 32) {
        frame = window.requestAnimationFrame(draw);
        return;
      }
      lastDraw = time;
      context.clearRect(0, 0, width, height);

      const center = { x: width * 0.5, y: height * 0.48 };
      context.save();
      context.strokeStyle = "rgba(69, 186, 222, .07)";
      context.lineWidth = 1;
      [0.2, 0.31, 0.43].forEach((radius) => {
        context.beginPath();
        context.ellipse(center.x, center.y, width * radius, height * radius * 0.82, 0, 0, Math.PI * 2);
        context.stroke();
      });
      context.restore();

      memoryClusters.forEach((cluster, index) => {
        const point = { x: width * cluster.x * 0.01, y: height * cluster.y * 0.01 };
        const isMatch = matchingClusters.has(cluster.id);
        const isActive = activeCluster === cluster.id;
        const alpha = !isMatch ? 0.08 : isActive ? 0.92 : 0.48;
        const bend = (index % 2 === 0 ? -1 : 1) * height * 0.08;

        context.save();
        for (let noteIndex = 0; noteIndex < 22; noteIndex += 1) {
          const angle = ((noteIndex * 137.5 + index * 29) * Math.PI) / 180;
          const distance = 27 + (noteIndex % 7) * 8;
          const drift = reduceMotion ? 0 : Math.sin(time * 0.00018 + noteIndex) * 1.8;
          const noteX = point.x + Math.cos(angle) * (distance + drift) * 1.55;
          const noteY = point.y + Math.sin(angle) * (distance + drift) * 0.68;
          context.beginPath();
          context.arc(noteX, noteY, noteIndex % 5 === 0 ? 2.7 : 1.55, 0, Math.PI * 2);
          context.fillStyle = isMatch
            ? `rgba(105, 216, 247, ${isActive ? 0.7 : 0.4})`
            : "rgba(105, 216, 247, .04)";
          context.fill();

          if (noteIndex % 4 === 0) {
            context.beginPath();
            context.moveTo(noteX, noteY);
            context.lineTo(point.x, point.y);
            context.strokeStyle = isMatch
              ? `rgba(92, 217, 255, ${isActive ? 0.12 : 0.055})`
              : "rgba(92, 217, 255, .015)";
            context.lineWidth = 0.7;
            context.stroke();
          }
        }

        context.beginPath();
        context.moveTo(center.x, center.y);
        context.bezierCurveTo(
          center.x + (point.x - center.x) * 0.34,
          center.y + bend,
          center.x + (point.x - center.x) * 0.7,
          point.y - bend * 0.25,
          point.x,
          point.y,
        );
        context.strokeStyle = `rgba(73, 207, 247, ${alpha})`;
        context.lineWidth = isActive ? 2.4 : 1.15;
        context.stroke();

        if (isMatch) {
          const progress = reduceMotion ? 0.62 : ((time * 0.000045 + index * 0.17) % 1);
          const oneMinus = 1 - progress;
          const packetX =
            oneMinus ** 3 * center.x +
            3 * oneMinus ** 2 * progress * (center.x + (point.x - center.x) * 0.34) +
            3 * oneMinus * progress ** 2 * (center.x + (point.x - center.x) * 0.7) +
            progress ** 3 * point.x;
          const packetY =
            oneMinus ** 3 * center.y +
            3 * oneMinus ** 2 * progress * (center.y + bend) +
            3 * oneMinus * progress ** 2 * (point.y - bend * 0.25) +
            progress ** 3 * point.y;
          context.beginPath();
          context.arc(packetX, packetY, isActive ? 3.2 : 2.1, 0, Math.PI * 2);
          context.fillStyle = isActive ? "rgba(205, 248, 255, .95)" : "rgba(92, 218, 255, .72)";
          context.shadowColor = "#5cdcff";
          context.shadowBlur = 10;
          context.fill();
        }
        context.restore();
      });

      frame = window.requestAnimationFrame(draw);
    };

    frame = window.requestAnimationFrame(draw);
    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [activeCluster, matchingClusters, memoryClusters]);

  return (
    <div className="memory-stage" ref={stageRef}>
      <canvas ref={canvasRef} aria-hidden="true" />
      <button className="vault-core" type="button" onClick={() => onSelectCluster("projects")}>
        <span className="vault-core__halo" aria-hidden="true" />
        <Database aria-hidden="true" />
        <strong>{vaultLabel}</strong>
        <span>{noteCountLabel}</span>
      </button>

      {memoryClusters.map((cluster) => {
        const Icon = cluster.icon;
        const isMatch = matchingClusters.has(cluster.id);
        const isActive = activeCluster === cluster.id;
        return (
          <button
            key={cluster.id}
            className={`cluster ${isActive ? "is-active" : ""} ${isMatch ? "" : "is-dimmed"}`}
            style={{ left: `${cluster.x}%`, top: `${cluster.y}%` }}
            type="button"
            onClick={() => onSelectCluster(cluster.id)}
            aria-pressed={isActive}
          >
            <span className="cluster__icon">
              <Icon aria-hidden="true" />
            </span>
            <span className="cluster__title">
              {cluster.label}
              <small>{cluster.count.toLocaleString()}</small>
            </span>
            <span className="cluster__notes">
              {cluster.notes.map((note) => (
                <span key={note.title}>
                  <i aria-hidden="true" />
                  <b>{note.title}</b>
                  <em>{note.date}</em>
                </span>
              ))}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function Metric({ icon: Icon, label, value, tone = "cyan" }) {
  return (
    <div className={`metric metric--${tone}`}>
      <Icon aria-hidden="true" />
      <span>
        {label}
        <strong>{value}</strong>
      </span>
    </div>
  );
}

function BottomNav() {
  const items = [
    ["Dashboard", LayoutDashboard, "/jarvis"],
    ["Memory", Network, "/jarvis/memory"],
    ["Agents", UsersRound, "/jarvis/agents"],
    ["Automations", Zap, "/cron"],
    ["Vault", Box, "/jarvis/memory"],
    ["Settings", Settings2, "/config"],
  ];

  return (
    <nav className="bottom-nav" aria-label="Primary navigation">
      {items.map(([label, Icon, href]) => (
        <Link key={label} to={href} className={label === "Memory" ? "is-current" : ""}>
          <Icon aria-hidden="true" />
          <span>{label}</span>
        </Link>
      ))}
    </nav>
  );
}

export default function JarvisMemoryPage() {
  const [query, setQuery] = useState("");
  const [activeCluster, setActiveCluster] = useState("projects");
  const [syncing, setSyncing] = useState(false);
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
  const liveClusters = useMemo(() => buildLiveClusters(overview?.memory_vault), [overview?.memory_vault]);
  const activeClusterData = liveClusters.find((cluster) => cluster.id === activeCluster);
  const selectedNote = memoryVault?.recent_notes?.[0];
  const relatedLiveNotes = (memoryVault?.recent_notes || []).slice(0, 3);
  const vaultHealthLabel = overviewError ? "Unavailable" : memoryVault?.status === "available" ? "Live" : memoryVault?.status === "setup_needed" ? "Setup" : "Loading";
  const vaultHealthValue = overviewError ? "Offline" : memoryVault?.status === "available" ? "100%" : memoryVault?.status === "setup_needed" ? "Setup" : "—";
  const vaultPath = memoryVault?.path || "Obsidian vault not configured";
  const selectedTitle = selectedNote?.title || "Jarvis Visual Direction";
  const selectedHref = selectedNote?.href || memoryVault?.href || "/files";
  const selectedRelativePath = selectedNote?.relative_path || vaultPath;
  const selectedUpdated = selectedNote?.updated_at ? new Date(selectedNote.updated_at).toLocaleString([], { month: "short", day: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "Live overview pending";

  const handleSync = () => {
    setSyncing(true);
    window.setTimeout(() => setSyncing(false), 900);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Orbit aria-hidden="true" />
          <span>Hermes OS</span>
          <i>/</i>
          <strong>Memory Vault</strong>
        </div>
        <div className="system-state">
          <span>
            <ShieldCheck aria-hidden="true" />
            Local Secure
          </span>
          <span>
            <i className="live-dot" aria-hidden="true" />
            Live
          </span>
        </div>
        <div className="topbar__actions">
          <button type="button" className="sync-button" onClick={handleSync}>
            <CircleGauge aria-hidden="true" className={syncing ? "is-spinning" : ""} />
            {syncing ? "Syncing" : "Sync"}
          </button>
          <time dateTime={overview?.generated_at || new Date().toISOString()}>{overview?.generated_at ? new Date(overview.generated_at).toLocaleString([], { month: "short", day: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "Loading live data"}</time>
        </div>
      </header>

      <main className="memory-page">
        <section className="overview-strip" aria-label="Vault overview">
          <div className="metric-group">
            <Metric icon={FileText} label="Notes" value={formatCount(memoryVault?.note_count)} />
            <Metric icon={FileCheck2} label="Decisions" value={formatCount(memoryVault?.decision_count)} tone="green" />
            <Metric icon={Package} label="Products" value={formatCount(memoryVault?.product_note_count)} />
            <Metric icon={ShieldCheck} label={vaultHealthLabel} value={vaultHealthValue} tone={memoryTone(memoryVault?.status)} />
          </div>
          <label className="search-box">
            <Search aria-hidden="true" />
            <span className="sr-only">Search memory</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") setQuery("");
              }}
              placeholder="SEARCH MEMORY"
            />
            <kbd>⌘ K</kbd>
          </label>
        </section>

        <section className="atlas-panel" aria-label="Obsidian knowledge atlas">
          <div className="atlas-panel__title">
            <span>
              <Sparkles aria-hidden="true" />
              Obsidian Memory Vault
            </span>
            <small>
              Knowledge constellation
              <i aria-hidden="true" />
              {query ? `${query} filter active` : `${activeClusterData?.label} selected`}
            </small>
          </div>
          <MemoryField
            activeCluster={activeCluster}
            query={query}
            clusters={liveClusters}
            vaultLabel={memoryVault?.label || "Obsidian Vault"}
            noteCountLabel={`${formatCount(memoryVault?.note_count)} indexed notes`}
            onSelectCluster={setActiveCluster}
          />
        </section>

        <section className="note-panel" aria-label="Selected memory note">
          <div className="selected-note">
            <div className="note-glyph">
              <FileText aria-hidden="true" />
            </div>
            <div className="note-copy">
              <span className="eyebrow">Selected note</span>
              <h1>{selectedTitle}</h1>
              <Link to={selectedHref}>{selectedRelativePath}</Link>
              <p>
                {memoryVault?.message || "Approved constellation dashboard direction and voice-reactive core behavior."}
              </p>
              <div className="note-meta">
                <span>Updated {selectedUpdated}</span>
                <span>
                  <Bot aria-hidden="true" />
                  {memoryVault?.configured ? "Indexed from Obsidian Memory" : "Waiting for vault setup"}
                </span>
              </div>
            </div>
          </div>

          <div className="related-notes">
            <span className="eyebrow">Related notes</span>
            {(relatedLiveNotes.length ? relatedLiveNotes : relatedNotes).map((note, index) => (
              <button key={note.title} type="button" onClick={() => setActiveCluster("projects")}>
                <span className="related-notes__path" aria-hidden="true">
                  <i />
                  <b />
                </span>
                <FileText aria-hidden="true" />
                <span>
                  <strong>{note.title}</strong>
                  <small>{note.relative_path || note.path}</small>
                </span>
                <time>{note.updated_at ? new Date(note.updated_at).toLocaleString([], { month: "short", day: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }) : note.updated}</time>
                {index === 0 ? <span className="recent-tag">Recent</span> : null}
              </button>
            ))}
          </div>

          <Link
            className="obsidian-button"
            to={memoryVault?.href || "/files"}
          >
            <ExternalLink aria-hidden="true" />
            Open memory source
          </Link>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
