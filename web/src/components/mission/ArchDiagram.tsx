import { type FC, useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { HealthChip } from "./types";

interface ArchDiagramProps {
  className?: string;
}

interface NodeDef {
  id: string;
  label: string;
  sub?: string;
  runtimeKey?: string;
}

interface ColDef {
  title: string;
  color: string;
  glowColor: string;
  nodes: NodeDef[];
}

const COLS: ColDef[] = [
  {
    title: "Interaction",
    color: "#00bcd4",
    glowColor: "rgba(0,188,212,0.25)",
    nodes: [
      { id: "chat-ui", label: "Claude Code", sub: "Chat UI :9119" },
      { id: "claude-code", label: "claude", sub: "CLI process", runtimeKey: "claude-code" },
      { id: "codex", label: "Codex", sub: "binary process", runtimeKey: "codex" },
    ],
  },
  {
    title: "Control Plane",
    color: "#43a047",
    glowColor: "rgba(67,160,71,0.25)",
    nodes: [
      { id: "hermes", label: "Hermes Agent", sub: "gateway :8642", runtimeKey: "hermes" },
      { id: "plugins", label: "Plugins", sub: "plugin loader" },
      { id: "billing", label: "Billing Guard", sub: "auth / oauth" },
    ],
  },
  {
    title: "Runtimes",
    color: "#ffa726",
    glowColor: "rgba(255,167,38,0.25)",
    nodes: [
      { id: "ruflo", label: "Ruflo", sub: "swarm / hive", runtimeKey: "ruflo" },
      { id: "kanban", label: "Kanban", sub: "dispatcher", runtimeKey: "kanban" },
      { id: "cron", label: "Cron", sub: "scheduler", runtimeKey: "cron" },
    ],
  },
  {
    title: "Mem / State",
    color: "#ab47bc",
    glowColor: "rgba(171,71,188,0.25)",
    nodes: [
      { id: "redis", label: "Redis", sub: "session cache" },
      { id: "pg", label: "PostgreSQL", sub: "kanban DB" },
      { id: "mvms", label: "MVMS", sub: "memory store" },
      { id: "vec", label: "VectorDB", sub: "embeddings" },
    ],
  },
];

const STATUS_COLOR: Record<string, string> = {
  online: "#00e87a",
  degraded: "#f5a623",
  offline: "#e84040",
  unknown: "#6a8099",
};

interface TooltipState {
  nodeId: string;
  chip: HealthChip | null;
  loading: boolean;
  x: number;
  y: number;
}

export const ArchDiagram: FC<ArchDiagramProps> = ({ className }) => {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const handleNodeClick = useCallback(
    async (node: NodeDef, x: number, y: number) => {
      if (!node.runtimeKey) {
        setTooltip({ nodeId: node.id, chip: null, loading: false, x, y });
        return;
      }

      if (tooltip?.nodeId === node.id) {
        setTooltip(null);
        return;
      }

      setTooltip({ nodeId: node.id, chip: null, loading: true, x, y });

      try {
        const chip = await api.getRuntimeHealth(node.runtimeKey);
        setTooltip((prev) =>
          prev?.nodeId === node.id
            ? { ...prev, chip, loading: false }
            : prev,
        );
      } catch {
        setTooltip((prev) =>
          prev?.nodeId === node.id ? { ...prev, loading: false } : prev,
        );
      }
    },
    [tooltip],
  );

  const W = 800;
  const H = 340;
  const colW = W / COLS.length;
  const nodeH = 40;
  const nodeGap = 8;
  const headerH = 32;
  const padX = 10;
  const padTop = 12;

  return (
    <div className={className} style={{ position: "relative", height: "100%", minHeight: 200 }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", height: "100%", display: "block" }}
        aria-label="Hermes architecture diagram"
        onClick={(e) => {
          const target = e.target as Element;
          if (!target.closest("[data-node]")) setTooltip(null);
        }}
      >
        <defs>
          {COLS.map((col) => (
            <filter key={`glow-${col.title}`} id={`glow-${col.title.replace(/\s/g, "")}`}>
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          ))}
        </defs>

        {COLS.map((col, ci) => {
          const cx = ci * colW;
          return (
            <g key={col.title}>
              {/* Column border */}
              <rect
                x={cx + 2}
                y={2}
                width={colW - 4}
                height={H - 4}
                rx={4}
                fill="none"
                stroke={col.color}
                strokeOpacity={0.25}
                strokeWidth={1}
              />

              {/* Column header */}
              <rect
                x={cx + 2}
                y={2}
                width={colW - 4}
                height={headerH}
                rx={4}
                fill={col.color}
                fillOpacity={0.15}
              />
              <text
                x={cx + colW / 2}
                y={headerH / 2 + 2 + 2}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={col.color}
                fontSize={11}
                fontFamily="ui-monospace, monospace"
                fontWeight="600"
                letterSpacing="0.06em"
              >
                {col.title.toUpperCase()}
              </text>

              {/* Nodes */}
              {col.nodes.map((node, ni) => {
                const ny = headerH + padTop + ni * (nodeH + nodeGap);
                const isActive = tooltip?.nodeId === node.id;
                const hasRuntime = !!node.runtimeKey;

                return (
                  <g
                    key={node.id}
                    data-node="true"
                    style={{ cursor: hasRuntime ? "pointer" : "default" }}
                    onClick={(e) => {
                      if (!hasRuntime) return;
                      e.stopPropagation();
                      const rect = (e.currentTarget.closest("svg") as SVGSVGElement)
                        ?.getBoundingClientRect();
                      const svgRect = e.currentTarget.getBoundingClientRect();
                      handleNodeClick(
                        node,
                        svgRect.left - (rect?.left ?? 0) + svgRect.width / 2,
                        svgRect.top - (rect?.top ?? 0),
                      );
                    }}
                  >
                    <rect
                      x={cx + padX}
                      y={ny}
                      width={colW - padX * 2}
                      height={nodeH}
                      rx={3}
                      fill={col.color}
                      fillOpacity={isActive ? 0.22 : 0.1}
                      stroke={col.color}
                      strokeOpacity={isActive ? 0.8 : 0.35}
                      strokeWidth={isActive ? 1.5 : 1}
                    />

                    {/* Status dot for runtime nodes */}
                    {hasRuntime && (
                      <circle
                        cx={cx + padX + 8}
                        cy={ny + nodeH / 2}
                        r={3.5}
                        fill={
                          tooltip?.nodeId === node.id && tooltip.chip
                            ? STATUS_COLOR[tooltip.chip.status] ?? STATUS_COLOR.unknown
                            : STATUS_COLOR.unknown
                        }
                      />
                    )}

                    <text
                      x={cx + padX + (hasRuntime ? 18 : 8)}
                      y={ny + nodeH / 2 - 5}
                      fill={col.color}
                      fontSize={10}
                      fontFamily="ui-monospace, monospace"
                      fontWeight="600"
                    >
                      {node.label}
                    </text>

                    {node.sub && (
                      <text
                        x={cx + padX + (hasRuntime ? 18 : 8)}
                        y={ny + nodeH / 2 + 8}
                        fill={col.color}
                        fillOpacity={0.6}
                        fontSize={9}
                        fontFamily="ui-monospace, monospace"
                      >
                        {node.sub}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Inter-column arrows */}
        {[0, 1, 2].map((ci) => {
          const x1 = (ci + 1) * colW - 2;
          const x2 = (ci + 1) * colW + 2;
          const midY = H / 2;
          return (
            <g key={`arrow-${ci}`}>
              <line
                x1={x1}
                y1={midY}
                x2={x2}
                y2={midY}
                stroke="#6a8099"
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
            </g>
          );
        })}
      </svg>

      {/* Tooltip overlay */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            bottom: "100%",
            left: `${tooltip.x}px`,
            transform: "translateX(-50%)",
            background: "var(--color-background, #07090f)",
            border: "1px solid #6a8099",
            borderRadius: 4,
            padding: "6px 10px",
            fontSize: 11,
            fontFamily: "ui-monospace, monospace",
            color: "var(--color-foreground, #a8c0d6)",
            whiteSpace: "nowrap",
            zIndex: 10,
            pointerEvents: "none",
            marginBottom: 4,
          }}
        >
          {tooltip.loading ? (
            <span style={{ color: "#6a8099" }}>probing…</span>
          ) : tooltip.chip ? (
            <>
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: STATUS_COLOR[tooltip.chip.status] ?? STATUS_COLOR.unknown,
                  marginRight: 6,
                  verticalAlign: "middle",
                }}
              />
              <strong>{tooltip.chip.label}</strong>:{" "}
              {tooltip.chip.status}
              {tooltip.chip.latencyMs != null &&
                ` · ${tooltip.chip.latencyMs.toFixed(1)}ms`}
              {tooltip.chip.detail && ` · ${tooltip.chip.detail}`}
            </>
          ) : (
            <span style={{ color: "#6a8099" }}>no runtime probe</span>
          )}
        </div>
      )}
    </div>
  );
};
