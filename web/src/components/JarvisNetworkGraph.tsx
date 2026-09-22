import React, { useState } from 'react';
import { Network } from 'lucide-react';

export interface GraphNode {
  id: string;
  label: string;
  category: 'core' | 'widget' | 'saas' | 'telemetry' | 'model';
  status: 'Online' | 'Active' | 'Syncing' | 'Standby';
  details: string;
  latency: string;
  value: number;
  color: string;
  x: number;
  y: number;
}

export interface GraphLink {
  source: string;
  target: string;
}

const NODES: GraphNode[] = [
  {
    id: 'jarvis-core',
    label: 'J.A.R.V.I.S. Core',
    category: 'core',
    status: 'Active',
    details: 'Chief of Staff Mark 85 AI Engine',
    latency: '12ms',
    value: 28,
    color: '#00f0ff',
    x: 250,
    y: 160,
  },
  {
    id: 'hermes-agent',
    label: 'Hermes Agent',
    category: 'widget',
    status: 'Online',
    details: 'Autonomous Core, Multi-Channel Swarm & Tools',
    latency: '18ms',
    value: 22,
    color: '#00d2ff',
    x: 100,
    y: 80,
  },
  {
    id: 'github-hub',
    label: 'GitHub Hub',
    category: 'saas',
    status: 'Active',
    details: '@IbrahimAbdelsattar - 41 Repositories Synced',
    latency: '22ms',
    value: 24,
    color: '#38bdf8',
    x: 400,
    y: 80,
  },
  {
    id: 'biometrics',
    label: 'Biometrics',
    category: 'telemetry',
    status: 'Syncing',
    details: 'Commander Ibrahim Telemetry: Focus 94%, Energy 88%',
    latency: '5ms',
    value: 18,
    color: '#ffb700',
    x: 250,
    y: 30,
  },
  {
    id: 'google-cloud',
    label: 'Google Cloud',
    category: 'saas',
    status: 'Active',
    details: 'BigQuery, Cloud Run, GCS, Dataproc & Cloud SQL',
    latency: '15ms',
    value: 22,
    color: '#22c55e',
    x: 80,
    y: 250,
  },
  {
    id: 'rag-chatbot',
    label: 'MR-NLP RAG',
    category: 'model',
    status: 'Online',
    details: 'Robust Vector RAG Pipeline & NLP Systems',
    latency: '28ms',
    value: 20,
    color: '#a855f7',
    x: 250,
    y: 280,
  },
  {
    id: 'voice-engine',
    label: 'Voice Sentinel',
    category: 'widget',
    status: 'Active',
    details: 'Real-time Audio Call, Speech & Watchdog',
    latency: '35ms',
    value: 20,
    color: '#80f7ff',
    x: 420,
    y: 250,
  },
];

const LINKS: GraphLink[] = [
  { source: 'jarvis-core', target: 'hermes-agent' },
  { source: 'jarvis-core', target: 'github-hub' },
  { source: 'jarvis-core', target: 'google-cloud' },
  { source: 'jarvis-core', target: 'rag-chatbot' },
  { source: 'jarvis-core', target: 'voice-engine' },
  { source: 'jarvis-core', target: 'biometrics' },
  { source: 'hermes-agent', target: 'google-cloud' },
];

interface JarvisNetworkGraphProps {
  onNodeSelect?: (node: GraphNode) => void;
  height?: number;
}

export const JarvisNetworkGraph: React.FC<JarvisNetworkGraphProps> = ({
  onNodeSelect,
  height = 360,
}) => {
  const [selectedNode, setSelectedNode] = useState<GraphNode>(NODES[0]);

  const handleSelect = (node: GraphNode) => {
    setSelectedNode(node);
    if (onNodeSelect) onNodeSelect(node);
  };

  return (
    <div
      className="relative w-full rounded-xl bg-[#040d1a]/95 border border-[#00f0ff]/30 overflow-hidden flex flex-col"
      style={{ height }}
    >
      {/* Topology Header */}
      <div className="px-3 py-2 border-b border-[#00f0ff]/20 bg-[#071526]/80 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-2 text-[#00f0ff]">
          <Network className="size-4 animate-pulse" />
          <span className="font-bold tracking-wider">HOLOGRAPHIC NEURAL TOPOLOGY</span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-[#80f7ff]/70">
          <span className="inline-block size-2 rounded-full bg-emerald-400 animate-ping" />
          <span>7 NODES ACTIVE</span>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative flex-1 w-full overflow-hidden">
        <svg
          viewBox="0 0 500 320"
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <filter id="neon-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <linearGradient id="cyber-line" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#0088ff" stopOpacity="0.2" />
            </linearGradient>
          </defs>

          {/* Links */}
          {LINKS.map((link, i) => {
            const src = NODES.find((n) => n.id === link.source);
            const tgt = NODES.find((n) => n.id === link.target);
            if (!src || !tgt) return null;
            const isSelected = selectedNode.id === src.id || selectedNode.id === tgt.id;
            return (
              <g key={i}>
                <line
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  stroke={isSelected ? '#00f0ff' : 'rgba(0, 240, 255, 0.25)'}
                  strokeWidth={isSelected ? 2 : 1}
                  strokeDasharray={isSelected ? '4 2' : undefined}
                />
              </g>
            );
          })}

          {/* Nodes */}
          {NODES.map((node) => {
            const isSelected = selectedNode.id === node.id;
            return (
              <g
                key={node.id}
                onClick={() => handleSelect(node)}
                className="cursor-pointer transition-transform duration-200"
                style={{ transformOrigin: `${node.x}px ${node.y}px` }}
              >
                {isSelected && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.value + 6}
                    fill="none"
                    stroke={node.color}
                    strokeWidth="1.5"
                    opacity="0.6"
                    className="animate-pulse"
                  />
                )}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.value}
                  fill="#06162a"
                  stroke={node.color}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  filter={isSelected ? 'url(#neon-glow)' : undefined}
                />
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.value * 0.35}
                  fill={node.color}
                  opacity={isSelected ? 1 : 0.7}
                />
                <text
                  x={node.x}
                  y={node.y + node.value + 13}
                  textAnchor="middle"
                  fill="#c8c6c5"
                  fontSize="9.5"
                  fontFamily="monospace"
                  className={isSelected ? 'font-bold fill-[#00f0ff]' : ''}
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Floating Selected Node Info Card */}
        {selectedNode && (
          <div className="absolute bottom-2 left-2 right-2 sm:right-auto sm:max-w-xs bg-[#07172b]/95 border border-[#00f0ff]/40 p-2.5 rounded-lg backdrop-blur-md shadow-[0_0_15px_rgba(0,240,255,0.15)] text-xs font-mono">
            <div className="flex items-center justify-between gap-2 border-b border-[#00f0ff]/20 pb-1.5 mb-1.5">
              <span className="font-bold text-[#00f0ff]">{selectedNode.label}</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30">
                {selectedNode.status}
              </span>
            </div>
            <p className="text-[11px] text-[#80f7ff]/75 leading-tight mb-1.5">
              {selectedNode.details}
            </p>
            <div className="flex items-center justify-between text-[10px] text-[#80f7ff]/60">
              <span>LATENCY: <strong className="text-amber-300">{selectedNode.latency}</strong></span>
              <span className="uppercase text-[#00f0ff]">SYS: OK</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default JarvisNetworkGraph;
