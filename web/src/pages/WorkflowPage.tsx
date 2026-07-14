import { useCallback, useEffect, useState, useMemo, useLayoutEffect } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Handle,
  Position,
  type NodeProps,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  Play,
  Save,
  Trash2,
  Plus,
  GitBranch,
  Cpu,
  Package,
} from "lucide-react";

import { api } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";

// ── Custom Node Components ──

function StartNode() {
  return (
    <div className="px-4 py-2 rounded-md bg-emerald-950 border border-emerald-500/50 shadow-md text-emerald-300 text-xs font-mono font-bold flex items-center gap-2">
      <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
      START
      <Handle type="source" position={Position.Bottom} className="w-2 h-2 !bg-emerald-500" />
    </div>
  );
}

function EndNode() {
  return (
    <div className="px-4 py-2 rounded-md bg-rose-950 border border-rose-500/50 shadow-md text-rose-300 text-xs font-mono font-bold flex items-center gap-2">
      <div className="h-2 w-2 rounded-full bg-rose-400" />
      END
      <Handle type="target" position={Position.Top} className="w-2 h-2 !bg-rose-500" />
    </div>
  );
}

type AgentNodeData = {
  label: string;
  model: string;
  toolsets: string[];
  status?: "idle" | "pending" | "running" | "completed" | "failed";
  prompt?: string;
};

function AgentNode({ data }: NodeProps<Node<AgentNodeData>>) {
  const statusColor = {
    idle: "bg-muted-foreground/30 border-muted-foreground/20 text-muted-foreground",
    pending: "bg-blue-950/60 border-blue-500/30 text-blue-400 animate-pulse",
    running: "bg-indigo-950 border-indigo-500 text-indigo-300 shadow-[0_0_10px_rgba(99,102,241,0.2)]",
    completed: "bg-emerald-950/80 border-emerald-500/50 text-emerald-300",
    failed: "bg-rose-950/80 border-rose-500/50 text-rose-300",
  }[data.status || "idle"];

  return (
    <div className={cn("p-3 rounded-lg border bg-card text-left min-w-[180px] shadow-lg transition-all", statusColor)}>
      <Handle type="target" position={Position.Top} className="w-2 h-2" />
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <span className="font-bold text-xs truncate max-w-[130px]">
          {data.label || "Subagent Worker"}
        </span>
        <Badge tone={data.status === "completed" ? "success" : data.status === "failed" ? "destructive" : "secondary"}>
          {data.status || "idle"}
        </Badge>
      </div>
      <div className="flex flex-col gap-1 text-[10px] opacity-80">
        <div className="flex items-center gap-1">
          <Cpu className="h-2.5 w-2.5" />
          <span className="truncate max-w-[140px] font-mono">{data.model || "default"}</span>
        </div>
        <div className="flex items-center gap-1">
          <Package className="h-2.5 w-2.5" />
          <span className="truncate max-w-[140px] font-mono">
            {data.toolsets?.length ? data.toolsets.join(", ") : "none"}
          </span>
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
}

type GateNodeData = {
  label: string;
  status?: "idle" | "pending" | "approved" | "denied";
  prompt?: string;
};

function GateNode({ data }: NodeProps<Node<GateNodeData>>) {
  const statusColor = {
    idle: "bg-amber-950/30 border-amber-500/30 text-amber-300/80",
    pending: "bg-amber-950 border-amber-500 text-amber-300 animate-pulse shadow-[0_0_10px_rgba(245,158,11,0.2)]",
    approved: "bg-emerald-950/80 border-emerald-500/50 text-emerald-300",
    denied: "bg-rose-950/80 border-rose-500/50 text-rose-300",
  }[data.status || "idle"];

  return (
    <div className={cn("p-3 rounded-lg border text-center min-w-[140px] shadow-lg", statusColor)}>
      <Handle type="target" position={Position.Top} className="w-2 h-2" />
      <div className="font-mono font-bold text-[10px] uppercase opacity-75 mb-0.5">Approval Gate</div>
      <div className="font-bold text-xs truncate">{data.label || "Manual Sign-off"}</div>
      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
}

const nodeTypes = {
  start: StartNode,
  end: EndNode,
  agent: AgentNode,
  gate: GateNode,
};

// ── Main Component ──

export default function WorkflowPage() {
  const { setEnd } = usePageHeader();
  const { toast, showToast } = useToast();

  const [workflows, setWorkflows] = useState<string[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState("");
  const [workflowName, setWorkflowName] = useState("New Workflow");

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([
    { id: "start", type: "start", position: { x: 250, y: 50 }, data: {} },
    { id: "end", type: "end", position: { x: 250, y: 400 }, data: {} },
  ]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const [models, setModels] = useState<string[]>([]);
  const [toolsets, setToolsetOptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  // ── Node Inspector State ──
  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId),
    [nodes, selectedNodeId],
  );

  const loadToolsetOptions = useCallback(async () => {
    try {
      const res = await api.getToolsets();
      setToolsetOptions(res.map((t) => t.name));
    } catch {
      setToolsetOptions(["terminal", "file", "web", "mcp"]);
    }
  }, []);

  const loadModelOptions = useCallback(async () => {
    try {
      const res = await api.getModelOptions();
      // Combine all provider models into a flat list
      const options: string[] = [];
      if (res.providers) {
        res.providers.forEach((provider) => {
          if (provider.models) {
            provider.models.forEach((m: string) => {
              options.push(`${provider.slug}/${m}`);
            });
          }
        });
      }
      setModels(options);
    } catch {
      setModels(["anthropic/claude-3-5-sonnet", "google/gemini-2.5-pro", "openai/gpt-4o"]);
    }
  }, []);

  const loadWorkflows = useCallback(async () => {
    try {
      const res = await api.getWorkflows();
      setWorkflows(res.workflows || []);
    } catch {
      showToast("Failed to fetch workflows list", "error");
    }
  }, [showToast]);

  useEffect(() => {
    void loadToolsetOptions();
    void loadModelOptions();
    void loadWorkflows();
  }, [loadToolsetOptions, loadModelOptions, loadWorkflows]);

  // Page header Create button
  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={async () => {
          setSelectedWorkflow("");
          setWorkflowName("New Workflow");
          setNodes([
            { id: "start", type: "start", position: { x: 250, y: 50 }, data: {} },
            { id: "end", type: "end", position: { x: 250, y: 450 }, data: {} },
          ]);
          setEdges([]);
          setSelectedNodeId(null);
        }}
      >
        <Plus className="mr-2 h-4 w-4" />
        New Graph
      </Button>,
    );
    return () => setEnd(null);
  }, [setEnd]);

  // Connect edges
  const onConnect = useCallback(
    (params: any) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  // Add Nodes helper
  const addNode = (type: "agent" | "gate") => {
    const id = `${type}-${Date.now()}`;
    const newNode: Node = {
      id,
      type,
      position: { x: 250, y: 200 + nodes.length * 20 },
      data:
        type === "agent"
          ? { label: `Agent ${nodes.length - 1}`, model: "default", toolsets: ["file", "web"], status: "idle" }
          : { label: "Approval Sign-off", status: "idle" },
    };
    setNodes((nds) => [...nds, newNode]);
    setSelectedNodeId(id);
  };

  // Delete Node helper
  const deleteSelectedNode = () => {
    if (!selectedNodeId || selectedNodeId === "start" || selectedNodeId === "end") return;
    setNodes((nds) => nds.filter((n) => n.id !== selectedNodeId));
    setEdges((eds) => eds.filter((e) => e.source !== selectedNodeId && e.target !== selectedNodeId));
    setSelectedNodeId(null);
  };

  // Edit Node property
  const updateSelectedNodeData = (key: string, value: any) => {
    if (!selectedNodeId) return;
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === selectedNodeId) {
          return {
            ...node,
            data: {
              ...node.data,
              [key]: value,
            },
          };
        }
        return node;
      }),
    );
  };

  // Load Workflow
  const handleLoadWorkflow = async (name: string) => {
    if (!name) return;
    setLoading(true);
    try {
      const res = await api.getWorkflow(name);
      const graph = res.graph || {};
      setNodes(graph.nodes || []);
      setEdges(graph.edges || []);
      setWorkflowName(res.name || name);
      setSelectedWorkflow(name);
      setSelectedNodeId(null);
      showToast(`Loaded ${name}`, "success");
    } catch (e) {
      showToast(`Failed to load: ${e}`, "error");
    } finally {
      setLoading(false);
    }
  };

  // Save Workflow
  const handleSaveWorkflow = async () => {
    const slug = workflowName.toLowerCase().replace(/[^a-z0-9-_]/g, "-").trim();
    if (!slug) {
      showToast("Please specify a valid workflow name", "error");
      return;
    }
    setLoading(true);
    try {
      const graph = { nodes, edges };
      await api.saveWorkflow(slug, graph);
      showToast(`Saved workflow ${slug} successfully`, "success");
      void loadWorkflows();
      setSelectedWorkflow(slug);
    } catch (e) {
      showToast(`Failed to save: ${e}`, "error");
    } finally {
      setLoading(false);
    }
  };

  // Delete Workflow blueprint
  const handleDeleteWorkflow = async () => {
    if (!selectedWorkflow) return;
    setLoading(true);
    try {
      await api.deleteWorkflow(selectedWorkflow);
      showToast(`Deleted ${selectedWorkflow}`, "success");
      setSelectedWorkflow("");
      setWorkflowName("New Workflow");
      void loadWorkflows();
    } catch (e) {
      showToast(`Delete failed: ${e}`, "error");
    } finally {
      setLoading(false);
    }
  };

  // Execute Workflow
  const handleExecute = async () => {
    if (!selectedWorkflow) {
      showToast("Please save the workflow first before running it", "error");
      return;
    }
    setExecuting(true);
    setLogs([`[System] Launching execution for workflow: ${selectedWorkflow}...`]);
    try {
      const res = await api.executeWorkflow(selectedWorkflow);
      showToast(`Workflow execution started: ${res.name}`, "success");
      // Poll execution status
      pollStatus(res.name);
    } catch (e) {
      showToast(`Failed to launch execution: ${e}`, "error");
      setExecuting(false);
    }
  };

  // Poll execution status
  const pollStatus = (actionName: string) => {
    const timer = setInterval(async () => {
      try {
        const res = await api.getActionStatus(actionName);
        if (res.lines) {
          setLogs(res.lines);
        }
        if (!res.running) {
          clearInterval(timer);
          setExecuting(false);
          if (res.exit_code === 0) {
            showToast("Workflow completed successfully!", "success");
          } else {
            showToast("Workflow execution failed.", "error");
          }
          void handleLoadWorkflow(selectedWorkflow); // refresh node statuses
        }
      } catch {
        clearInterval(timer);
        setExecuting(false);
      }
    }, 2000);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-80px)] gap-4 select-none">
      <Toast toast={toast} />

      {/* Toolbar */}
      <Card className="shrink-0 border border-current/15">
        <CardContent className="flex flex-wrap items-center justify-between gap-4 py-3 px-4">
          <div className="flex items-center gap-3">
            <GitBranch className="h-5 w-5 text-indigo-400" />
            <Input
              className="w-48 font-mono"
              placeholder="Workflow name"
              value={workflowName}
              onChange={(e) => setWorkflowName(e.target.value)}
            />
            <Button size="sm" ghost onClick={handleSaveWorkflow} disabled={loading}>
              {loading ? <Spinner /> : <Save className="h-4 w-4 mr-1.5" />}
              Save
            </Button>
            {selectedWorkflow && (
              <Button size="sm" destructive ghost onClick={handleDeleteWorkflow} disabled={loading}>
                <Trash2 className="h-4 w-4 mr-1.5" />
                Delete
              </Button>
            )}
          </div>

          <div className="flex items-center gap-3">
            <Label htmlFor="workflow-loader" className="text-xs text-text-secondary">
              Load:
            </Label>
            <Select
              id="workflow-loader"
              className="w-48"
              value={selectedWorkflow}
              onValueChange={handleLoadWorkflow}
            >
              <SelectOption value="">Select a blueprint</SelectOption>
              {workflows.map((w) => (
                <SelectOption key={w} value={w}>
                  {w}
                </SelectOption>
              ))}
            </Select>

            <Button
              size="sm"
              disabled={executing || !selectedWorkflow}
              onClick={handleExecute}
              className="bg-indigo-600 hover:bg-indigo-700 text-white"
            >
              {executing ? <Spinner /> : <Play className="h-4 w-4 mr-1.5 fill-current" />}
              Run Pipeline
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Canvas + Sidebar */}
      <div className="flex flex-1 min-h-0 gap-4">
        {/* React Flow Editor */}
        <div className="flex-1 min-w-0 border border-current/15 rounded-md relative bg-[#0a0a0c]">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
            onPaneClick={() => setSelectedNodeId(null)}
            fitView
          >
            <Controls />
            <MiniMap style={{ background: "#111" }} />
            <Background color="#333" gap={16} />
          </ReactFlow>

          {/* Canvas Toolbar overlay */}
          <div className="absolute top-4 left-4 z-10 flex gap-2">
            <Button size="sm" onClick={() => addNode("agent")}>
              <Plus className="mr-1.5 h-4 w-4" />
              Add Agent
            </Button>
            <Button size="sm" onClick={() => addNode("gate")}>
              <Plus className="mr-1.5 h-4 w-4" />
              Add Approval Gate
            </Button>
          </div>
        </div>

        {/* Sidebar Inspector Panel */}
        <Card className="w-80 shrink-0 border border-current/15 flex flex-col min-h-0">
          <CardHeader className="py-4 px-5 border-b border-current/10 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-mondwest tracking-wider uppercase">Node Inspector</CardTitle>
            {selectedNodeId && (
              <Button ghost size="icon" className="h-6 w-6 text-rose-500 hover:bg-rose-500/10" onClick={deleteSelectedNode}>
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto p-5 grid gap-4">
            {(() => {
              if (!selectedNode) {
                return (
                  <div className="text-center text-xs text-text-tertiary py-12">
                    Click a node to inspect and configure its settings.
                  </div>
                );
              }
              const data = selectedNode.data as any;
              if (selectedNode.type === "start" || selectedNode.type === "end") {
                return (
                  <div className="text-xs text-text-secondary">
                    <span className="font-bold uppercase font-mono text-indigo-400">{selectedNode.type} Node</span>
                    <p className="mt-2 opacity-80">
                      This represents the execution trigger point. Connect edges from Start to your workers, and finally to End.
                    </p>
                  </div>
                );
              }
              if (selectedNode.type === "agent") {
                return (
                  <div className="grid gap-4">
                    <div className="grid gap-1.5">
                      <Label htmlFor="node-label">Task Label</Label>
                      <Input
                        id="node-label"
                        value={data.label || ""}
                        onChange={(e) => updateSelectedNodeData("label", e.target.value)}
                      />
                    </div>

                    <div className="grid gap-1.5">
                      <Label htmlFor="node-model">LLM Model</Label>
                      <Select
                        id="node-model"
                        value={data.model || "default"}
                        onValueChange={(val) => updateSelectedNodeData("model", val)}
                      >
                        <SelectOption value="default">Default Model</SelectOption>
                        {models.map((m) => (
                          <SelectOption key={m} value={m}>
                            {m}
                          </SelectOption>
                        ))}
                      </Select>
                    </div>

                    <div className="grid gap-1.5">
                      <Label>Toolsets Available</Label>
                      <div className="grid grid-cols-2 gap-2 border border-current/10 p-2.5 rounded bg-background-base/20 max-h-[140px] overflow-y-auto">
                        {toolsets.map((t) => {
                          const enabled = data.toolsets?.includes(t);
                          return (
                            <label key={t} className="flex items-center gap-2 text-[10px] font-mono cursor-pointer select-none">
                              <input
                                type="checkbox"
                                checked={enabled}
                                onChange={(e) => {
                                  const list = data.toolsets || [];
                                  const updated = e.target.checked ? [...list, t] : list.filter((item: string) => item !== t);
                                  updateSelectedNodeData("toolsets", updated);
                                }}
                                className="rounded border-current/20 text-indigo-600 focus:ring-0 cursor-pointer"
                              />
                              {t}
                            </label>
                          );
                        })}
                      </div>
                    </div>

                    <div className="grid gap-1.5">
                      <Label htmlFor="node-prompt">Goal Description / Instructions</Label>
                      <textarea
                        id="node-prompt"
                        className="flex min-h-[90px] w-full border border-border bg-background/40 px-3 py-2 text-xs font-courier shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30"
                        placeholder="Describe the instructions for this agent..."
                        value={data.prompt || ""}
                        onChange={(e) => updateSelectedNodeData("prompt", e.target.value)}
                      />
                    </div>

                  </div>
                );
              }
              return (
                <div className="grid gap-4">
                  <div className="grid gap-1.5">
                    <Label htmlFor="gate-label">Gate Label</Label>
                    <Input
                      id="gate-label"
                      value={data.label || ""}
                      onChange={(e) => updateSelectedNodeData("label", e.target.value)}
                    />
                  </div>
                  <div className="grid gap-1.5">
                    <Label htmlFor="gate-prompt">Gate Description / Condition</Label>
                    <textarea
                      id="gate-prompt"
                      className="flex min-h-[90px] w-full border border-border bg-background/40 px-3 py-2 text-xs font-courier shadow-sm focus-visible:outline-none focus-visible:ring-1"
                      placeholder="Specify gate description or validation conditions..."
                      value={data.prompt || ""}
                      onChange={(e) => updateSelectedNodeData("prompt", e.target.value)}
                    />
                  </div>
                </div>
              );
            })()}
          </CardContent>
        </Card>
      </div>

      {/* Log Console overlay */}
      <Card className="h-44 shrink-0 border border-current/15 flex flex-col min-h-0 bg-black/90">
        <CardHeader className="py-2.5 px-4 border-b border-current/10 flex flex-row items-center justify-between">
          <CardTitle className="text-xs font-mono tracking-wider flex items-center gap-1.5">
            <div className={cn("h-2 w-2 rounded-full", executing ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground")} />
            Pipeline Logs
          </CardTitle>
          <Button ghost size="sm" className="h-6 text-xs text-text-tertiary hover:bg-current/5" onClick={() => setLogs([])}>
            Clear Logs
          </Button>
        </CardHeader>
        <CardContent className="flex-1 overflow-y-auto p-3 font-mono text-xs text-emerald-400/90 leading-5">
          {logs.length === 0 ? (
            <div className="text-text-disabled text-center py-6">Logs console idle...</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="whitespace-pre-wrap">
                {log}
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
