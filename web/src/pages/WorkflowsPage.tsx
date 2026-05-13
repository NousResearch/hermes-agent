import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { H2 } from "@/components/NouiTypography";
import { api } from "@/lib/api";
import type {
  WorkflowDagFacts,
  WorkflowNode,
  WorkflowNodeFacts,
  WorkflowSummary,
} from "@/lib/api";
import { cn } from "@/lib/utils";

function formatTime(value?: string | number | null): string {
  if (value === undefined || value === null || value === "") return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function formatJson(value: unknown): string {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function nodeTitle(node: WorkflowNode): string {
  return node.title || node.id;
}

function workflowTitle(workflow: WorkflowSummary): string {
  return workflow.title || workflow.id;
}

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [dag, setDag] = useState<WorkflowDagFacts | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeDetail, setNodeDetail] = useState<WorkflowNodeFacts | null>(null);
  const [loadingWorkflows, setLoadingWorkflows] = useState(true);
  const [loadingDag, setLoadingDag] = useState(false);
  const [loadingNode, setLoadingNode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedWorkflow = useMemo(
    () => workflows.find((workflow) => workflow.id === selectedWorkflowId) ?? null,
    [selectedWorkflowId, workflows],
  );

  const selectedNode = useMemo(
    () => dag?.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [dag?.nodes, selectedNodeId],
  );

  const loadWorkflows = useCallback(() => {
    setLoadingWorkflows(true);
    setError(null);
    api
      .getWorkflows({ limit: 100 })
      .then((response) => {
        const rows = response.facts.workflows;
        setWorkflows(rows);
        setSelectedWorkflowId((current) => current ?? rows[0]?.id ?? null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load workflows"))
      .finally(() => setLoadingWorkflows(false));
  }, []);

  useEffect(() => {
    loadWorkflows();
  }, [loadWorkflows]);

  useEffect(() => {
    if (!selectedWorkflowId) {
      setDag(null);
      setSelectedNodeId(null);
      setNodeDetail(null);
      return;
    }

    setLoadingDag(true);
    setError(null);
    setDag(null);
    setSelectedNodeId(null);
    setNodeDetail(null);

    api
      .getWorkflowDag(selectedWorkflowId)
      .then((response) => setDag(response.facts))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load workflow DAG"))
      .finally(() => setLoadingDag(false));
  }, [selectedWorkflowId]);

  useEffect(() => {
    if (!selectedWorkflowId || !selectedNodeId) {
      setNodeDetail(null);
      return;
    }

    setLoadingNode(true);
    api
      .getWorkflowNode(selectedWorkflowId, selectedNodeId)
      .then((response) => setNodeDetail(response.facts))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load workflow node"))
      .finally(() => setLoadingNode(false));
  }, [selectedNodeId, selectedWorkflowId]);

  if (loadingWorkflows) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-col gap-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <H2>Workflows</H2>
          <p className="mt-1 text-xs tracking-[0.08em] text-midground/60 normal-case">
            Read-only workflow graph, gates, artifacts, and node state.
          </p>
        </div>
        <Button ghost size="sm" onClick={loadWorkflows}>
          Refresh
        </Button>
      </div>

      {error && (
        <Card className="border-destructive/40">
          <CardContent className="py-3 text-sm text-destructive normal-case">{error}</CardContent>
        </Card>
      )}

      {workflows.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-sm text-midground/60 normal-case">
            No workflows found.
          </CardContent>
        </Card>
      ) : (
        <div className="grid min-h-0 gap-4 xl:grid-cols-[18rem_minmax(0,1fr)_22rem]">
          <Card className="min-h-0">
            <CardHeader>
              <CardTitle>Workflow runs</CardTitle>
            </CardHeader>
            <CardContent className="flex max-h-[calc(100vh-14rem)] flex-col gap-2 overflow-auto">
              {workflows.map((workflow) => (
                <button
                  key={workflow.id}
                  className={cn(
                    "flex flex-col gap-1 rounded border border-current/10 px-3 py-2 text-left transition-colors",
                    selectedWorkflowId === workflow.id
                      ? "bg-midground/10 text-midground"
                      : "text-midground/70 hover:bg-midground/5 hover:text-midground",
                  )}
                  onClick={() => setSelectedWorkflowId(workflow.id)}
                >
                  <span className="truncate text-sm tracking-[0.08em]">{workflowTitle(workflow)}</span>
                  <span className="truncate text-[0.65rem] text-midground/45">{workflow.id}</span>
                  <span className="flex items-center gap-2 text-[0.65rem] text-midground/55">
                    <Badge>{workflow.status}</Badge>
                    <span>{workflow.board}</span>
                  </span>
                </button>
              ))}
            </CardContent>
          </Card>

          <Card className="min-h-0">
            <CardHeader>
              <CardTitle>{selectedWorkflow ? workflowTitle(selectedWorkflow) : "Workflow DAG"}</CardTitle>
            </CardHeader>
            <CardContent className="flex max-h-[calc(100vh-14rem)] flex-col gap-4 overflow-auto">
              {loadingDag ? (
                <div className="flex justify-center py-12"><Spinner /></div>
              ) : dag ? (
                <>
                  <div className="grid gap-2 sm:grid-cols-3">
                    <Metric label="Status" value={dag.workflow.status} />
                    <Metric label="Nodes" value={String(dag.nodes.length)} />
                    <Metric label="Edges" value={String(dag.edges.length)} />
                  </div>

                  <section className="grid gap-2">
                    {dag.nodes.map((node) => (
                      <button
                        key={node.id}
                        className={cn(
                          "rounded border border-current/10 p-3 text-left transition-colors",
                          selectedNodeId === node.id
                            ? "bg-primary/10 text-midground"
                            : "hover:bg-midground/5",
                        )}
                        onClick={() => setSelectedNodeId(node.id)}
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-sm tracking-[0.08em]">{nodeTitle(node)}</span>
                          <Badge>{node.status}</Badge>
                          {node.role && <Badge>{node.role}</Badge>}
                          {node.profile && <Badge>{node.profile}</Badge>}
                        </div>
                        <div className="mt-2 text-[0.7rem] text-midground/55 normal-case">
                          Parents: {node.parents.length ? node.parents.join(", ") : "—"} · Children: {node.children.length ? node.children.join(", ") : "—"}
                        </div>
                      </button>
                    ))}
                  </section>

                  <section className="grid gap-2 border-t border-current/10 pt-3">
                    <h3 className="text-xs tracking-[0.12em] text-midground/60">Edges</h3>
                    {dag.edges.length === 0 ? (
                      <p className="text-xs text-midground/45 normal-case">No dependencies.</p>
                    ) : (
                      dag.edges.map((edge) => (
                        <div key={`${edge.source}:${edge.target}:${edge.kind}`} className="text-xs text-midground/60">
                          {edge.source} → {edge.target} <span className="text-midground/35">({edge.kind})</span>
                        </div>
                      ))
                    )}
                  </section>
                </>
              ) : (
                <p className="text-sm text-midground/60 normal-case">Select a workflow to inspect its DAG.</p>
              )}
            </CardContent>
          </Card>

          <Card className="min-h-0">
            <CardHeader>
              <CardTitle>{selectedNode ? nodeTitle(selectedNode) : "Node detail"}</CardTitle>
            </CardHeader>
            <CardContent className="max-h-[calc(100vh-14rem)] overflow-auto text-xs text-midground/70 normal-case">
              {loadingNode ? (
                <div className="flex justify-center py-12"><Spinner /></div>
              ) : nodeDetail ? (
                <div className="flex flex-col gap-4">
                  <Metric label="Status" value={nodeDetail.node.status} />
                  <Metric label="Kanban task" value={nodeDetail.node.kanbanTaskId ?? "—"} />
                  <Metric label="Workspace" value={nodeDetail.node.workspace?.worktreePath ?? "—"} />
                  <DetailList title="Gates" items={nodeDetail.gates.map((gate) => `${gate.level}: ${gate.status}`)} />
                  <DetailList title="Artifacts" items={nodeDetail.artifacts.map((artifact) => `${artifact.kind}: ${artifact.id}`)} />
                  <DetailList title="Events" items={nodeDetail.events.map((event) => `${formatTime(event.createdAt)} · ${event.eventType}`)} />
                  {Boolean(nodeDetail.node.definitionOfDone) && (
                    <pre className="whitespace-pre-wrap rounded border border-current/10 p-2 text-[0.7rem]">
                      {formatJson(nodeDetail.node.definitionOfDone)}
                    </pre>
                  )}
                </div>
              ) : (
                <p className="text-sm text-midground/60 normal-case">Select a node to inspect gates, artifacts, and events.</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-current/10 p-2">
      <div className="text-[0.6rem] tracking-[0.12em] text-midground/40 uppercase">{label}</div>
      <div className="mt-1 truncate text-xs text-midground">{value}</div>
    </div>
  );
}

function DetailList({ title, items }: { title: string; items: string[] }) {
  return (
    <section>
      <h3 className="text-[0.65rem] tracking-[0.12em] text-midground/45 uppercase">{title}</h3>
      {items.length === 0 ? (
        <p className="mt-1 text-midground/40">—</p>
      ) : (
        <ul className="mt-1 flex flex-col gap-1">
          {items.map((item, index) => (
            <li key={`${title}:${index}`} className="rounded bg-midground/5 px-2 py-1">
              {item}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
