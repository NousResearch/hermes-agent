import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AlertTriangle,
  Bug,
  CheckCircle2,
  ExternalLink,
  GitBranch,
  GitCommit,
  Github,
  RefreshCw,
  Send,
  UploadCloud,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Toast } from "@/components/Toast";
import { api, type ProjectInfo, type ProjectsResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/useToast";
import { usePageHeader } from "@/contexts/usePageHeader";

const KIND_OPTIONS = ["issue", "bug", "feature", "task"];
const SEVERITY_OPTIONS = ["low", "normal", "high", "critical"];

function formatDate(value: string | null): string {
  if (!value) return "No commits yet";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function labelsFromText(value: string): string[] {
  return value
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);
}

export default function ProjectsPage() {
  const [payload, setPayload] = useState<ProjectsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [kind, setKind] = useState("bug");
  const [severity, setSeverity] = useState("normal");
  const [labels, setLabels] = useState("");
  const [parentTaskIds, setParentTaskIds] = useState("");
  const [repoUrl, setRepoUrl] = useState("");
  const [repoBranch, setRepoBranch] = useState("");
  const [importing, setImporting] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  const load = useCallback(async () => {
    const next = await api.getProjects();
    setPayload(next);
    setSelectedId((current) => {
      if (current && next.projects.some((project) => project.id === current)) return current;
      return next.projects[0]?.id ?? null;
    });
  }, []);

  useEffect(() => {
    // Initial API synchronization for the Projects dashboard.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load()
      .catch((error) => showToast(error instanceof Error ? error.message : "Failed to load projects", "error"))
      .finally(() => setLoading(false));
  }, [load, showToast]);

  useEffect(() => {
    setEnd(
      <Button
        ghost
        size="sm"
        className="w-max gap-2"
        disabled={loading}
        onClick={() => {
          setLoading(true);
          void load().finally(() => setLoading(false));
        }}
      >
        {loading ? <Spinner /> : <RefreshCw className="h-3.5 w-3.5" />}
        Refresh projects
      </Button>,
    );
    return () => setEnd(null);
  }, [load, loading, setEnd]);

  const projects = useMemo(() => payload?.projects ?? [], [payload]);
  const filteredProjects = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return projects;
    return projects.filter((project) =>
      [project.name, project.path, project.remote_url ?? "", project.branch ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [projects, query]);

  const selectedProject = projects.find((project) => project.id === selectedId) ?? filteredProjects[0] ?? null;

  const importRepo = async () => {
    const trimmed = repoUrl.trim();
    if (!trimmed) {
      showToast("Paste a repository URL first", "error");
      return;
    }
    setImporting(true);
    try {
      const result = await api.importProject({ repo_url: trimmed, branch: repoBranch.trim() || undefined });
      setRepoUrl("");
      setRepoBranch("");
      showToast(result.message || "Repository added", "success");
      await load();
      setSelectedId(result.project.id);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Failed to import repository", "error");
    } finally {
      setImporting(false);
    }
  };

  const runSourceControl = async (action: "fetch" | "pull" | "push" | "checkout" | "create_branch", branch?: string, createFrom?: string) => {
    if (!selectedProject) return;
    try {
      const result = await api.projectSourceControl(selectedProject.id, { action, branch, create_from: createFrom });
      showToast(`${action.replace("_", " ")} complete`, "success");
      await load();
      setSelectedId(result.project.id);
    } catch (error) {
      showToast(error instanceof Error ? error.message : `Failed to ${action}`, "error");
    }
  };

  const submitIssue = async () => {
    if (!selectedProject) return;
    const trimmed = title.trim();
    if (!trimmed) {
      showToast("Add a title first", "error");
      return;
    }
    setSubmitting(true);
    try {
      const result = await api.createProjectIssue(selectedProject.id, {
        title: trimmed,
        body,
        kind,
        severity,
        labels: labelsFromText(labels),
        parent_task_ids: labelsFromText(parentTaskIds),
      });
      setTitle("");
      setBody("");
      setLabels("");
      setParentTaskIds("");
      showToast(result.issue.todo?.task_id ? `Logged locally and created todo ${result.issue.todo.task_id}` : "Logged locally in Hermes", "success");
      if (result.issue.github_new_issue_url) {
        window.open(result.issue.github_new_issue_url, "_blank", "noopener,noreferrer");
      }
      await load();
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Failed to log issue", "error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Toast toast={toast} />

      <div className="grid gap-4 xl:grid-cols-[minmax(320px,420px)_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Projects</CardTitle>
            <p className="text-[0.7rem] tracking-[0.08em] text-midground/55 normal-case">
              Local git repositories Hermes can see. Scan roots: {payload?.roots.join(", ") || "loading..."}
            </p>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="rounded-lg border border-midground/15 p-3">
              <div className="mb-2 text-xs uppercase tracking-[0.14em] text-midground/45">Add repository</div>
              <div className="grid gap-2">
                <Input value={repoUrl} placeholder="https://github.com/owner/repo.git or git@github.com:owner/repo.git" onChange={(event) => setRepoUrl(event.target.value)} />
                <div className="flex gap-2">
                  <Input value={repoBranch} placeholder="Optional branch" onChange={(event) => setRepoBranch(event.target.value)} />
                  <Button className="w-max gap-2" disabled={importing} onClick={() => void importRepo()}>
                    {importing ? <Spinner /> : <UploadCloud className="h-3.5 w-3.5" />}
                    Add
                  </Button>
                </div>
              </div>
              <p className="mt-2 text-xs text-midground/50">Hermes clones/fetches into the managed local project root so agents can edit and push branches quickly.</p>
            </div>

            <Input
              value={query}
              placeholder="Filter by name, path, branch, or remote..."
              onChange={(event) => setQuery(event.target.value)}
            />

            {loading && projects.length === 0 ? (
              <div className="flex items-center gap-2 text-sm text-midground/70">
                <Spinner /> Scanning projects...
              </div>
            ) : filteredProjects.length === 0 ? (
              <div className="rounded-md border border-midground/15 p-4 text-sm text-midground/70">
                No git projects matched this filter.
              </div>
            ) : (
              <div className="flex max-h-[66vh] flex-col gap-2 overflow-auto pr-1">
                {filteredProjects.map((project) => (
                  <button
                    key={project.id}
                    type="button"
                    onClick={() => setSelectedId(project.id)}
                    className={cn(
                      "rounded-lg border p-3 text-left transition-colors",
                      selectedProject?.id === project.id
                        ? "border-midground/45 bg-midground/10"
                        : "border-midground/15 hover:bg-midground/5",
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate font-mondwest text-[0.95rem] tracking-[0.1em]">
                          {project.name}
                        </div>
                        <div className="mt-1 truncate text-xs text-midground/55">{project.path}</div>
                      </div>
                      {project.dirty ? (
                        <Badge tone="secondary" className="gap-1">
                          <AlertTriangle className="h-3 w-3" /> dirty
                        </Badge>
                      ) : (
                        <Badge tone="secondary" className="gap-1">
                          <CheckCircle2 className="h-3 w-3" /> clean
                        </Badge>
                      )}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-midground/70">
                      {project.branch && (
                        <span className="inline-flex items-center gap-1">
                          <GitBranch className="h-3 w-3" /> {project.branch}
                        </span>
                      )}
                      <span className="inline-flex items-center gap-1">
                        <Bug className="h-3 w-3" /> {project.issue_counts.open} open
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {selectedProject ? (
          <ProjectDetail
            project={selectedProject}
            title={title}
            body={body}
            kind={kind}
            severity={severity}
            labels={labels}
            submitting={submitting}
            setTitle={setTitle}
            setBody={setBody}
            setKind={setKind}
            setSeverity={setSeverity}
            setLabels={setLabels}
            parentTaskIds={parentTaskIds}
            setParentTaskIds={setParentTaskIds}
            runSourceControl={runSourceControl}
            submitIssue={submitIssue}
          />
        ) : (
          <Card>
            <CardContent className="p-6 text-sm text-midground/70">Select a project to inspect it.</CardContent>
          </Card>
        )}
      </div>

      {payload && (
        <p className="text-xs text-midground/45">
          Local dashboard issue store: {payload.issue_store_path}. These notes are available for later Hermes sessions; GitHub links open separately when available.
        </p>
      )}
    </div>
  );
}

function ProjectDetail(props: {
  project: ProjectInfo;
  title: string;
  body: string;
  kind: string;
  severity: string;
  labels: string;
  parentTaskIds: string;
  submitting: boolean;
  setTitle: (value: string) => void;
  setBody: (value: string) => void;
  setKind: (value: string) => void;
  setSeverity: (value: string) => void;
  setLabels: (value: string) => void;
  setParentTaskIds: (value: string) => void;
  runSourceControl: (action: "fetch" | "pull" | "push" | "checkout" | "create_branch", branch?: string, createFrom?: string) => Promise<void>;
  submitIssue: () => void;
}) {
  const { project } = props;
  const [branchName, setBranchName] = useState("");
  const [newBranchName, setNewBranchName] = useState("");
  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <CardTitle>{project.name}</CardTitle>
              <p className="mt-1 break-all text-sm text-midground/60">{project.path}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {project.repo_url && <ProjectLink href={project.repo_url} label="Repo" icon={<Github className="h-3.5 w-3.5" />} />}
              {project.issues_url && <ProjectLink href={project.issues_url} label="Issues" />}
              {project.releases_url && <ProjectLink href={project.releases_url} label="Releases" />}
            </div>
          </div>
        </CardHeader>
        <CardContent className="grid gap-4 lg:grid-cols-3">
          <InfoTile label="Branch" value={project.branch ?? "unknown"} icon={<GitBranch className="h-4 w-4" />} />
          <InfoTile label="Sync" value={`${project.ahead} ahead / ${project.behind} behind${project.upstream ? ` vs ${project.upstream}` : ""}`} />
          <InfoTile label="Working tree" value={project.dirty ? "Uncommitted changes" : "Clean"} icon={project.dirty ? <AlertTriangle className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />} />
          <InfoTile label="Latest commit" value={[project.latest_commit, project.latest_commit_message].filter(Boolean).join(" — ") || "unknown"} icon={<GitCommit className="h-4 w-4" />} />
          <InfoTile label="Last commit time" value={formatDate(project.last_commit_at)} />
          <InfoTile label="Open local notes" value={`${project.issue_counts.open} open / ${project.issue_counts.total} total`} />
          <InfoTile label="Bug reports" value={`${project.issue_counts.bugs}`} icon={<Bug className="h-4 w-4" />} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Source control</CardTitle>
          <p className="text-[0.7rem] tracking-[0.08em] text-midground/55 normal-case">
            Fetch, pull, push, checkout, or create working branches for this local copy.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            <Button ghost size="sm" onClick={() => void props.runSourceControl("fetch")}>Fetch</Button>
            <Button ghost size="sm" onClick={() => void props.runSourceControl("pull")}>Pull ff-only</Button>
            <Button ghost size="sm" onClick={() => void props.runSourceControl("push")}>Push branch</Button>
          </div>
          <div className="grid gap-2 lg:grid-cols-[1fr_auto]">
            <select className="rounded-md border border-midground/20 bg-background px-3 py-2 text-sm" value={branchName} onChange={(event) => setBranchName(event.target.value)}>
              <option value="">Select existing branch...</option>
              {project.branches.map((branch) => <option key={branch} value={branch}>{branch}</option>)}
            </select>
            <Button ghost size="sm" disabled={!branchName} onClick={() => void props.runSourceControl("checkout", branchName)}>Checkout</Button>
          </div>
          <div className="grid gap-2 lg:grid-cols-[1fr_auto]">
            <Input value={newBranchName} onChange={(event) => setNewBranchName(event.target.value)} placeholder="new branch, e.g. feature/project-import" />
            <Button ghost size="sm" disabled={!newBranchName.trim()} onClick={() => void props.runSourceControl("create_branch", newBranchName.trim())}>Create branch</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Log issue or bug report</CardTitle>
          <p className="text-[0.7rem] tracking-[0.08em] text-midground/55 normal-case">
            Saves a local Hermes note, creates a Kanban todo with relevant skills, and opens a prefilled GitHub issue page when available.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid gap-4 lg:grid-cols-[1fr_150px_150px]">
            <div className="grid gap-2">
              <Label htmlFor="project-issue-title">Title</Label>
              <Input id="project-issue-title" value={props.title} onChange={(event) => props.setTitle(event.target.value)} placeholder="Short bug/feature/task title" />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="project-issue-kind">Type</Label>
              <select id="project-issue-kind" className="rounded-md border border-midground/20 bg-background px-3 py-2 text-sm" value={props.kind} onChange={(event) => props.setKind(event.target.value)}>
                {KIND_OPTIONS.map((option) => <option key={option} value={option}>{option}</option>)}
              </select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="project-issue-severity">Severity</Label>
              <select id="project-issue-severity" className="rounded-md border border-midground/20 bg-background px-3 py-2 text-sm" value={props.severity} onChange={(event) => props.setSeverity(event.target.value)}>
                {SEVERITY_OPTIONS.map((option) => <option key={option} value={option}>{option}</option>)}
              </select>
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="project-issue-body">Details</Label>
            <textarea
              id="project-issue-body"
              className="min-h-28 rounded-md border border-midground/20 bg-background px-3 py-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-midground"
              value={props.body}
              onChange={(event) => props.setBody(event.target.value)}
              placeholder="Steps to reproduce, expected behavior, screenshots to collect later, acceptance criteria, etc."
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="project-issue-labels">Extra labels</Label>
            <Input id="project-issue-labels" value={props.labels} onChange={(event) => props.setLabels(event.target.value)} placeholder="comma, separated, labels" />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="project-issue-parents">Parent Kanban task ids</Label>
            <Input id="project-issue-parents" value={props.parentTaskIds} onChange={(event) => props.setParentTaskIds(event.target.value)} placeholder="optional parent task ids, comma separated" />
          </div>
          <Button className="w-fit gap-2" disabled={props.submitting} onClick={() => props.submitIssue()}>
            {props.submitting ? <Spinner /> : <Send className="h-3.5 w-3.5" />}
            Save issue
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent local notes</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {project.issues.length === 0 ? (
            <div className="rounded-md border border-midground/15 p-4 text-sm text-midground/70">No local issues logged yet.</div>
          ) : (
            project.issues.map((issue) => (
              <div key={issue.id} className="rounded-lg border border-midground/15 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="secondary">{issue.kind}</Badge>
                  <Badge tone="secondary">{issue.severity}</Badge>
                  <span className="text-xs text-midground/45">{formatDate(issue.created_at)}</span>
                </div>
                <div className="mt-2 font-medium">{issue.title}</div>
                {issue.body && <p className="mt-1 whitespace-pre-wrap text-sm text-midground/65">{issue.body}</p>}
                {issue.todo && (
                  <div className="mt-2 rounded-md bg-midground/5 p-2 text-xs text-midground/65">
                    {issue.todo.task_id ? <>Todo: {issue.todo.task_id} assigned to {issue.todo.assignee ?? "default"}</> : <>Todo creation warning: {issue.todo.error}</>}
                    {issue.todo.skills && <div className="mt-1">Skills: {issue.todo.skills.join(", ")}</div>}
                    {issue.todo.recommended_branch && <div className="mt-1">Suggested branch: {issue.todo.recommended_branch}</div>}
                  </div>
                )}
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ProjectLink({ href, label, icon }: { href: string; label: string; icon?: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 rounded-md border border-midground/20 px-3 py-1.5 text-sm hover:bg-midground/10">
      {icon}
      {label}
      <ExternalLink className="h-3.5 w-3.5" />
    </a>
  );
}

function InfoTile({ label, value, icon }: { label: string; value: string; icon?: ReactNode }) {
  return (
    <div className="rounded-lg border border-midground/15 p-3">
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-midground/45">
        {icon}
        {label}
      </div>
      <div className="mt-2 break-words text-sm text-midground/85">{value}</div>
    </div>
  );
}
