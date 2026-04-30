import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  CircleStop,
  Folder,
  GitBranch,
  MessageSquarePlus,
  Play,
  Plus,
  Send,
  Terminal,
  Wrench,
} from "lucide-react";
import { api } from "@/lib/api";
import type { DeskRunEvent, ProjectInfo, SessionInfo } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Toast } from "@/components/Toast";
import { useToast } from "@/hooks/useToast";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { timeAgo } from "@/lib/utils";

interface LocalMessage {
  role: "user" | "assistant";
  content: string;
}

interface TimelineItem {
  id: string;
  label: string;
  detail?: string;
  tone: "normal" | "success" | "warning" | "error";
}

function projectLabel(project?: ProjectInfo | null) {
  if (!project) return "No project";
  return project.name || project.path;
}

export default function DeskPage() {
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [projectId, setProjectId] = useState<string>("");
  const [newProjectPath, setNewProjectPath] = useState("");
  const [newProjectName, setNewProjectName] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState("");
  const { toast, showToast } = useToast();
  const scrollRef = useRef<HTMLDivElement>(null);

  const activeProject = useMemo(
    () => projects.find((p) => p.id === projectId) ?? null,
    [projects, projectId],
  );

  const loadData = () => {
    api.getProjects()
      .then((resp) => {
        setProjects(resp.projects);
        if (!projectId && resp.projects.length > 0) setProjectId(resp.projects[0].id);
      })
      .catch(() => showToast("Failed to load projects", "error"));
    api.getSessions()
      .then((rows) => setSessions(rows.filter((s) => s.source === "desk")))
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, timeline]);

  const addTimeline = (item: Omit<TimelineItem, "id">) => {
    setTimeline((prev) => [...prev, { ...item, id: `${Date.now()}-${prev.length}` }]);
  };

  const handleCreateProject = async () => {
    if (!newProjectPath.trim()) return;
    try {
      const project = await api.createProject({
        path: newProjectPath.trim(),
        name: newProjectName.trim() || undefined,
      });
      setProjects((prev) => [project, ...prev.filter((p) => p.id !== project.id)]);
      setProjectId(project.id);
      setNewProjectPath("");
      setNewProjectName("");
      showToast("Project added", "success");
    } catch (err) {
      showToast(`Project failed: ${err}`, "error");
    }
  };

  const handleNewSession = async () => {
    try {
      const resp = await api.createDeskSession({
        project_id: activeProject?.id ?? null,
        title: activeProject ? `${activeProject.name} Desk` : "Desk Session",
      });
      setSessionId(resp.session_id);
      setMessages([]);
      setTimeline([{
        id: "session-started",
        label: "Session ready",
        detail: resp.session_id,
        tone: "success",
      }]);
      loadData();
    } catch (err) {
      showToast(`Session failed: ${err}`, "error");
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || running) return;

    let currentSession = sessionId;
    if (!currentSession) {
      const resp = await api.createDeskSession({
        project_id: activeProject?.id ?? null,
        title: activeProject ? `${activeProject.name} Desk` : "Desk Session",
      });
      currentSession = resp.session_id;
      setSessionId(currentSession);
    }

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setRunning(true);
    setTimeline((prev) => [...prev, {
      id: `run-${Date.now()}`,
      label: "Run started",
      detail: currentSession,
      tone: "normal",
    }]);

    try {
      const resp = await api.startDeskRun({ session_id: currentSession, message: text });
      setActiveRunId(resp.run_id);
      const events = new EventSource(`/api/desk/runs/${encodeURIComponent(resp.run_id)}/events`);

      events.onmessage = (event) => {
        const data = JSON.parse(event.data) as DeskRunEvent;
        if (data.event === "message.delta") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              next[next.length - 1] = { ...last, content: last.content + data.delta };
            }
            return next;
          });
        } else if (data.event === "tool.started") {
          addTimeline({
            label: `Tool started: ${data.tool || "unknown"}`,
            detail: data.preview,
            tone: "warning",
          });
        } else if (data.event === "tool.completed") {
          addTimeline({
            label: `Tool completed: ${data.tool || "unknown"}`,
            detail: `${data.duration ?? 0}s`,
            tone: data.error ? "error" : "success",
          });
        } else if (data.event === "reasoning.available") {
          addTimeline({ label: "Reasoning available", detail: data.text.slice(0, 160), tone: "normal" });
        } else if (data.event === "run.completed") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && !last.content && data.output) {
              next[next.length - 1] = { ...last, content: data.output };
            }
            return next;
          });
          addTimeline({ label: "Run completed", tone: "success" });
          setRunning(false);
          setActiveRunId("");
          events.close();
          loadData();
        } else if (data.event === "run.failed") {
          addTimeline({ label: "Run failed", detail: data.error, tone: "error" });
          setRunning(false);
          setActiveRunId("");
          events.close();
        }
      };

      events.onerror = () => {
        addTimeline({ label: "Event stream closed", tone: "warning" });
        setRunning(false);
        setActiveRunId("");
        events.close();
      };
    } catch (err) {
      showToast(`Run failed: ${err}`, "error");
      setRunning(false);
      setActiveRunId("");
    }
  };

  const handleInterrupt = async () => {
    if (!activeRunId) return;
    await api.interruptDeskRun(activeRunId).catch(() => {});
    addTimeline({ label: "Interrupt requested", tone: "warning" });
  };

  return (
    <div className="grid min-h-[calc(100vh-9rem)] gap-4 lg:grid-cols-[280px_minmax(0,1fr)_320px]">
      <Toast toast={toast} />

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Folder className="h-4 w-4" />
            Projects
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Input
              placeholder="Project name"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
            />
            <Input
              placeholder="/absolute/project/path"
              value={newProjectPath}
              onChange={(e) => setNewProjectPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreateProject();
              }}
            />
            <Button variant="outline" size="sm" onClick={handleCreateProject}>
              <Plus className="h-3.5 w-3.5" />
              Add
            </Button>
          </div>

          <div className="flex flex-col gap-2">
            {projects.length === 0 && (
              <div className="border border-border p-3 text-xs text-muted-foreground">
                Add a local folder to bind Desk runs to a workspace.
              </div>
            )}
            {projects.map((project) => (
              <button
                key={project.id}
                type="button"
                onClick={() => setProjectId(project.id)}
                className={`border p-3 text-left transition-colors hover:bg-secondary/40 ${
                  projectId === project.id ? "border-foreground/40 bg-foreground/10" : "border-border"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm">{projectLabel(project)}</span>
                  {project.metadata?.git_branch && <GitBranch className="h-3.5 w-3.5 text-muted-foreground" />}
                </div>
                <div className="mt-1 truncate font-courier text-[11px] text-muted-foreground">{project.path}</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {project.metadata?.git_branch && <Badge variant="outline">{project.metadata.git_branch}</Badge>}
                  {project.metadata?.git_dirty && <Badge variant="warning">dirty</Badge>}
                  {project.metadata?.context_files?.slice(0, 2).map((name) => (
                    <Badge key={name} variant="secondary">{name}</Badge>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="flex min-h-[640px] flex-col overflow-hidden">
        <CardHeader className="flex-row items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              Desk
            </CardTitle>
            <p className="mt-1 truncate font-courier text-xs text-muted-foreground">
              {activeProject ? activeProject.path : "No project cwd selected"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {sessionId && <Badge variant="outline">{sessionId.slice(0, 14)}</Badge>}
            <Button variant="outline" size="sm" onClick={handleNewSession}>
              <MessageSquarePlus className="h-3.5 w-3.5" />
              New
            </Button>
          </div>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
          <div className="min-h-0 flex-1 overflow-y-auto border border-border bg-background/30 p-4">
            {messages.length === 0 ? (
              <div className="flex h-full min-h-[420px] items-center justify-center text-center text-sm text-muted-foreground">
                <div>
                  <Play className="mx-auto mb-3 h-5 w-5 opacity-60" />
                  Start a Desk session in {projectLabel(activeProject)}.
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`max-w-[88%] border p-3 ${
                      msg.role === "user"
                        ? "ml-auto border-foreground/20 bg-foreground/10"
                        : "mr-auto border-border bg-card/70"
                    }`}
                  >
                    <div className="mb-1 font-compressed text-[0.7rem] tracking-[0.15em] uppercase text-muted-foreground">
                      {msg.role}
                    </div>
                    {msg.content ? (
                      msg.role === "assistant" ? <Markdown content={msg.content} /> : <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                    ) : (
                      <div className="text-sm text-muted-foreground">Streaming...</div>
                    )}
                  </div>
                ))}
                <div ref={scrollRef} />
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask Hermes to work in this project..."
              disabled={running}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) handleSend();
              }}
            />
            {running ? (
              <Button variant="destructive" onClick={handleInterrupt}>
                <CircleStop className="h-4 w-4" />
                Stop
              </Button>
            ) : (
              <Button onClick={handleSend} disabled={!input.trim()}>
                <Send className="h-4 w-4" />
                Send
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wrench className="h-4 w-4" />
            Run Timeline
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="border border-border p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Terminal className="h-3.5 w-3.5" />
              cwd
            </div>
            <div className="mt-1 break-all font-courier text-xs">
              {activeProject?.path || "default"}
            </div>
          </div>

          {timeline.length === 0 ? (
            <div className="border border-border p-3 text-xs text-muted-foreground">
              Tool calls and run events appear here.
            </div>
          ) : (
            timeline.map((item) => (
              <div key={item.id} className="border border-border p-3">
                <div className={`text-xs ${
                  item.tone === "error" ? "text-destructive" :
                  item.tone === "success" ? "text-success" :
                  item.tone === "warning" ? "text-warning" :
                  "text-foreground"
                }`}>
                  {item.label}
                </div>
                {item.detail && (
                  <div className="mt-1 break-words font-courier text-[11px] text-muted-foreground">
                    {item.detail}
                  </div>
                )}
              </div>
            ))
          )}

          <div className="mt-2">
            <div className="mb-2 text-xs text-muted-foreground">Recent Desk sessions</div>
            <div className="flex max-h-52 flex-col gap-2 overflow-y-auto">
              {sessions.slice(0, 8).map((session) => (
                <button
                  key={session.id}
                  type="button"
                  className="border border-border p-2 text-left hover:bg-secondary/30"
                  onClick={() => {
                    setSessionId(session.id);
                    setMessages([]);
                    setTimeline([{ id: "resumed", label: "Session selected", detail: session.id, tone: "normal" }]);
                  }}
                >
                  <div className="truncate text-xs">{session.title || session.preview || session.id}</div>
                  <div className="mt-1 text-[10px] text-muted-foreground">{timeAgo(session.last_active)}</div>
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
