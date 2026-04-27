/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { codeApi } from "@/lib/codeApi";
import type {
  CodeWorkspace,
  GitStatus,
  GitDiff,
  CodeSession,
  CodeSessionEvent,
  CodeCommand,
  CodeArtifact,
  DiagnosticsResult,
  AgentFlow,
  SkillInfo,
  SkillRun,
  Approval,
} from "@/types/code";

interface CodeWorkspaceContextValue {
  workspaces: CodeWorkspace[];
  selectedWorkspaceId: string | null;
  gitStatus: Record<string, GitStatus>;
  gitDiff: Record<string, GitDiff>;
  loading: boolean;
  error: string | null;
  fetchWorkspaces: () => Promise<void>;
  selectWorkspace: (id: string | null) => void;
  refreshWorkspace: (id: string) => Promise<void>;
  openWorkspace: (path: string) => Promise<CodeWorkspace>;
  fetchGitStatus: (workspaceId: string, codeSessionId?: string) => Promise<void>;
  fetchGitDiff: (workspaceId: string, path?: string) => Promise<void>;
}

const CodeWorkspaceContext = createContext<CodeWorkspaceContextValue | null>(null);

export function CodeWorkspaceProvider({ children }: { children: ReactNode }) {
  const [workspaces, setWorkspaces] = useState<CodeWorkspace[]>([]);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string | null>(null);
  const [gitStatus, setGitStatus] = useState<Record<string, GitStatus>>({});
  const [gitDiff, setGitDiff] = useState<Record<string, GitDiff>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkspaces = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getWorkspaces();
      setWorkspaces(data.workspaces);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const selectWorkspace = useCallback((id: string | null) => {
    setSelectedWorkspaceId(id);
  }, []);

  const refreshWorkspace = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.refreshWorkspace(id);
      setWorkspaces((prev) => prev.map((w) => (w.id === id ? data.workspace : w)));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const openWorkspace = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.openWorkspace(path);
      setWorkspaces((prev) => [...prev, data.workspace]);
      setSelectedWorkspaceId(data.workspace.id);
      return data.workspace;
    } catch (err) {
      setError(String(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchGitStatus = useCallback(async (workspaceId: string, codeSessionId?: string) => {
    try {
      const data = await codeApi.getGitStatus(workspaceId, codeSessionId);
      setGitStatus((prev) => ({ ...prev, [workspaceId]: data.status }));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const fetchGitDiff = useCallback(async (workspaceId: string, path?: string) => {
    try {
      const data = await codeApi.getGitDiff(workspaceId, path);
      setGitDiff((prev) => ({ ...prev, [workspaceId]: data.diff }));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  return (
    <CodeWorkspaceContext.Provider
      value={{
        workspaces,
        selectedWorkspaceId,
        gitStatus,
        gitDiff,
        loading,
        error,
        fetchWorkspaces,
        selectWorkspace,
        refreshWorkspace,
        openWorkspace,
        fetchGitStatus,
        fetchGitDiff,
      }}
    >
      {children}
    </CodeWorkspaceContext.Provider>
  );
}

export function useCodeWorkspaceContext() {
  const ctx = useContext(CodeWorkspaceContext);
  if (!ctx) throw new Error("useCodeWorkspaceContext must be used within CodeWorkspaceProvider");
  return ctx;
}

interface CodeSessionContextValue {
  sessions: CodeSession[];
  selectedSessionId: string | null;
  currentEvents: CodeSessionEvent[];
  commands: Record<string, CodeCommand[]>;
  artifacts: Record<string, CodeArtifact[]>;
  loading: boolean;
  error: string | null;
  fetchSessions: (params?: { workspace_id?: string; status?: string; limit?: number }) => Promise<void>;
  selectSession: (id: string | null) => void;
  fetchSession: (id: string) => Promise<void>;
  createSession: (payload: {
    workspace_id: string;
    hermes_session_id?: string;
    task_id?: string;
    title?: string;
    provider?: string;
    model?: string;
  }) => Promise<CodeSession>;
  cancelSession: (id: string, reason?: string) => Promise<void>;
  resumeSession: (id: string) => Promise<void>;
  fetchEvents: (sessionId: string) => Promise<void>;
  fetchCommands: (sessionId: string) => Promise<void>;
  runCommand: (
    sessionId: string,
    payload: { command: string; cwd?: string; timeout_seconds?: number }
  ) => Promise<CodeCommand>;
  cancelCommand: (commandId: string) => Promise<void>;
  fetchArtifacts: (sessionId: string) => Promise<void>;
  appendCommandOutput: (commandId: string, chunk: { stdout?: string; stderr?: string }) => void;
}

const CodeSessionContext = createContext<CodeSessionContextValue | null>(null);

export function CodeSessionProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<CodeSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [currentEvents, setCurrentEvents] = useState<CodeSessionEvent[]>([]);
  const [commands, setCommands] = useState<Record<string, CodeCommand[]>>({});
  const [artifacts, setArtifacts] = useState<Record<string, CodeArtifact[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = useCallback(async (params?: { workspace_id?: string; status?: string; limit?: number }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getCodeSessions(params);
      setSessions(data.sessions);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const selectSession = useCallback((id: string | null) => {
    setSelectedSessionId(id);
  }, []);

  const fetchSession = useCallback(async (id: string) => {
    try {
      const data = await codeApi.getCodeSession(id);
      setSessions((prev) => prev.map((s) => (s.id === id ? data.code_session : s)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const createSession = useCallback(async (payload: {
    workspace_id: string;
    hermes_session_id?: string;
    task_id?: string;
    title?: string;
    provider?: string;
    model?: string;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.createCodeSession(payload);
      setSessions((prev) => [...prev, data.code_session]);
      setSelectedSessionId(data.code_session.id);
      return data.code_session;
    } catch (err) {
      setError(String(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const cancelSession = useCallback(async (id: string, reason?: string) => {
    try {
      await codeApi.cancelCodeSession(id, reason);
      setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, status: "cancelled" as const } : s)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const resumeSession = useCallback(async (id: string) => {
    try {
      const data = await codeApi.updateCodeSession(id, { status: "running" });
      setSessions((prev) => prev.map((s) => (s.id === id ? data.code_session : s)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const fetchEvents = useCallback(async (sessionId: string) => {
    try {
      const data = await codeApi.getCodeSessionEvents(sessionId);
      setCurrentEvents(data.events);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const fetchCommands = useCallback(async (sessionId: string) => {
    try {
      const data = await codeApi.getCommands(sessionId);
      setCommands((prev) => ({ ...prev, [sessionId]: data.commands }));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const runCommand = useCallback(async (
    sessionId: string,
    payload: { command: string; cwd?: string; timeout_seconds?: number }
  ) => {
    try {
      const data = await codeApi.runCommand(sessionId, payload);
      setCommands((prev) => ({
        ...prev,
        [sessionId]: [...(prev[sessionId] || []), data.command],
      }));
      return data.command;
    } catch (err) {
      setError(String(err));
      throw err;
    }
  }, []);

  const cancelCommand = useCallback(async (commandId: string) => {
    try {
      await codeApi.cancelCommand(commandId);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const fetchArtifacts = useCallback(async (sessionId: string) => {
    try {
      const data = await codeApi.getCodeSessionArtifacts(sessionId);
      setArtifacts((prev) => ({ ...prev, [sessionId]: data.artifacts }));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const appendCommandOutput = useCallback((commandId: string, chunk: { stdout?: string; stderr?: string }) => {
    setCommands((prev) => {
      const newCommands = { ...prev };
      for (const sessionId of Object.keys(newCommands)) {
        newCommands[sessionId] = newCommands[sessionId].map((cmd) => {
          if (cmd.id === commandId) {
            return {
              ...cmd,
              stdout: chunk.stdout ? cmd.stdout + chunk.stdout : cmd.stdout,
              stderr: chunk.stderr ? cmd.stderr + chunk.stderr : cmd.stderr,
            };
          }
          return cmd;
        });
      }
      return newCommands;
    });
  }, []);

  return (
    <CodeSessionContext.Provider
      value={{
        sessions,
        selectedSessionId,
        currentEvents,
        commands,
        artifacts,
        loading,
        error,
        fetchSessions,
        selectSession,
        fetchSession,
        createSession,
        cancelSession,
        resumeSession,
        fetchEvents,
        fetchCommands,
        runCommand,
        cancelCommand,
        fetchArtifacts,
        appendCommandOutput,
      }}
    >
      {children}
    </CodeSessionContext.Provider>
  );
}

export function useCodeSessionContext() {
  const ctx = useContext(CodeSessionContext);
  if (!ctx) throw new Error("useCodeSessionContext must be used within CodeSessionProvider");
  return ctx;
}

interface DiagnosticsContextValue {
  diagnostics: Record<string, DiagnosticsResult>;
  loading: boolean;
  error: string | null;
  fetchDiagnostics: (workspaceId: string, codeSessionId?: string) => Promise<void>;
  runDiagnostics: (workspaceId: string, codeSessionId?: string) => Promise<void>;
  restartLanguageServices: (workspaceId: string) => Promise<void>;
}

const DiagnosticsContext = createContext<DiagnosticsContextValue | null>(null);

export function DiagnosticsProvider({ children }: { children: ReactNode }) {
  const [diagnostics, setDiagnostics] = useState<Record<string, DiagnosticsResult>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDiagnostics = useCallback(async (workspaceId: string, codeSessionId?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getDiagnostics(workspaceId, codeSessionId);
      setDiagnostics((prev) => ({ ...prev, [workspaceId]: data.result }));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const runDiagnostics = useCallback(async (workspaceId: string, codeSessionId?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getDiagnostics(workspaceId, codeSessionId);
      setDiagnostics((prev) => ({ ...prev, [workspaceId]: data.result }));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const restartLanguageServices = useCallback(async (workspaceId: string) => {
    try {
      await codeApi.restartLanguageServices(workspaceId);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  return (
    <DiagnosticsContext.Provider
      value={{
        diagnostics,
        loading,
        error,
        fetchDiagnostics,
        runDiagnostics,
        restartLanguageServices,
      }}
    >
      {children}
    </DiagnosticsContext.Provider>
  );
}

export function useDiagnosticsContext() {
  const ctx = useContext(DiagnosticsContext);
  if (!ctx) throw new Error("useDiagnosticsContext must be used within DiagnosticsProvider");
  return ctx;
}

interface AgentFlowContextValue {
  flows: AgentFlow[];
  selectedFlowId: string | null;
  loading: boolean;
  error: string | null;
  fetchFlows: (params?: { code_session_id?: string; workspace_id?: string; status?: string }) => Promise<void>;
  selectFlow: (id: string | null) => void;
  runFlow: (id: string) => Promise<void>;
  cancelFlow: (id: string, reason?: string) => Promise<void>;
  resumeFlow: (id: string) => Promise<void>;
}

const AgentFlowContext = createContext<AgentFlowContextValue | null>(null);

export function AgentFlowProvider({ children }: { children: ReactNode }) {
  const [flows, setFlows] = useState<AgentFlow[]>([]);
  const [selectedFlowId, setSelectedFlowId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFlows = useCallback(async (params?: { code_session_id?: string; workspace_id?: string; status?: string }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getAgentFlows(params);
      setFlows(data.flows);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const selectFlow = useCallback((id: string | null) => {
    setSelectedFlowId(id);
  }, []);

  const runFlow = useCallback(async (id: string) => {
    try {
      const data = await codeApi.runAgentFlow(id);
      setFlows((prev) => prev.map((f) => (f.id === id ? data.flow : f)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const cancelFlow = useCallback(async (id: string, reason?: string) => {
    try {
      const data = await codeApi.cancelAgentFlow(id, reason);
      setFlows((prev) => prev.map((f) => (f.id === id ? data.flow : f)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const resumeFlow = useCallback(async (id: string) => {
    try {
      const data = await codeApi.resumeAgentFlow(id);
      setFlows((prev) => prev.map((f) => (f.id === id ? data.flow : f)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  return (
    <AgentFlowContext.Provider
      value={{
        flows,
        selectedFlowId,
        loading,
        error,
        fetchFlows,
        selectFlow,
        runFlow,
        cancelFlow,
        resumeFlow,
      }}
    >
      {children}
    </AgentFlowContext.Provider>
  );
}

export function useAgentFlowContext() {
  const ctx = useContext(AgentFlowContext);
  if (!ctx) throw new Error("useAgentFlowContext must be used within AgentFlowProvider");
  return ctx;
}

interface SkillContextValue {
  skills: SkillInfo[];
  skillRuns: SkillRun[];
  loading: boolean;
  error: string | null;
  fetchSkills: () => Promise<void>;
  fetchSkillRuns: (params?: { workspace_id?: string; code_session_id?: string; skill_name?: string; status?: string }) => Promise<void>;
  runSkill: (skillName: string, payload: {
    workspace_id: string;
    code_session_id?: string;
    task_id?: string;
    input?: Record<string, unknown>;
  }) => Promise<SkillRun>;
  cancelSkillRun: (runId: string, reason?: string) => Promise<void>;
  resumeSkillRun: (runId: string) => Promise<void>;
}

const SkillContext = createContext<SkillContextValue | null>(null);

export function SkillProvider({ children }: { children: ReactNode }) {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [skillRuns, setSkillRuns] = useState<SkillRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSkills = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getSkills();
      setSkills(data.skills);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchSkillRuns = useCallback(async (params?: { workspace_id?: string; code_session_id?: string; skill_name?: string; status?: string }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getSkillRuns(params);
      setSkillRuns(data.runs);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const runSkill = useCallback(async (skillName: string, payload: {
    workspace_id: string;
    code_session_id?: string;
    task_id?: string;
    input?: Record<string, unknown>;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.runSkillShortcut(skillName, payload);
      setSkillRuns((prev) => [...prev, data.run]);
      return data.run;
    } catch (err) {
      setError(String(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const cancelSkillRun = useCallback(async (runId: string, reason?: string) => {
    try {
      const data = await codeApi.cancelSkillRun(runId, reason);
      setSkillRuns((prev) => prev.map((r) => (r.id === runId ? data.run : r)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const resumeSkillRun = useCallback(async (runId: string) => {
    try {
      const data = await codeApi.resumeSkillRun(runId);
      setSkillRuns((prev) => prev.map((r) => (r.id === runId ? data.run : r)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  return (
    <SkillContext.Provider
      value={{
        skills,
        skillRuns,
        loading,
        error,
        fetchSkills,
        fetchSkillRuns,
        runSkill,
        cancelSkillRun,
        resumeSkillRun,
      }}
    >
      {children}
    </SkillContext.Provider>
  );
}

export function useSkillContext() {
  const ctx = useContext(SkillContext);
  if (!ctx) throw new Error("useSkillContext must be used within SkillProvider");
  return ctx;
}

interface ApprovalContextValue {
  approvals: Approval[];
  loading: boolean;
  error: string | null;
  fetchApprovals: (status?: string) => Promise<void>;
  approve: (id: string) => Promise<void>;
  reject: (id: string, reason?: string) => Promise<void>;
}

const ApprovalContext = createContext<ApprovalContextValue | null>(null);

export function ApprovalProvider({ children }: { children: ReactNode }) {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchApprovals = useCallback(async (status?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await codeApi.getApprovals(status);
      setApprovals(data.approvals);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const approve = useCallback(async (id: string) => {
    try {
      await codeApi.approve(id);
      setApprovals((prev) => prev.map((a) => (a.id === id ? { ...a, status: "approved" as const } : a)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const reject = useCallback(async (id: string, reason?: string) => {
    try {
      await codeApi.reject(id, reason);
      setApprovals((prev) => prev.map((a) => (a.id === id ? { ...a, status: "rejected" as const } : a)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  return (
    <ApprovalContext.Provider
      value={{
        approvals,
        loading,
        error,
        fetchApprovals,
        approve,
        reject,
      }}
    >
      {children}
    </ApprovalContext.Provider>
  );
}

export function useApprovalContext() {
  const ctx = useContext(ApprovalContext);
  if (!ctx) throw new Error("useApprovalContext must be used within ApprovalProvider");
  return ctx;
}

export function CodeProvider({ children }: { children: ReactNode }) {
  return (
    <CodeWorkspaceProvider>
      <CodeSessionProvider>
        <DiagnosticsProvider>
          <AgentFlowProvider>
            <SkillProvider>
              <ApprovalProvider>
                {children}
              </ApprovalProvider>
            </SkillProvider>
          </AgentFlowProvider>
        </DiagnosticsProvider>
      </CodeSessionProvider>
    </CodeWorkspaceProvider>
  );
}
