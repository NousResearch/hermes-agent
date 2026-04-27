import { create } from "zustand";
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

interface CodeWorkspaceState {
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

export const useCodeWorkspaceStore = create<CodeWorkspaceState>((set) => ({
  workspaces: [],
  selectedWorkspaceId: null,
  gitStatus: {},
  gitDiff: {},
  loading: false,
  error: null,

  fetchWorkspaces: async () => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getWorkspaces();
      set({ workspaces: data.workspaces, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  selectWorkspace: (id) => {
    set({ selectedWorkspaceId: id });
  },

  refreshWorkspace: async (id) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.refreshWorkspace(id);
      set((state) => ({
        workspaces: state.workspaces.map((w) =>
          w.id === id ? data.workspace : w
        ),
        loading: false,
      }));
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  openWorkspace: async (path) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.openWorkspace(path);
      set((state) => ({
        workspaces: [...state.workspaces, data.workspace],
        selectedWorkspaceId: data.workspace.id,
        loading: false,
      }));
      return data.workspace;
    } catch (err) {
      set({ error: String(err), loading: false });
      throw err;
    }
  },

  fetchGitStatus: async (workspaceId, codeSessionId) => {
    try {
      const data = await codeApi.getGitStatus(workspaceId, codeSessionId);
      set((state) => ({
        gitStatus: { ...state.gitStatus, [workspaceId]: data.status },
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  fetchGitDiff: async (workspaceId, path) => {
    try {
      const data = await codeApi.getGitDiff(workspaceId, path);
      set((state) => ({
        gitDiff: { ...state.gitDiff, [workspaceId]: data.diff },
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));

interface CodeSessionState {
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

export const useCodeSessionStore = create<CodeSessionState>((set) => ({
  sessions: [],
  selectedSessionId: null,
  currentEvents: [],
  commands: {},
  artifacts: {},
  loading: false,
  error: null,

  fetchSessions: async (params) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getCodeSessions(params);
      set({ sessions: data.sessions, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  selectSession: (id) => {
    set({ selectedSessionId: id });
  },

  fetchSession: async (id) => {
    try {
      const data = await codeApi.getCodeSession(id);
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? data.code_session : s
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  createSession: async (payload) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.createCodeSession(payload);
      set((state) => ({
        sessions: [...state.sessions, data.code_session],
        selectedSessionId: data.code_session.id,
        loading: false,
      }));
      return data.code_session;
    } catch (err) {
      set({ error: String(err), loading: false });
      throw err;
    }
  },

  cancelSession: async (id, reason) => {
    try {
      await codeApi.cancelCodeSession(id, reason);
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? { ...s, status: "cancelled" as const } : s
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  resumeSession: async (id) => {
    try {
      const data = await codeApi.updateCodeSession(id, { status: "running" });
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? data.code_session : s
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  fetchEvents: async (sessionId) => {
    try {
      const data = await codeApi.getCodeSessionEvents(sessionId);
      set({ currentEvents: data.events });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  fetchCommands: async (sessionId) => {
    try {
      const data = await codeApi.getCommands(sessionId);
      set((state) => ({
        commands: { ...state.commands, [sessionId]: data.commands },
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  runCommand: async (sessionId, payload) => {
    try {
      const data = await codeApi.runCommand(sessionId, payload);
      set((state) => ({
        commands: {
          ...state.commands,
          [sessionId]: [...(state.commands[sessionId] || []), data.command],
        },
      }));
      return data.command;
    } catch (err) {
      set({ error: String(err) });
      throw err;
    }
  },

  cancelCommand: async (commandId) => {
    try {
      await codeApi.cancelCommand(commandId);
    } catch (err) {
      set({ error: String(err) });
    }
  },

  fetchArtifacts: async (sessionId) => {
    try {
      const data = await codeApi.getCodeSessionArtifacts(sessionId);
      set((state) => ({
        artifacts: { ...state.artifacts, [sessionId]: data.artifacts },
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  appendCommandOutput: (commandId, chunk) => {
    set((state) => {
      const newCommands = { ...state.commands };
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
      return { commands: newCommands };
    });
  },
}));

interface DiagnosticsState {
  diagnostics: Record<string, DiagnosticsResult>;
  loading: boolean;
  error: string | null;

  fetchDiagnostics: (workspaceId: string, codeSessionId?: string) => Promise<void>;
  runDiagnostics: (workspaceId: string, codeSessionId?: string) => Promise<void>;
  restartLanguageServices: (workspaceId: string) => Promise<void>;
}

export const useDiagnosticsStore = create<DiagnosticsState>((set) => ({
  diagnostics: {},
  loading: false,
  error: null,

  fetchDiagnostics: async (workspaceId, codeSessionId) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getDiagnostics(workspaceId, codeSessionId);
      set((state) => ({
        diagnostics: { ...state.diagnostics, [workspaceId]: data.result },
        loading: false,
      }));
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  runDiagnostics: async (workspaceId, codeSessionId) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getDiagnostics(workspaceId, codeSessionId);
      set((state) => ({
        diagnostics: { ...state.diagnostics, [workspaceId]: data.result },
        loading: false,
      }));
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  restartLanguageServices: async (workspaceId) => {
    try {
      await codeApi.restartLanguageServices(workspaceId);
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));

interface AgentFlowState {
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

export const useAgentFlowStore = create<AgentFlowState>((set) => ({
  flows: [],
  selectedFlowId: null,
  loading: false,
  error: null,

  fetchFlows: async (params) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getAgentFlows(params);
      set({ flows: data.flows, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  selectFlow: (id) => {
    set({ selectedFlowId: id });
  },

  runFlow: async (id) => {
    try {
      const data = await codeApi.runAgentFlow(id);
      set((state) => ({
        flows: state.flows.map((f) => (f.id === id ? data.flow : f)),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  cancelFlow: async (id, reason) => {
    try {
      const data = await codeApi.cancelAgentFlow(id, reason);
      set((state) => ({
        flows: state.flows.map((f) => (f.id === id ? data.flow : f)),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  resumeFlow: async (id) => {
    try {
      const data = await codeApi.resumeAgentFlow(id);
      set((state) => ({
        flows: state.flows.map((f) => (f.id === id ? data.flow : f)),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));

interface SkillState {
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

export const useSkillStore = create<SkillState>((set) => ({
  skills: [],
  skillRuns: [],
  loading: false,
  error: null,

  fetchSkills: async () => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getSkills();
      set({ skills: data.skills, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  fetchSkillRuns: async (params) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getSkillRuns(params);
      set({ skillRuns: data.runs, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  runSkill: async (skillName, payload) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.runSkillShortcut(skillName, payload);
      set((state) => ({
        skillRuns: [...state.skillRuns, data.run],
        loading: false,
      }));
      return data.run;
    } catch (err) {
      set({ error: String(err), loading: false });
      throw err;
    }
  },

  cancelSkillRun: async (runId, reason) => {
    try {
      const data = await codeApi.cancelSkillRun(runId, reason);
      set((state) => ({
        skillRuns: state.skillRuns.map((r) =>
          r.id === runId ? data.run : r
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  resumeSkillRun: async (runId) => {
    try {
      const data = await codeApi.resumeSkillRun(runId);
      set((state) => ({
        skillRuns: state.skillRuns.map((r) =>
          r.id === runId ? data.run : r
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));

interface ApprovalState {
  approvals: Approval[];
  loading: boolean;
  error: string | null;

  fetchApprovals: (status?: string) => Promise<void>;
  approve: (id: string) => Promise<void>;
  reject: (id: string, reason?: string) => Promise<void>;
}

export const useApprovalStore = create<ApprovalState>((set) => ({
  approvals: [],
  loading: false,
  error: null,

  fetchApprovals: async (status) => {
    set({ loading: true, error: null });
    try {
      const data = await codeApi.getApprovals(status);
      set({ approvals: data.approvals, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  approve: async (id) => {
    try {
      await codeApi.approve(id);
      set((state) => ({
        approvals: state.approvals.map((a) =>
          a.id === id ? { ...a, status: "approved" as const } : a
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },

  reject: async (id, reason) => {
    try {
      await codeApi.reject(id, reason);
      set((state) => ({
        approvals: state.approvals.map((a) =>
          a.id === id ? { ...a, status: "rejected" as const } : a
        ),
      }));
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));
