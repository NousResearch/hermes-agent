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
  ProviderInfo,
  ProviderSelection,
  Approval,
} from "@/types/code";

const BASE = "";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = (window as Window & { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__;
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const res = await fetch(`${BASE}${url}`, { ...init, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export const codeApi = {
  // Workspaces
  getWorkspaces: () =>
    fetchJSON<{ workspaces: CodeWorkspace[]; total: number }>("/api/code/workspaces"),

  openWorkspace: (path: string) =>
    fetchJSON<{ workspace: CodeWorkspace }>("/api/code/workspaces/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),

  getWorkspace: (workspaceId: string) =>
    fetchJSON<{ workspace: CodeWorkspace }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}`
    ),

  refreshWorkspace: (workspaceId: string) =>
    fetchJSON<{ workspace: CodeWorkspace }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/refresh`,
      { method: "POST" }
    ),

  // Git
  getGitStatus: (workspaceId: string, codeSessionId?: string) => {
    const qs = codeSessionId ? `?code_session_id=${encodeURIComponent(codeSessionId)}` : "";
    return fetchJSON<{ status: GitStatus }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/status${qs}`
    );
  },

  getGitDiff: (workspaceId: string, path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : "";
    return fetchJSON<{ diff: GitDiff }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/diff${qs}`
    );
  },

  getGitBranch: (workspaceId: string) =>
    fetchJSON<{ branch: string; remote: string | null }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/branch`
    ),

  getGitRemote: (workspaceId: string) =>
    fetchJSON<{ remote: string; url: string | null }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/remote`
    ),

  prepareBranch: (workspaceId: string) =>
    fetchJSON<{ branch: string; has_changes: boolean }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/branch/prepare`,
      { method: "POST" }
    ),

  prepareCommit: (workspaceId: string) =>
    fetchJSON<{ message: string | null; has_changes: boolean; staged: string[]; unstaged: string[] }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/commit/prepare`,
      { method: "POST" }
    ),

  snapshot: (workspaceId: string) =>
    fetchJSON<{ snapshot_id: string; created_at: string }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/git/snapshot`,
      { method: "POST" }
    ),

  // Code Sessions
  getCodeSessions: (params?: { workspace_id?: string; status?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    if (params?.workspace_id) qs.set("workspace_id", params.workspace_id);
    if (params?.status) qs.set("status", params.status);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    const query = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJSON<{ sessions: CodeSession[]; total: number }>(`/api/code/sessions${query}`);
  },

  createCodeSession: (payload: {
    workspace_id: string;
    hermes_session_id?: string;
    task_id?: string;
    title?: string;
    provider?: string;
    model?: string;
  }) =>
    fetchJSON<{ code_session: CodeSession }>("/api/code/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getCodeSession: (codeSessionId: string) =>
    fetchJSON<{ code_session: CodeSession }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}`
    ),

  updateCodeSession: (
    codeSessionId: string,
    updates: Partial<{
      title: string;
      status: string;
      summary: string;
      provider: string;
      model: string;
    }>
  ) =>
    fetchJSON<{ code_session: CodeSession }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      }
    ),

  cancelCodeSession: (codeSessionId: string, reason?: string) =>
    fetchJSON<{ code_session: CodeSession }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      }
    ),

  completeCodeSession: (codeSessionId: string, summary?: string) =>
    fetchJSON<{ code_session: CodeSession }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/complete`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ summary }),
      }
    ),

  getCodeSessionEvents: (codeSessionId: string) =>
    fetchJSON<{ events: CodeSessionEvent[]; total: number }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/events`
    ),

  sendEvent: (codeSessionId: string, payload: { type: string; message?: string }) =>
    fetchJSON<{ event: CodeSessionEvent }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/events`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),

  // Commands
  getCommands: (codeSessionId: string) =>
    fetchJSON<{ commands: CodeCommand[]; total: number }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/commands`
    ),

  runCommand: (
    codeSessionId: string,
    payload: { command: string; cwd?: string; timeout_seconds?: number }
  ) =>
    fetchJSON<{ command: CodeCommand }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/commands/run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),

  cancelCommand: (commandId: string) =>
    fetchJSON<{ command: CodeCommand }>(`/api/code/commands/${encodeURIComponent(commandId)}/cancel`, {
      method: "POST",
    }),

  // Artifacts
  getCodeSessionArtifacts: (codeSessionId: string) =>
    fetchJSON<{ code_session_id: string; artifacts: CodeArtifact[]; total: number }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/artifacts`
    ),

  getSessionArtifacts: (sessionId: string) =>
    fetchJSON<{ session_id: string; artifacts: CodeArtifact[]; total: number }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/artifacts`
    ),

  // Diagnostics
  getDiagnostics: (workspaceId: string, codeSessionId?: string) => {
    const qs = codeSessionId ? `?code_session_id=${encodeURIComponent(codeSessionId)}` : "";
    return fetchJSON<{ result: DiagnosticsResult }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/diagnostics${qs}`
    );
  },

  getFileDiagnostics: (workspaceId: string, filePath: string, codeSessionId?: string) => {
    const params = new URLSearchParams({ path: filePath });
    if (codeSessionId) params.set("code_session_id", codeSessionId);
    return fetchJSON<{ result: DiagnosticsResult }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/diagnostics/file?${params.toString()}`
    );
  },

  getSupportedLanguages: (workspaceId: string) =>
    fetchJSON<{ languages: string[] }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/languages`
    ),

  restartLanguageServices: (workspaceId: string) =>
    fetchJSON<{ workspace_id: string; action: string; status: string; message: string }>(
      `/api/code/workspaces/${encodeURIComponent(workspaceId)}/lsp/restart`,
      { method: "POST" }
    ),

  // Agent Flows
  getAgentFlows: (params?: {
    code_session_id?: string;
    workspace_id?: string;
    status?: string;
    limit?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.code_session_id) qs.set("code_session_id", params.code_session_id);
    if (params?.workspace_id) qs.set("workspace_id", params.workspace_id);
    if (params?.status) qs.set("status", params.status);
    if (params?.limit) qs.set("limit", String(params.limit));
    const query = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJSON<{ flows: AgentFlow[] }>(`/api/code/agent-flows${query}`);
  },

  getAgentFlow: (flowId: string) =>
    fetchJSON<{ flow: AgentFlow }>(`/api/code/agent-flows/${encodeURIComponent(flowId)}`),

  runAgentFlow: (flowId: string) =>
    fetchJSON<{ flow: AgentFlow }>(`/api/code/agent-flows/${encodeURIComponent(flowId)}/run`, {
      method: "POST",
    }),

  cancelAgentFlow: (flowId: string, reason?: string) =>
    fetchJSON<{ flow: AgentFlow }>(
      `/api/code/agent-flows/${encodeURIComponent(flowId)}/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      }
    ),

  resumeAgentFlow: (flowId: string) =>
    fetchJSON<{ flow: AgentFlow }>(
      `/api/code/agent-flows/${encodeURIComponent(flowId)}/resume`,
      { method: "POST" }
    ),

  getSessionAgentFlows: (codeSessionId: string, limit = 50) =>
    fetchJSON<{ flows: AgentFlow[] }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/agent-flows?limit=${limit}`
    ),

  // Skills
  getSkills: () => fetchJSON<{ skills: SkillInfo[] }>("/api/code/skills"),

  getSkillRuns: (params?: {
    workspace_id?: string;
    code_session_id?: string;
    skill_name?: string;
    status?: string;
    limit?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.workspace_id) qs.set("workspace_id", params.workspace_id);
    if (params?.code_session_id) qs.set("code_session_id", params.code_session_id);
    if (params?.skill_name) qs.set("skill_name", params.skill_name);
    if (params?.status) qs.set("status", params.status);
    if (params?.limit) qs.set("limit", String(params.limit));
    const query = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJSON<{ runs: SkillRun[] }>(`/api/code/skill-runs${query}`);
  },

  createSkillRun: (payload: {
    skill_name: string;
    workspace_id: string;
    code_session_id?: string;
    task_id?: string;
    input?: Record<string, unknown>;
  }) =>
    fetchJSON<{ run: SkillRun }>("/api/code/skill-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getSkillRun: (runId: string) =>
    fetchJSON<{ run: SkillRun }>(`/api/code/skill-runs/${encodeURIComponent(runId)}`),

  runSkill: (runId: string) =>
    fetchJSON<{ run: SkillRun }>(`/api/code/skill-runs/${encodeURIComponent(runId)}/run`, {
      method: "POST",
    }),

  cancelSkillRun: (runId: string, reason?: string) =>
    fetchJSON<{ run: SkillRun }>(
      `/api/code/skill-runs/${encodeURIComponent(runId)}/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      }
    ),

  resumeSkillRun: (runId: string) =>
    fetchJSON<{ run: SkillRun }>(
      `/api/code/skill-runs/${encodeURIComponent(runId)}/resume`,
      { method: "POST" }
    ),

  getSessionSkillRuns: (codeSessionId: string, limit = 50) =>
    fetchJSON<{ runs: SkillRun[] }>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/skill-runs?limit=${limit}`
    ),

  runSkillShortcut: (
    skillName: string,
    payload: {
      workspace_id: string;
      code_session_id?: string;
      task_id?: string;
      input?: Record<string, unknown>;
    }
  ) =>
    fetchJSON<{ run: SkillRun }>(`/api/code/skills/${skillName}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  // Provider
  getProviders: () =>
    fetchJSON<{
      providers: ProviderInfo[];
      current: ProviderSelection;
    }>("/api/providers"),

  selectProvider: (providerId: string, modelId?: string) =>
    fetchJSON<{ ok: boolean; provider: string; model: string }>("/api/providers/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider_id: providerId, model_id: modelId }),
    }),

  getSessionModel: (codeSessionId: string) =>
    fetchJSON<Record<string, unknown>>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/model`
    ),

  updateSessionModel: (codeSessionId: string, provider: string, model: string) =>
    fetchJSON<Record<string, unknown>>(
      `/api/code/sessions/${encodeURIComponent(codeSessionId)}/model`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model }),
      }
    ),

  // Approvals
  getApprovals: (status?: string, limit = 50) => {
    const qs = status ? `?status=${status}&limit=${limit}` : `?limit=${limit}`;
    return fetchJSON<{ approvals: Approval[]; total: number }>(`/api/approvals${qs}`);
  },

  approve: (approvalId: string) =>
    fetchJSON<{ ok: boolean; approval_id: string; status: string }>(
      `/api/approvals/${encodeURIComponent(approvalId)}/approve`,
      { method: "POST" }
    ),

  reject: (approvalId: string, reason?: string) =>
    fetchJSON<{ ok: boolean; approval_id: string; status: string }>(
      `/api/approvals/${encodeURIComponent(approvalId)}/reject`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      }
    ),
};
