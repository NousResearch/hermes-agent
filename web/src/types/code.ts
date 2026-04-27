export interface CodeWorkspace {
  id: string;
  path: string;
  name: string;
  branch: string;
  stack: string[];
  package_manager: string | null;
  commands: string[];
  created_at: string;
  updated_at: string;
}

export interface GitStatus {
  workspace_id: string;
  branch: string;
  files: GitFile[];
  staged: GitFile[];
  unstaged: GitFile[];
  untracked: GitFile[];
  ahead: number;
  behind: number;
  clean: boolean;
}

export interface GitFile {
  path: string;
  status: "added" | "modified" | "deleted" | "renamed" | "copied";
  additions?: number;
  deletions?: number;
}

export interface GitDiff {
  workspace_id: string;
  diffs: FileDiff[];
  total_additions: number;
  total_deletions: number;
}

export interface FileDiff {
  path: string;
  old_path?: string;
  status: "added" | "modified" | "deleted" | "renamed";
  diff: string;
  additions: number;
  deletions: number;
}

export interface CodeSession {
  id: string;
  workspace_id: string;
  hermes_session_id: string | null;
  task_id: string | null;
  title: string;
  status: "created" | "running" | "waiting_approval" | "completed" | "failed" | "cancelled";
  provider: string | null;
  model: string | null;
  summary: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  metadata: Record<string, unknown> | null;
}

export interface CodeSessionEvent {
  id: string;
  code_session_id: string;
  type: string;
  message: string | null;
  payload: Record<string, unknown> | null;
  created_at: string;
}

export interface CodeCommand {
  id: string;
  code_session_id: string;
  command: string;
  cwd: string | null;
  classification: "safe" | "needs_approval" | "blocked";
  status: "created" | "running" | "completed" | "failed" | "timeout" | "cancelled";
  exit_code: number | null;
  stdout: string;
  stderr: string;
  duration_ms: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface CodeArtifact {
  id: string;
  code_session_id: string;
  tool_name: string;
  path: string;
  status: "added" | "modified" | "deleted";
  diff: string;
  additions: number;
  deletions: number;
  created_at: string;
}

export interface DiagnosticsSummary {
  errors: number;
  warnings: number;
  info: number;
  hints: number;
  total: number;
}

export interface Diagnostic {
  file: string;
  line: number | null;
  column: number | null;
  severity: "error" | "warning" | "info" | "hint";
  source: string;
  code: string | null;
  message: string;
}

export interface DiagnosticsResult {
  workspace_id: string;
  status: "ok" | "error" | "partial" | "unsupported";
  diagnostics: Diagnostic[];
  summary: DiagnosticsSummary;
  commands_run: string[];
  duration_ms: number;
  created_at: string;
}

export interface AgentFlow {
  id: string;
  code_session_id: string;
  workspace_id: string;
  task_id: string | null;
  title: string | null;
  description: string | null;
  status: "created" | "running" | "waiting_approval" | "completed" | "failed" | "cancelled";
  current_role: string | null;
  steps: AgentFlowStep[];
  result: Record<string, unknown> | null;
  error: string | null;
  provider: string | null;
  model: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface AgentFlowStep {
  id: string;
  flow_id: string;
  role: "orchestrator" | "coder" | "tester" | "reviewer";
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  title: string;
  content: string | null;
  result: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface SkillInfo {
  name: string;
  title: string;
  description: string;
  category: string;
  enabled: boolean;
}

export interface SkillRun {
  id: string;
  skill_name: string;
  workspace_id: string;
  code_session_id: string | null;
  task_id: string | null;
  agent_flow_id: string | null;
  status: "created" | "running" | "waiting_approval" | "completed" | "failed" | "cancelled";
  input: Record<string, unknown> | null;
  output: Record<string, unknown> | null;
  summary: string | null;
  diagnostics_before: DiagnosticsSummary | null;
  diagnostics_after: DiagnosticsSummary | null;
  commands: CodeCommand[];
  artifacts: CodeArtifact[];
  approval_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface ProviderInfo {
  id: string;
  name: string;
  status: "configured" | "missing_token" | "invalid_token" | "unknown";
  models: ModelInfo[];
  defaultModel: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  contextWindow?: number;
  supportsTools?: boolean;
  supportsVision?: boolean;
}

export interface ProviderSelection {
  provider: string;
  model: string;
}

export interface Approval {
  id: string;
  session_id: string;
  agent_id: string;
  status: "pending" | "approved" | "rejected";
  title: string;
  kind: "command" | "code_review" | "destructive_action" | "skill";
  details: string;
  command: string | null;
  created_at: string;
}

export type CodeCockpitTab = "timeline" | "commands" | "diff" | "diagnostics" | "skills" | "agents" | "providers";

export interface TimelineEvent {
  id: string;
  timestamp: string;
  type: "session" | "command" | "artifact" | "diagnostics" | "agent_flow" | "skill" | "approval";
  status: string;
  title: string;
  description: string | null;
  link?: string;
}
