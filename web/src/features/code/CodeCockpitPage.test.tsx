import { render, screen } from "@testing-library/react";
import CodeCockpitPage from "./CodeCockpitPage";
import { describe, it, expect, vi } from "vitest";

// Mock Zustand stores - note: vi.mock is hoisted, imports below are not directly used
vi.mock("@/stores/codeStore", () => ({
  useCodeWorkspaceStore: () => ({
    workspaces: [],
    selectedWorkspaceId: null,
    gitDiff: {},
    fetchWorkspaces: vi.fn(),
    fetchGitStatus: vi.fn(),
    fetchGitDiff: vi.fn(),
  }),
  useCodeSessionStore: () => ({
    sessions: [],
    selectedSessionId: null,
    currentEvents: [],
    commands: {},
    artifacts: {},
    fetchSessions: vi.fn(),
    selectSession: vi.fn(),
    fetchSession: vi.fn(),
    fetchEvents: vi.fn(),
    fetchCommands: vi.fn(),
    fetchArtifacts: vi.fn(),
    cancelSession: vi.fn(),
  }),
  useDiagnosticsStore: () => ({
    diagnostics: {},
    fetchDiagnostics: vi.fn(),
    runDiagnostics: vi.fn(),
  }),
  useAgentFlowStore: () => ({
    flows: [],
    fetchFlows: vi.fn(),
  }),
  useSkillStore: () => ({
    skillRuns: [],
    fetchSkillRuns: vi.fn(),
  }),
  useApprovalStore: () => ({
    approvals: [],
    fetchApprovals: vi.fn(),
  }),
}));

describe("CodeCockpitPage", () => {
  it("renders correctly with empty states", () => {
    render(<CodeCockpitPage />);
    expect(screen.getByText(/Code Cockpit/i)).toBeInTheDocument();
    expect(screen.getByText(/No sessions yet/i)).toBeInTheDocument();
    expect(screen.getByText(/Open a workspace to see Git status/i)).toBeInTheDocument();
  });
});
