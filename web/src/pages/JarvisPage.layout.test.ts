import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("./JarvisPage.tsx", import.meta.url), "utf8");
const css = readFileSync(new URL("./jarvis-dashboard.css", import.meta.url), "utf8");

describe("JarvisPage prototype dashboard integration", () => {
  it("uses the approved repository dashboard as the main Jarvis route", () => {
    expect(source).toContain("Neural command field");
    expect(source).toContain("Hermes OS");
    expect(source).toContain("Jarvis");
    expect(source).toContain("VoiceCoreField");
    expect(source).toContain("dashboard-shell");
    expect(css).toContain(".dashboard-shell");
    expect(css).toContain(".command-center");
  });

  it("keeps Memory and Agent Operations as distinct primary screens under /jarvis", () => {
    expect(source).toContain('["Dashboard", LayoutDashboard, "/jarvis"]');
    expect(source).toContain('["Memory", Network, "/jarvis/memory"]');
    expect(source).toContain('["Agents", UsersRound, "/jarvis/agents"]');
    expect(source).toContain('["Vault", Box, "/jarvis/memory"]');
    expect(source).toContain('to="/jarvis/agents"');
    expect(source).toContain('to="/jarvis/memory"');
    expect(source).not.toContain('href="/agents"');
    expect(source).not.toContain('href="/memory"');
    expect(source).not.toContain('href="/jarvis');
  });

  it("preserves the approved Jarvis visual layout concepts", () => {
    expect(source).toContain("Agent Ops");
    expect(source).toContain("Data Vault");
    expect(source).toContain("System Events");
    expect(source).toContain("Production deployment requires operator review");
    expect(source).toContain("jarvis:voice-frame");
    expect(source).toContain("prefers-reduced-motion: reduce");
    expect(css).toContain("grid-template-columns: 250px minmax(650px, 1fr) 245px");
  });

  it("wires live Jarvis overview data into the visible dashboard panels", () => {
    expect(source).toContain("api.getJarvisOverview()");
    expect(source).toContain("memoryVault?.note_count");
    expect(source).toContain("memoryVault?.decision_count");
    expect(source).toContain("memoryVault?.product_note_count");
    expect(source).toContain("serviceHealth?.system?.cpu_percent");
    expect(source).toContain("agentStatus?.active_sessions");
    expect(source).toContain("liveAgentOps(overview)");
    expect(source).toContain("overview?.agent_status?.profiles");
    expect(source).not.toContain("<strong>1.8 TB</strong>");
    expect(source).not.toContain("<strong>12,842</strong>");
    expect(source).not.toContain("<strong>184</strong>");
  });
});
