import { describe, expect, it } from "vitest";

import {
  schemaZhDescription,
  schemaZhLabel,
  schemaZhLeafLabel,
} from "./schemaZh";

describe("schemaZhLabel", () => {
  it("keeps curated wording for known configuration fields", () => {
    expect(schemaZhLabel("display.language")).toBe("显示 → 语言");
  });

  it("builds a readable label for fields added after the curated catalog", () => {
    expect(schemaZhLabel("gateway.restart_loop_guard.max_restarts")).toBe(
      "网关 → 重启循环保护 → 最大重启次数",
    );
  });

  it("preserves unknown technical terms instead of dropping path context", () => {
    expect(schemaZhLabel("vertex.project_id")).toBe("Vertex → 项目 ID");
  });

  it("keeps technical identifiers separated from surrounding Chinese words", () => {
    expect(schemaZhLabel("gateway.api_server.max_concurrent_runs")).toBe(
      "网关 → API 服务器 → 最大并发运行次数",
    );
  });

  it("provides contextual descriptions only for informative schema overrides", () => {
    expect(schemaZhDescription("display.busy_input_mode")).toBe(
      "Agent 运行时收到新输入后的处理方式。",
    );
    expect(schemaZhDescription("agent.max_turns")).toBe("");
  });

  it("provides concise labels for nested editors", () => {
    expect(schemaZhLeafLabel("gateway.api_server.max_concurrent_runs")).toBe(
      "最大并发运行次数",
    );
  });
});
