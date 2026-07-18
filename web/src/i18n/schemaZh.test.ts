import { describe, expect, it } from "vitest";

import {
  resolveSchemaDescription,
  resolveSchemaLabel,
  resolveSchemaLeafLabel,
} from "./schema";
import { resolveTranslations } from "./runtime";

const zhSchema = resolveTranslations("zh").schema;

describe("schema localization", () => {
  it("keeps curated wording for known configuration fields", () => {
    expect(resolveSchemaLabel(zhSchema, "display.language", "Language")).toBe(
      "显示 → 语言",
    );
  });

  it("builds a readable label for fields added after the curated catalog", () => {
    expect(
      resolveSchemaLabel(
        zhSchema,
        "gateway.restart_loop_guard.max_restarts",
        "Max Restarts",
      ),
    ).toBe("网关 → 重启循环保护 → 最大重启次数");
  });

  it("preserves unknown technical terms instead of dropping path context", () => {
    expect(resolveSchemaLabel(zhSchema, "vertex.project_id", "Project ID")).toBe(
      "Vertex → 项目 ID",
    );
  });

  it("keeps technical identifiers separated from surrounding Chinese words", () => {
    expect(
      resolveSchemaLabel(
        zhSchema,
        "gateway.api_server.max_concurrent_runs",
        "Max Concurrent Runs",
      ),
    ).toBe("网关 → API 服务器 → 最大并发运行次数");
  });

  it("provides contextual descriptions only for informative schema overrides", () => {
    expect(
      resolveSchemaDescription(
        zhSchema,
        "display.busy_input_mode",
        "English fallback",
      ),
    ).toBe(
      "Agent 运行时收到新输入后的处理方式。",
    );
    expect(
      resolveSchemaDescription(
        zhSchema,
        "agent.max_turns",
        "Maximum turns",
      ),
    ).toBe("Maximum turns");
  });

  it("localizes the Codex commentary setting added by the runtime", () => {
    expect(
      resolveSchemaLabel(
        zhSchema,
        "display.show_commentary",
        "Show Commentary",
      ),
    ).toBe("显示 → 显示进度解说");
    expect(
      resolveSchemaDescription(
        zhSchema,
        "display.show_commentary",
        "Show commentary emitted by Codex models.",
      ),
    ).toContain("Codex 模型");
  });

  it("provides concise labels for nested editors", () => {
    expect(
      resolveSchemaLeafLabel(
        zhSchema,
        "gateway.api_server.max_concurrent_runs",
        "max_concurrent_runs",
      ),
    ).toBe("最大并发运行次数");
  });

  it("keeps English as the fallback for locale packs without schema wording", () => {
    const schema = resolveTranslations("zh-hant").schema;

    expect(resolveSchemaLabel(schema, "display.language", "Language")).toBe(
      "Language",
    );
    expect(
      resolveSchemaDescription(schema, "display.language", "Display language"),
    ).toBe("Display language");
  });
});
