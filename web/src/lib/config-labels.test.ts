import { describe, expect, it } from "vitest";

import { ar } from "@/i18n/ar";
import { en } from "@/i18n/en";

import { configDescription, configFieldLabel } from "./config-labels";

const ARABIC = /\p{Script=Arabic}/u;

// The dashboard's config editor is generated from the backend schema, so most
// field labels have no catalog entry and fall through to a shape-based
// fallback. That fallback must stay language-aware: English has to keep
// rendering exactly what upstream rendered — `key.replace(/_/g, " ")`
// title-cased — and must never pick up another locale's wording.
describe("config-labels", () => {
  it("title-cases unlabelled schema keys for English, exactly as upstream did", () => {
    const cases: ReadonlyArray<readonly [string, string]> = [
      ["agent.max_turns", "Max Turns"],
      ["agent.min_interval_hours", "Min Interval Hours"],
      ["agent.gateway_timeout", "Gateway Timeout"],
      ["auxiliary.vision.base_url", "Base Url"],
      ["memory.memory_enabled", "Memory Enabled"],
      ["security.tirith_path", "Tirith Path"],
      ["dashboard.approval_mode", "Approval Mode"],
      ["gateway.retry_count", "Retry Count"],
      ["agent.token_limit", "Token Limit"],
    ];

    for (const [key, expected] of cases) {
      expect(configFieldLabel(key, en.config)).toBe(expected);
    }
  });

  it("never renders non-Latin script in the English config editor", () => {
    const shapes = [
      "max_x",
      "min_x",
      "x_enabled",
      "x_disabled",
      "x_timeout",
      "x_count",
      "x_mode",
      "x_path",
      "x_url",
      "x_interval",
      "x_limit",
    ];

    for (const key of shapes) {
      expect(configFieldLabel(`section.${key}`, en.config)).not.toMatch(ARABIC);
    }
  });

  it("passes the backend's English description through untouched", () => {
    // `hermes_cli/web_server.py` generates these as "Section → Field"; the
    // arrow direction and the sentence casing are the server's, not ours.
    expect(
      configDescription("agent.max_turns", "Agent → Max turns", en.config),
    ).toBe("Agent → Max turns");
    expect(
      configDescription(
        "agent.max_turns",
        "Maximum number of concurrent sessions",
        en.config,
      ),
    ).toBe("Maximum number of concurrent sessions");
  });

  it("still localizes the same keys for Arabic", () => {
    for (const key of ["section.max_x", "section.x_timeout", "section.x_url"]) {
      expect(configFieldLabel(key, ar.config)).toMatch(ARABIC);
    }
  });
});
