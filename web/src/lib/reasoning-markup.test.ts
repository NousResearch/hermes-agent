import { describe, expect, it } from "vitest";
import {
  hasReasoningMarkup,
  parseReasoningMarkup,
  shouldRenderStructuredReasoning,
} from "./reasoning-markup";

describe("reasoning markup parsing", () => {
  it("unwraps thinking prose", () => {
    expect(parseReasoningMarkup("<thinking>plain prose</thinking>")).toEqual([
      { type: "prose", content: "plain prose" },
    ]);
  });

  it("separates prose from action and result segments", () => {
    expect(
      parseReasoningMarkup(
        '<thinking>prose <action>tool_call</action><result>{"ok":true}</result></thinking>',
      ),
    ).toEqual([
      { type: "prose", content: "prose " },
      { type: "action", content: "tool_call" },
      { type: "result", content: '{"ok":true}' },
    ]);
  });

  it("does not treat unrelated angle-bracket text as reasoning markup", () => {
    const content = "Compare <div> and <custom-tag> in prose.";

    expect(hasReasoningMarkup(content)).toBe(false);
    expect(parseReasoningMarkup(content)).toEqual([
      { type: "prose", content },
    ]);
  });

  it("preserves fenced XML and HTML as ordinary markdown content", () => {
    const content = [
      "```xml",
      "<thinking>keep this literal</thinking>",
      "<div>example</div>",
      "```",
    ].join("\n");

    expect(hasReasoningMarkup(content)).toBe(false);
    expect(parseReasoningMarkup(content)).toEqual([
      { type: "prose", content },
    ]);
  });

  it("leaves non-matching known tags as plain text", () => {
    const content = "before <thinking>unfinished";

    expect(hasReasoningMarkup(content)).toBe(false);
    expect(parseReasoningMarkup(content)).toEqual([
      { type: "prose", content },
    ]);
  });

  it("keeps literal known tags on the Markdown path for user and tool roles", () => {
    const content =
      'Please echo <action>tool_call</action> and <result>{"ok":true}</result>';

    expect(hasReasoningMarkup(content)).toBe(true);
    expect(shouldRenderStructuredReasoning("assistant", content)).toBe(true);
    expect(shouldRenderStructuredReasoning("user", content)).toBe(false);
    expect(shouldRenderStructuredReasoning("tool", content)).toBe(false);
    expect(shouldRenderStructuredReasoning("system", content)).toBe(false);
  });
});
