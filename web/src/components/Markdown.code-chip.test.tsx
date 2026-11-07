// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";

import { Markdown } from "./Markdown";

function renderMarkdown(content: string): HTMLElement {
  const container = document.createElement("div");
  const root = createRoot(container);
  act(() => {
    root.render(<Markdown content={content} />);
  });
  return container;
}

// Chat code surfaces must read as distinct areas against the bubble card.
// The bubble is card (4% midground tint); `secondary` at 60% opacity landed
// at ~3.6%, which is indistinguishable from the bubble itself. Inline code
// and fences therefore use the dedicated code-chip token pair.
describe("Markdown code surfaces", () => {
  it("renders inline code as a code chip, not the near-invisible secondary", () => {
    const container = renderMarkdown("Run `python3 scripts/ship.sh` first");
    const code = container.querySelector("code");
    expect(code).not.toBeNull();
    expect(code.className).toContain("bg-code-chip");
    expect(code.className).toContain("rounded");
    expect(code.className).not.toContain("bg-secondary");
  });

  it("renders fenced code as a bordered code chip", () => {
    const container = renderMarkdown("```bash\nhermes skin set ui_tool #00FFFF\n```");
    const pre = container.querySelector("pre");
    expect(pre).not.toBeNull();
    expect(pre.className).toContain("bg-code-chip");
    expect(pre.className).toContain("border-code-chip-border");
  });
});
