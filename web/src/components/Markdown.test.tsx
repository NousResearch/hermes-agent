// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Markdown } from "./Markdown";

/**
 * T4.5 — transcript markdown. The hand-written renderer covered code/inline/
 * headers/lists/links but dropped GFM tables (pipe rows painted as a
 * paragraph), nested lists (child items flattened into siblings), and task
 * lists (`[ ]`/`[x]` printed literally) — and code blocks had no copy affordance.
 *
 * (Plan premise adaptation: remark/rehype + highlighter would need new
 * dependencies; `pnpm install` is forbidden in this worktree, so the existing
 * parser gained the missing constructs instead.)
 */

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

function render(content: string, streaming?: boolean) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  act(() => {
    root.render(<Markdown content={content} streaming={streaming} />);
  });
  return container;
}

beforeEach(() => {
  vi.stubGlobal("navigator", { clipboard: { writeText: vi.fn(async () => undefined) } });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("markdown tables, nested lists, task lists, code copy (T4.5)", () => {
  it("renders a GFM pipe table as a <table>", () => {
    const el = render("| a | b |\n|---|---|\n| 1 | 2 |\n");
    expect(el.innerHTML).toContain("<table");
    const cells = el.querySelectorAll("td");
    expect(cells.length).toBe(2);
    expect(el.querySelectorAll("th").length).toBe(2);
    expect(cells[0].textContent).toBe("1");
  });

  it("keeps nested list items nested", () => {
    const el = render("- x\n  - y\n");
    expect(el.innerHTML).toContain("<ul");
    const childList = el.querySelector("li ul");
    expect(childList).not.toBeNull();
    expect(childList!.querySelector("li")?.textContent).toContain("y");
  });

  it("renders task list checkboxes reflecting state", () => {
    const el = render("- [ ] todo\n- [x] done\n");
    const boxes = el.querySelectorAll('input[type="checkbox"]');
    expect(boxes.length).toBe(2);
    expect((boxes[0] as HTMLInputElement).checked).toBe(false);
    expect((boxes[1] as HTMLInputElement).checked).toBe(true);
  });

  it("offers a copy button on fenced code blocks", () => {
    const el = render("```py\nprint(1)\n```");
    const copy = el.querySelector<HTMLButtonElement>("button");
    expect(copy).not.toBeNull();
    expect(copy!.textContent).toContain("Copy");
    copy!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  it("keeps the streaming caret on the last block", () => {
    const el = render("hello", true);
    expect(el.innerHTML).toContain("animate-pulse");
  });

  it("still drops unsafe link schemes", () => {
    const el = render("[click](javascript:alert(1))");
    expect(el.innerHTML).not.toContain("javascript:");
    expect(el.innerHTML).not.toContain("<a");
  });
});
