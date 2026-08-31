// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ToolActivity, type ToolActivityItem } from "./ToolActivity";

describe("ToolActivity", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(() => {
    root.unmount();
    host.remove();
  });

  const render = async (item: ToolActivityItem) => act(async () => { root.render(createElement(ToolActivity, { item })); });

  it("renders running and complete states accessibly", async () => {
    await render({ id: "t1", name: "terminal", state: "running" });
    expect(host.querySelector("[role='status']")?.getAttribute("aria-label")).toContain("running");
    expect(host.textContent).toContain("Running");

    await act(async () => { root.render(createElement(ToolActivity, { item: { id: "t1", name: "terminal", state: "complete" } })); });
    expect(host.querySelector("[role='status']")?.getAttribute("aria-label")).toContain("complete");
    expect(host.textContent).toContain("Complete");
  });

  it("shows progress, context, summary, elapsed time, and expandable details", async () => {
    await render({ id: "t2", name: "search", state: "complete", context: "Finding docs", progress: "3/5", summary: "Found 5 results", elapsedMs: 1250, args: { query: "docs" }, result: { count: 5 } });
    expect(host.textContent).toContain("Finding docs");
    expect(host.textContent).toContain("3/5");
    expect(host.textContent).toContain("Found 5 results");
    expect(host.textContent).toContain("1.3s");
    expect(host.querySelector("details")).toBeTruthy();
    expect(host.textContent).toContain('"query": "docs"');
    expect(host.textContent).toContain('"count": 5');
  });

  it("handles unknown details safely and redacts sensitive argument keys", async () => {
    await render({ id: "t3", name: "terminal", state: "complete", args: { token: "secret", nested: { password: "hidden" } }, result: new Error("command failed") });
    expect(host.textContent).not.toContain("[object Object]");
    expect(host.textContent).not.toContain("secret");
    expect(host.textContent).not.toContain("hidden");
    expect(host.textContent).toContain("command failed");
  });
});
