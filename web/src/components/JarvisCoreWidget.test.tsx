// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { JarvisCoreWidget } from "./JarvisCoreWidget";

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("JarvisCoreWidget", () => {
  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("renders Jarvis Chief of Staff and Pomodoro focus timer", async () => {
    await act(async () => {
      root.render(<JarvisCoreWidget />);
    });

    expect(container.textContent).toContain("JARVIS CHIEF OF STAFF");
    expect(container.textContent).toContain("MARK 85");
    expect(container.textContent).toContain("FOCUS GOAL");
    expect(container.textContent).toContain("DEEP WORK");
    expect(container.textContent).toContain("COMMANDS:");
    expect(container.textContent).toContain("Eng. Ibrahim Abdelsattar");
  });

  it("switches to topology view", async () => {
    await act(async () => {
      root.render(<JarvisCoreWidget />);
    });

    const topologyBtn = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Topology")
    );
    expect(topologyBtn).toBeDefined();

    await act(async () => {
      topologyBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.textContent).toContain("HOLOGRAPHIC NEURAL TOPOLOGY");
    expect(container.textContent).toContain("7 NODES ACTIVE");
  });

  it("switches to projects and code inspector view", async () => {
    await act(async () => {
      root.render(<JarvisCoreWidget />);
    });

    const projectsBtn = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Projects & Code")
    );
    expect(projectsBtn).toBeDefined();

    await act(async () => {
      projectsBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.textContent).toContain("GITHUB PROJECTS (41)");
    expect(container.textContent).toContain("Source Code Inspector");
    expect(container.textContent).toContain("hermes-agent");
  });

  it("switches to cloud and tasks view", async () => {
    await act(async () => {
      root.render(<JarvisCoreWidget />);
    });

    const cloudBtn = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Cloud & Tasks")
    );
    expect(cloudBtn).toBeDefined();

    await act(async () => {
      cloudBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.textContent).toContain("HERMES AGENT ORCHESTRATION & TASKS");
    expect(container.textContent).toContain("GOOGLE CLOUD PLATFORM (GCP)");
    expect(container.textContent).toContain("Google BigQuery");
  });

  it("triggers deep work mode toggle", async () => {
    await act(async () => {
      root.render(<JarvisCoreWidget />);
    });

    const deepWorkBtn = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("DEEP WORK")
    );
    expect(deepWorkBtn).toBeDefined();

    await act(async () => {
      deepWorkBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(deepWorkBtn?.className).toContain("bg-amber-400");
  });
});
