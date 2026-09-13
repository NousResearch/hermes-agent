// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Markdown } from "@/components/Markdown";

// React only routes updates through act() when this flag is set (same
// convention as ChatPage.test.tsx).
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLElement;
let root: Root;

async function render(ui: React.ReactElement) {
  await act(async () => {
    root.render(ui);
  });
}

function qs(selector: string): Element | null {
  return container.querySelector(selector);
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  container.remove();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("Markdown code blocks", () => {
  it("renders a bare <pre> with no copy button by default (desktop parity)", async () => {
    await render(
      <Markdown content={'```js\nconsole.log("hi");\n```'} />,
    );

    const pre = qs("pre");
    expect(pre).not.toBeNull();
    expect(pre!.className).toContain("bg-secondary/60");
    expect(qs("button")).toBeNull();
    expect(pre!.textContent).toContain('console.log("hi");');
  });

  it("codeCopy renders a header row with language label and copy button", async () => {
    await render(<Markdown codeCopy content={"```python\nprint('hi')\n```"} />);

    const button = qs("button");
    expect(button).not.toBeNull();
    expect(button!.getAttribute("aria-label")).toBe("Copy code");
    // Language label sits in the header row above the scrollable <pre>
    // (className carries the uppercase styling).
    expect(container.textContent).toContain("python");
    const pre = qs("pre");
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toContain("print('hi')");
  });

  it("copy button copies the code text and shows a brief copied state", async () => {
    vi.useFakeTimers();

    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    Object.defineProperty(window, "isSecureContext", {
      value: true,
      configurable: true,
    });

    await render(
      <Markdown codeCopy content={"```sh\nhermes status\n```"} />,
    );

    const button = container.querySelector("button")!;
    expect(button.textContent).toContain("Copy");

    await act(async () => {
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(writeText).toHaveBeenCalledWith("hermes status");
    expect(button.getAttribute("aria-label")).toBe("Copied");
    expect(button.textContent).toContain("Copied");

    // Brief check state: reverts after ~1.5s.
    await act(async () => {
      vi.advanceTimersByTime(1600);
    });
    expect(button.getAttribute("aria-label")).toBe("Copy code");
  });

  it("copy failure does not show the copied state", async () => {
    const writeText = vi.fn(async () => {
      throw new Error("denied");
    });
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    Object.defineProperty(window, "isSecureContext", {
      value: true,
      configurable: true,
    });

    await render(
      <Markdown codeCopy content={"```sh\nhermes status\n```"} />,
    );

    const button = container.querySelector("button")!;
    await act(async () => {
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(button.textContent).not.toContain("Copied");
  });
});
