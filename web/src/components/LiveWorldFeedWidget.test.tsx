// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { LiveWorldFeedWidget } from "./LiveWorldFeedWidget";

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("LiveWorldFeedWidget", () => {
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

  it("renders live world feed headers and cards", async () => {
    await act(async () => {
      root.render(<LiveWorldFeedWidget />);
    });

    expect(container.textContent).toContain("LIVE WORLD FEED");
    expect(container.textContent).toContain("REAL-TIME TELEMETRY");
    expect(container.textContent).toContain("CRYPTO TICKER");
    expect(container.textContent).toContain("CURRENCY & GOLD");
    expect(container.textContent).toContain("ISS SATELLITE");
    expect(container.textContent).toContain("GLOBAL NEWS WIRE");
    expect(container.textContent).toContain("GITHUB REPOSITORIES");
  });

  it("switches news tabs", async () => {
    await act(async () => {
      root.render(<LiveWorldFeedWidget />);
    });

    const mostReadTab = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("mostread")
    );
    expect(mostReadTab).toBeDefined();

    await act(async () => {
      mostReadTab?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(mostReadTab?.className).toContain("text-[#00f0ff]");
  });
});
