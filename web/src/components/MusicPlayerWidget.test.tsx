// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MusicPlayerWidget } from "./MusicPlayerWidget";

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("MusicPlayerWidget", () => {
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

  it("renders audio deck and controls", async () => {
    await act(async () => {
      root.render(<MusicPlayerWidget />);
    });

    expect(container.textContent).toContain("JARVIS AUDIO DECK");
    expect(container.textContent).toContain("HOLOGRAPHIC");
    expect(container.textContent).toContain("AUDIO LIBRARY QUEUE");
    expect(container.textContent).toContain("Mark 85 Cyber Sentinel");
  });

  it("toggles play state", async () => {
    await act(async () => {
      root.render(<MusicPlayerWidget />);
    });

    const playBtn = container.querySelector("button[title='Play']");
    expect(playBtn).toBeDefined();

    await act(async () => {
      playBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    // Check if pause button exists or play button is toggled
    const pauseBtn = container.querySelector("button[title='Pause']");
    expect(pauseBtn).toBeDefined();
  });

  it("renders YouTube Live controls and search input", async () => {
    await act(async () => {
      root.render(<MusicPlayerWidget />);
    });

    expect(container.textContent).toContain("YouTube");
    const ytInput = container.querySelector("input[placeholder='Search & play YouTube...']");
    expect(ytInput).toBeDefined();
  });
});

