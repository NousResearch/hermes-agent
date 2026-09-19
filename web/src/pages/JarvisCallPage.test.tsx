// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import JarvisCallPage from "./JarvisCallPage";

vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn(), setTitle: vi.fn() }),
}));

vi.mock("@/lib/gatewayClient", () => {
  return {
    GatewayClient: class MockGatewayClient {
      connectionState = "open";
      async connect() {
        return Promise.resolve();
      }
      close() {}
      async request(method: string) {
        if (method === "session.create") {
          return { session_id: "test-session-123" };
        }
        return { ok: true };
      }
      on() {
        return () => {};
      }
    },
  };
});

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("JarvisCallPage", () => {
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

  it("renders Jarvis Command Center tabs and Live Voice Call", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/jarvis?tab=call"]}>
          <JarvisCallPage />
        </MemoryRouter>
      );
    });

    expect(container.textContent).toContain("J.A.R.V.I.S. Executive OS & Autonomous Hub");
    expect(container.textContent).toContain("Live Call");
    expect(container.textContent).toContain("Core Assistant");
    expect(container.textContent).toContain("Music Player");
    expect(container.textContent).toContain("World Feed");
    expect(container.textContent).toContain("Start Call");
  });

  it("switches to Core Assistant tab", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/jarvis?tab=call"]}>
          <JarvisCallPage />
        </MemoryRouter>
      );
    });

    const coreButton = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Core Assistant")
    );
    expect(coreButton).toBeDefined();

    await act(async () => {
      coreButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.textContent).toContain("JARVIS CHIEF OF STAFF");
    expect(container.textContent).toContain("FOCUS GOAL");
  });

  it("switches to Music Player tab", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/jarvis?tab=call"]}>
          <JarvisCallPage />
        </MemoryRouter>
      );
    });

    const musicButton = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Music Player")
    );
    expect(musicButton).toBeDefined();

    await act(async () => {
      musicButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.textContent).toContain("JARVIS AUDIO DECK");
  });

  it("switches to World Feed tab", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/jarvis?tab=call"]}>
          <JarvisCallPage />
        </MemoryRouter>
      );
    });

    const feedButton = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("World Feed")
    );
    expect(feedButton).toBeDefined();

    await act(async () => {
      feedButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.textContent).toContain("LIVE WORLD FEED");
  });

  it("renders Voice Cadence and Anti-Interruption Shield controls", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/jarvis?tab=call"]}>
          <JarvisCallPage />
        </MemoryRouter>
      );
    });

    expect(container.textContent).toContain("Voice Cadence & Shield:");
    expect(container.textContent).toContain("Hands-Free (Smart Pauses)");
    expect(container.textContent).toContain("Push-to-Talk (Zero Cutoffs)");
    expect(container.textContent).toContain("Relaxed (3.0s)");
    expect(container.textContent).toContain("Balanced (2.2s)");
    expect(container.textContent).toContain("Anti-Interruption Shield: Active");

    // Toggle to push-to-talk mode
    const pttButton = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Push-to-Talk (Zero Cutoffs)")
    );
    expect(pttButton).toBeDefined();

    await act(async () => {
      pttButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    // In push-to-talk mode, pause tolerance pills are hidden
    expect(container.textContent).not.toContain("Relaxed (3.0s)");
  });

  it("renders 24/7 Live Mode toggle and Tony Stark Knock / Wake Up button", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/jarvis?tab=call"]}>
          <JarvisCallPage />
        </MemoryRouter>
      );
    });

    expect(container.textContent).toContain("24/7 Live Mode");
    expect(container.textContent).toContain("Knock / Wake Up");
    expect(container.textContent).toContain("24/7 IRON MAN LIVE MODE READY");

    const liveModeBtn = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("24/7 Live Mode")
    );
    expect(liveModeBtn).toBeDefined();

    await act(async () => {
      liveModeBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
  });
});

