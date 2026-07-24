import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import {
  createInstallPromptStore,
  type BeforeInstallPromptEvent,
  type InstallPromptHost,
} from "./install-prompt";

type HostListener = (event: Event) => void;
type DisplayModeListener = (event: { matches: boolean }) => void;

class FakeDisplayMode {
  matches: boolean;
  private listeners = new Set<DisplayModeListener>();

  constructor(matches: boolean) {
    this.matches = matches;
  }

  addEventListener(_type: "change", listener: DisplayModeListener) {
    this.listeners.add(listener);
  }

  removeEventListener(_type: "change", listener: DisplayModeListener) {
    this.listeners.delete(listener);
  }

  setMatches(matches: boolean) {
    this.matches = matches;
    this.listeners.forEach((listener) => listener({ matches }));
  }
}

class FakeHost implements InstallPromptHost {
  readonly displayMode: FakeDisplayMode;
  private listeners = new Map<string, Set<HostListener>>();

  constructor(standalone = false) {
    this.displayMode = new FakeDisplayMode(standalone);
  }

  addEventListener(type: string, listener: HostListener) {
    const listeners = this.listeners.get(type) ?? new Set<HostListener>();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: HostListener) {
    this.listeners.get(type)?.delete(listener);
  }

  matchMedia() {
    return this.displayMode;
  }

  dispatch(type: string, event: Event) {
    this.listeners.get(type)?.forEach((listener) => listener(event));
  }
}

function installEvent(outcome: "accepted" | "dismissed" = "accepted") {
  const event = {
    preventDefault: vi.fn(),
    prompt: vi.fn(async () => undefined),
    userChoice: Promise.resolve({ outcome, platform: "web" }),
  } as unknown as BeforeInstallPromptEvent;
  return event;
}

describe("install prompt store", () => {
  it("does not advertise installation without a browser prompt event", () => {
    const store = createInstallPromptStore(new FakeHost());

    expect(store.getSnapshot()).toEqual({
      available: false,
      prompting: false,
      standalone: false,
    });
  });

  it("captures a real prompt and consumes it only after a user action", async () => {
    const host = new FakeHost();
    const store = createInstallPromptStore(host);
    const event = installEvent("dismissed");
    const listener = vi.fn();
    store.subscribe(listener);

    host.dispatch("beforeinstallprompt", event);

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(event.prompt).not.toHaveBeenCalled();
    expect(store.getSnapshot().available).toBe(true);

    await expect(store.prompt()).resolves.toBe("dismissed");
    expect(event.prompt).toHaveBeenCalledOnce();
    expect(store.getSnapshot().available).toBe(false);
    await expect(store.prompt()).resolves.toBe("unavailable");
    expect(event.prompt).toHaveBeenCalledOnce();
    expect(listener).toHaveBeenCalled();
  });

  it("stays unavailable in standalone display mode", () => {
    const host = new FakeHost(true);
    const store = createInstallPromptStore(host);
    const event = installEvent();

    host.dispatch("beforeinstallprompt", event);

    expect(store.getSnapshot()).toEqual({
      available: false,
      prompting: false,
      standalone: true,
    });
    expect(event.prompt).not.toHaveBeenCalled();
  });

  it("hides an available prompt when standalone mode activates", () => {
    const host = new FakeHost();
    const store = createInstallPromptStore(host);
    host.dispatch("beforeinstallprompt", installEvent());
    expect(store.getSnapshot().available).toBe(true);

    host.displayMode.setMatches(true);

    expect(store.getSnapshot()).toEqual({
      available: false,
      prompting: false,
      standalone: true,
    });
  });

  it("hides the prompt after the appinstalled event", () => {
    const host = new FakeHost();
    const store = createInstallPromptStore(host);
    host.dispatch("beforeinstallprompt", installEvent());
    expect(store.getSnapshot().available).toBe(true);

    host.dispatch("appinstalled", {} as Event);

    expect(store.getSnapshot().available).toBe(false);
  });

  it("contains no browser storage persistence", () => {
    const source = readFileSync(new URL("./install-prompt.ts", import.meta.url), "utf8");

    expect(source).not.toContain("localStorage");
    expect(source).not.toContain("sessionStorage");
  });
});
