// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ProfileContext } from "@/contexts/profile-context";
import { getArtifactStorage, getArtifactStorageKey, makeArtifactId, setArtifactPinned } from "@/lib/artifact-storage";
import { ArtifactCard } from "./ArtifactCard";

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const profileValue = {
  profile: "writer",
  currentProfile: "default",
  profiles: ["default", "writer"],
  setProfile: vi.fn(),
};
const code = "<html><body>demo</body></html>";
const detection = { kind: "html" as const, language: "html", title: "Demo" };
const storedArtifact = {
  id: makeArtifactId("session-1", "html", "html", "Demo", code),
  sessionId: "session-1",
  kind: "html" as const,
  language: "html",
  title: "Demo",
  code,
  createdAt: 1700000000000,
};

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  localStorage.clear();
  profileValue.setProfile.mockReset();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  localStorage.clear();
});

describe("ArtifactCard same-tab storage synchronization", () => {
  it("updates its pin state when the library changes the same profile", async () => {
    await render(
      <ProfileContext.Provider value={profileValue}>
        <ArtifactCard code={code} detection={detection} sessionId="session-1" />
      </ProfileContext.Provider>,
    );

    await vi.waitFor(() => expect(container.querySelector('[aria-label="Pin artifact"]')).not.toBeNull());
    await act(async () => {
      setArtifactPinned(getArtifactStorage(), storedArtifact, true, "writer");
      await Promise.resolve();
    });

    await vi.waitFor(() => expect(container.querySelector('[aria-label="Unpin artifact"]')).not.toBeNull());
  });

  it("updates from a native storage event emitted by another tab", async () => {
    await render(
      <ProfileContext.Provider value={profileValue}>
        <ArtifactCard code={code} detection={detection} sessionId="session-1" />
      </ProfileContext.Provider>,
    );

    await vi.waitFor(() => expect(container.querySelector('[aria-label="Pin artifact"]')).not.toBeNull());
    const storageKey = getArtifactStorageKey("writer");
    localStorage.setItem(storageKey, JSON.stringify([storedArtifact]));
    await act(async () => {
      window.dispatchEvent(new StorageEvent("storage", { key: storageKey }));
      await Promise.resolve();
    });

    await vi.waitFor(() => expect(container.querySelector('[aria-label="Unpin artifact"]')).not.toBeNull());
  });

  it("keeps generated document previews inert", async () => {
    await render(
      <ProfileContext.Provider value={profileValue}>
        <ArtifactCard code={code} detection={detection} sessionId="session-1" />
      </ProfileContext.Provider>,
    );

    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="Preview artifact"]')?.click());
    const preview = container.querySelector<HTMLIFrameElement>('iframe[sandbox=""]');
    expect(preview).not.toBeNull();
    expect(preview?.getAttribute("referrerpolicy")).toBe("no-referrer");
    expect(preview?.getAttribute("srcdoc")).toContain("Content-Security-Policy");
  });
});