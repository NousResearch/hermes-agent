// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getArtifactStorageKey } from "@/lib/artifact-storage";

const profileMocks = vi.hoisted(() => ({
  profile: "writer",
  currentProfile: "default",
}));
const headerMocks = vi.hoisted(() => ({
  setAfterTitle: vi.fn(),
  setEnd: vi.fn(),
}));
const clipboardMocks = vi.hoisted(() => ({
  copyTextToClipboard: vi.fn(),
}));

vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => profileMocks,
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => headerMocks,
}));
vi.mock("@/lib/clipboard", () => clipboardMocks);

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  act(() => {
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

beforeEach(() => {
  profileMocks.profile = "writer";
  profileMocks.currentProfile = "default";
  headerMocks.setAfterTitle.mockReset();
  headerMocks.setEnd.mockReset();
  clipboardMocks.copyTextToClipboard.mockReset();
  clipboardMocks.copyTextToClipboard.mockResolvedValue(true);
  localStorage.clear();
  localStorage.setItem(
    "hermes.native-chat.artifacts.v2.writer",
    JSON.stringify([
      {
        id: "writer-one",
        sessionId: "writer-session",
        kind: "code",
        language: "typescript",
        title: "Writer helper",
        code: "export function writerHelper() {}",
        createdAt: 100,
      },
      {
        id: "writer-two",
        sessionId: "writer-session",
        kind: "html",
        language: "html",
        title: "Landing page",
        code: "<html><body>writer</body></html>",
        createdAt: 200,
      },
    ]),
  );
  localStorage.setItem(
    "hermes.native-chat.artifacts.v2.",
    JSON.stringify([
      {
        id: "default-only",
        sessionId: "default-session",
        kind: "code",
        language: "python",
        title: "Default only",
        code: "print('default')",
        createdAt: 300,
      },
    ]),
  );
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ArtifactsPage", () => {
  it("renders only the active profile's artifacts and filters by search text", async () => {
    const { default: ArtifactsPage } = await import("./ArtifactsPage");
    await render(
      <MemoryRouter>
        <ArtifactsPage />
      </MemoryRouter>,
    );

    expect(container.textContent).toContain("Writer helper");
    expect(container.textContent).toContain("Landing page");
    expect(container.textContent).not.toContain("Default only");

    const search = container.querySelector<HTMLInputElement>('input[type="search"]');
    expect(search).not.toBeNull();
    setInputValue(search!, "landing");
    expect(container.textContent).toContain("Landing page");
    expect(container.textContent).not.toContain("Writer helper");
  }, 15_000);

  it("uses the dashboard current profile when the selector is empty", async () => {
    profileMocks.profile = "";
    profileMocks.currentProfile = "ops";
    localStorage.setItem(
      "hermes.native-chat.artifacts.v2.ops",
      JSON.stringify([{
        id: "ops-one",
        sessionId: "ops-session",
        kind: "code",
        language: "python",
        title: "Ops helper",
        code: "print('ops')",
        createdAt: 400,
      }]),
    );
    const { default: ArtifactsPage } = await import("./ArtifactsPage");
    await render(<MemoryRouter><ArtifactsPage /></MemoryRouter>);

    expect(container.textContent).toContain("Ops helper");
    expect(container.textContent).not.toContain("Writer helper");
  }, 15_000);

  it("previews safely, copies, unpins, and clears only after confirmation", async () => {
    const { default: ArtifactsPage } = await import("./ArtifactsPage");
    await render(<MemoryRouter><ArtifactsPage /></MemoryRouter>);
    const writerStorageKey = getArtifactStorageKey("writer");

    const previewButton = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.includes("Preview"));
    expect(previewButton).toBeDefined();
    await act(async () => previewButton?.click());
    const preview = container.querySelector<HTMLIFrameElement>('iframe[sandbox=""]');
    expect(preview).not.toBeNull();
    expect(preview?.getAttribute("referrerpolicy")).toBe("no-referrer");
    expect(preview?.getAttribute("srcdoc")).toContain("Content-Security-Policy");
    expect(preview?.getAttribute("srcdoc")).toContain("<html>");
    await act(async () => document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(container.querySelector('iframe[sandbox=""]')).toBeNull();

    const closePreview = container.querySelector<HTMLButtonElement>('button[aria-label="Close artifact preview"]');
    await act(async () => closePreview?.click());
    expect(container.querySelector('iframe[sandbox=""]')).toBeNull();

    const copyButton = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.includes("Copy"));
    await act(async () => copyButton?.click());
    expect(clipboardMocks.copyTextToClipboard).toHaveBeenCalledWith("<html><body>writer</body></html>");

    const unpinButton = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.includes("Unpin"));
    await act(async () => unpinButton?.click());
    expect(container.textContent).not.toContain("Landing page");

    const clearButton = container.querySelector<HTMLButtonElement>('button[aria-label="Clear all pinned artifacts"]');
    expect(clearButton).not.toBeNull();
    await act(async () => clearButton?.click());
    expect(container.textContent).toContain("Clear pinned artifacts?");
    expect(localStorage.getItem(writerStorageKey)).not.toBeNull();

    const confirmClear = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.includes("Clear artifacts"));
    await act(async () => confirmClear?.click());
    expect(localStorage.getItem(writerStorageKey)).toBeNull();
    expect(localStorage.getItem("hermes.native-chat.artifacts.v2.")).not.toBeNull();
    expect(container.textContent).toContain("No pinned artifacts yet");
  }, 15_000);
});
