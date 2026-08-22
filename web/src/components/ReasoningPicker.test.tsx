// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  saveConfig: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("lucide-react", () => ({ Brain: () => null }));
vi.mock("@nous-research/ui/ui/components/select", () => ({
  Select: ({
    children,
    value,
    disabled,
    onValueChange,
  }: {
    children?: React.ReactNode;
    value?: string;
    disabled?: boolean;
    onValueChange?: (value: string) => void;
  }) => (
    <div data-select-value={value} data-select-disabled={String(Boolean(disabled))}>
      <button
        data-select-trigger
        disabled={disabled}
        onClick={() => onValueChange?.("high")}
      />
      {children}
    </div>
  ),
  SelectOption: ({ children, value }: { children?: React.ReactNode; value?: string }) => (
    <div data-option-value={value}>{children}</div>
  ),
}));

type Config = { agent?: { reasoning_effort?: string } };

let container: HTMLDivElement;
let root: Root;

async function renderPicker(profile: string) {
  const { ReasoningPicker } = await import("./ReasoningPicker");
  await act(async () => {
    root.render(
      <ReasoningPicker
        currentModel="provider/model"
        profile={profile}
        reasoningLevels={["low", "high"]}
      />,
    );
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  vi.clearAllMocks();
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
});

describe("ReasoningPicker profile scoping", () => {
  it("ignores a late config read from the previous profile", async () => {
    const requests: Array<(value: Config) => void> = [];
    apiMocks.getConfig.mockImplementation(
      () =>
        new Promise<Config>((resolve) => {
          requests.push(resolve);
        }),
    );

    await renderPicker("profile-a");
    await renderPicker("profile-b");
    expect(requests.length).toBeGreaterThanOrEqual(2);

    await act(async () => {
      requests.at(-1)!({ agent: { reasoning_effort: "high" } });
      await Promise.resolve();
    });
    await act(async () => {
      requests[0]({ agent: { reasoning_effort: "low" } });
      await Promise.resolve();
    });

    expect(container.querySelector("[data-select-value]")?.getAttribute("data-select-value")).toBe(
      "high",
    );
  });

  it("resets to the neutral effort when a new profile read fails", async () => {
    apiMocks.getConfig.mockResolvedValue({ agent: { reasoning_effort: "low" } });
    await renderPicker("profile-a");
    apiMocks.getConfig.mockRejectedValueOnce(new Error("profile read failed"));

    await renderPicker("profile-b");

    expect(container.querySelector("[data-select-value]")?.getAttribute("data-select-value")).toBe(
      "medium",
    );
    expect(container.querySelector("[data-select-disabled]")?.getAttribute("data-select-disabled")).toBe(
      "false",
    );
  });

  it("ignores a late save failure from the previous profile", async () => {
    apiMocks.getConfig.mockResolvedValue({ agent: { reasoning_effort: "low" } });
    await renderPicker("profile-a");

    let resolveSaveRead!: (value: Config) => void;
    apiMocks.getConfig.mockImplementationOnce(
      () =>
        new Promise<Config>((resolve) => {
          resolveSaveRead = resolve;
        }),
    );
    apiMocks.getConfig.mockResolvedValue({ agent: { reasoning_effort: "high" } });
    apiMocks.saveConfig.mockRejectedValueOnce(new Error("stale save"));

    await act(async () => {
      container.querySelector<HTMLButtonElement>("[data-select-trigger]")!.click();
      await Promise.resolve();
    });
    await renderPicker("profile-b");

    await act(async () => {
      resolveSaveRead({ agent: { reasoning_effort: "low" } });
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.querySelector("[data-select-value]")?.getAttribute("data-select-value")).toBe(
      "high",
    );
    expect(container.querySelector("[data-select-disabled]")?.getAttribute("data-select-disabled")).toBe(
      "false",
    );
  });
});
