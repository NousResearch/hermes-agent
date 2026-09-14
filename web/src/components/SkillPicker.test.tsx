// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getSkills: vi.fn(async () => [
    { name: "ffmpeg", description: "video encode", usage: 1 },
    { name: "hyperframes", description: "frames", usage: 9 },
  ]),
}));

vi.mock("@/lib/api", () => ({
  api: { getSkills: apiMocks.getSkills },
}));

vi.mock("@nous-research/ui/ui/components/card", () => ({
  Card: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
}));

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

function setInputValue(input: HTMLInputElement, value: string) {
  const proto = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  );
  proto?.set?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

beforeEach(() => {
  (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;
  vi.clearAllMocks();
  apiMocks.getSkills.mockResolvedValue([
    { name: "ffmpeg", description: "video encode", usage: 1 },
    { name: "hyperframes", description: "frames", usage: 9 },
  ]);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("SkillPicker", () => {
  it("renders skills returned by the API", async () => {
    const { SkillPicker } = await import("./SkillPicker");

    await render(<SkillPicker />);

    await vi.waitFor(() => {
      expect(container.textContent).toContain("ffmpeg");
      expect(container.textContent).toContain("hyperframes");
    });
  });

  it("filters the list when the search box is typed", async () => {
    const { SkillPicker } = await import("./SkillPicker");

    await render(<SkillPicker />);
    await vi.waitFor(() => expect(container.textContent).toContain("ffmpeg"));

    const input = container.querySelector("input");
    expect(input).toBeTruthy();

    await act(async () => {
      setInputValue(input as HTMLInputElement, "video");
    });

    expect(container.textContent).toContain("ffmpeg");
    expect(container.textContent).not.toContain("hyperframes");
  });

  it("calls onLaunch with /ffmpeg when that row is clicked", async () => {
    const onLaunch = vi.fn();
    const { SkillPicker } = await import("./SkillPicker");

    await render(<SkillPicker onLaunch={onLaunch} />);
    await vi.waitFor(() => expect(container.textContent).toContain("ffmpeg"));

    const row = [...container.querySelectorAll("button")].find((el) =>
      el.textContent?.includes("ffmpeg"),
    );
    expect(row).toBeTruthy();

    await act(async () => {
      row!.click();
    });

    expect(onLaunch).toHaveBeenCalledWith("/ffmpeg");
  });
});
