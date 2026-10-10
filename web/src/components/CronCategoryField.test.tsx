// @vitest-environment jsdom
/**
 * The editor's category picker: it offers the labels already in use and lets a
 * new one be typed and confirmed, so jobs do not drift into near-duplicate
 * categories ("family", "Family ", "familiy").
 *
 * The harness is stateful on purpose — the field filters on its `value` prop, so
 * a test that does not echo the change back (as the job form does) would never
 * see the suggestions move.
 */
import { useState } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/i18n", () => ({
  useI18n: () => ({ t: { cron: {} }, locale: "en" }),
}));

import { CronCategoryField } from "./CronCategoryField";

const OPTIONS = ["Backups", "Family", "Homelab & media"];

let container: HTMLDivElement;
let root: Root;
let changes: string[];

function Harness({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial);
  return (
    <CronCategoryField
      id="cat"
      value={value}
      options={OPTIONS}
      onChange={(next) => {
        changes.push(next);
        setValue(next);
      }}
    />
  );
}

beforeEach(() => {
  changes = [];
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

async function render(initial = ""): Promise<HTMLInputElement> {
  await act(async () => {
    root.render(<Harness initial={initial} />);
  });
  return container.querySelector("input")!;
}

function rows(kind: "option" | "create"): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(`[data-testid="cron-category-${kind}"]`),
  );
}

async function focusInput(input: HTMLInputElement): Promise<void> {
  await act(async () => {
    input.focus();
    input.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
  });
}

/** Type like a browser does: through the native value setter, or React's value
 *  tracker sees no change and the controlled input never updates. */
async function type(input: HTMLInputElement, text: string): Promise<void> {
  await focusInput(input);
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    setter.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("CronCategoryField", () => {
  it("closes its suggestion list until the field is focused", async () => {
    await render();
    expect(rows("option")).toHaveLength(0);

    await focusInput(container.querySelector("input")!);
    expect(rows("option").map((el) => el.textContent)).toEqual(OPTIONS);
  });

  it("filters the existing labels as you type and reports the pick", async () => {
    const input = await render();
    await type(input, "fam");

    expect(rows("option")).toHaveLength(1);
    expect(rows("option")[0].textContent).toContain("Family");
    expect(container.querySelector<HTMLInputElement>("input")!.value).toBe("fam");

    await act(async () => {
      rows("option")[0].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(changes).toEqual(["fam", "Family"]);
    expect(container.querySelector<HTMLInputElement>("input")!.value).toBe("Family");
  });

  it("offers to create a label that does not exist yet", async () => {
    const input = await render();
    await type(input, "Teaching");

    expect(rows("option")).toHaveLength(0);
    expect(rows("create")).toHaveLength(1);
    expect(rows("create")[0].textContent).toContain("Teaching");

    await act(async () => {
      rows("create")[0].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(container.querySelector<HTMLInputElement>("input")!.value).toBe("Teaching");
  });

  it("does not offer to create a label that already exists, ignoring case", async () => {
    const input = await render("family"); // another spelling of an existing label
    await focusInput(input);

    expect(rows("create")).toHaveLength(0);
    expect(rows("option").map((el) => el.textContent)).toContain("Family");
  });

  it("shows the job's category as the single selected option, and says so", async () => {
    const input = await render("Family");
    await focusInput(input);

    const options = rows("option");
    const selected = options.filter((el) => el.getAttribute("aria-selected") === "true");
    expect(selected).toHaveLength(1); // one category per job, never a set
    expect(selected[0].textContent).toContain("Family");
    expect(selected[0].textContent).toContain("current");
    // the replace semantics are stated in the list itself
    expect(container.textContent).toContain("picking another replaces it");
  });

  it("does not claim replace semantics when the job has no label yet", async () => {
    const input = await render();
    await focusInput(input);

    expect(container.textContent).not.toContain("picking another replaces it");
    expect(rows("option").every((el) => el.getAttribute("aria-selected") === "false")).toBe(true);
  });

  it("clears the value with the × control", async () => {
    await render("Family");
    const clear = container.querySelector<HTMLElement>('[aria-label="Clear the category"]');
    expect(clear).toBeTruthy();

    await act(async () => {
      clear!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(changes).toEqual([""]);
    expect(container.querySelector<HTMLInputElement>("input")!.value).toBe("");
  });
});
