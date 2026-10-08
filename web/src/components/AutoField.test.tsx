// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";

import { AutoField } from "./AutoField";

let container: HTMLDivElement;
let root: Root;
let saved: unknown;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  saved = undefined;
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

async function renderField(value: unknown, schema: Record<string, unknown>, schemaKey = "context.external_files") {
  function Harness() {
    const [current, setCurrent] = useState(value);
    return <AutoField schemaKey={schemaKey} schema={schema} value={current} onChange={(next) => {
      saved = next;
      setCurrent(next);
    }} />;
  }
  await act(async () => root.render(<Harness />));
}

async function typeValue(field: HTMLTextAreaElement | HTMLInputElement, value: string) {
  const prototype = field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(field, value);
  await act(async () => field.dispatchEvent(new Event("input", { bubbles: true })));
}

describe("external context files editor", () => {
  it("preserves commas, in-progress newlines and spaces, then cleans empty lines on blur", async () => {
    await renderField(["~/rules,team.md"], { type: "list", editor: "lines" });
    const field = container.querySelector("textarea")!;
    expect(field.value).toBe("~/rules,team.md");
    await typeValue(field, "~/rules,team.md\n ~/My Rules.md \n");
    expect(field.value).toBe("~/rules,team.md\n ~/My Rules.md \n");
    expect(saved).toEqual(["~/rules,team.md", " ~/My Rules.md ", ""]);
    await act(async () => field.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
    expect(saved).toEqual(["~/rules,team.md", "~/My Rules.md"]);
    expect(field.value).toBe("~/rules,team.md\n~/My Rules.md");
    await typeValue(field, "");
    await act(async () => field.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
    expect(saved).toEqual([]);
  });

  it("keeps generic lists comma-separated and nested values editable", async () => {
    await renderField(["web", "terminal"], { type: "list" }, "platform_toolsets.cli");
    expect(container.querySelector("textarea")).toBeNull();
    const field = container.querySelector("input")!;
    expect(field.value).toBe("web, terminal");
    await typeValue(field, "web, file");
    expect(saved).toEqual(["web", "file"]);
    await renderField([{ name: "demo", env: { mode: "off" } }], { type: "list" }, "custom_providers");
    const fields = container.querySelectorAll("input");
    expect(Array.from(fields, (input) => input.value)).toEqual(["demo", "off"]);
    await typeValue(fields[1], "on");
    expect(saved).toEqual([{ name: "demo", env: { mode: "on" } }]);
  });
});
