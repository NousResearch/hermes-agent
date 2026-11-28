import { describe, expect, it } from "vitest";

import {
  AUTO_ROUTE_EXPLANATION,
  auxTaskIsUnpinned,
  auxTaskLabel,
  auxTaskRouteLabel,
  auxTaskRows,
  BUILTIN_AUX_TASKS,
  effectiveAuxRoute,
} from "./aux-tasks";

const builtinKeys = BUILTIN_AUX_TASKS.map((t) => t.key);

describe("auxTaskRows", () => {
  it("puts the inherited default route before executable tasks", () => {
    expect(BUILTIN_AUX_TASKS[0]).toEqual({
      key: "default",
      label: "Default",
      hint: "Shared route for unpinned tasks",
    });
  });

  it("renders only the built-ins when the backend sends none or only built-ins (older backend)", () => {
    expect(auxTaskRows(undefined).map((t) => t.key)).toEqual(builtinKeys);
    expect(auxTaskRows(null).map((t) => t.key)).toEqual(builtinKeys);
    const onlyBuiltins = [{ task: "vision", provider: "auto", model: "", base_url: "" }];
    expect(auxTaskRows(onlyBuiltins).map((t) => t.key)).toEqual(builtinKeys);
  });

  it("appends plugin-registered tasks after the built-ins with the server label and hint (#40880)", () => {
    const rows = auxTaskRows([
      { task: "vision", provider: "auto", model: "", base_url: "" },
      {
        task: "grill_tab",
        provider: "openrouter",
        model: "openai/gpt-5-mini",
        base_url: "",
        label: "Grill Tab",
        hint: "Tab-to-grill questions",
        plugin: "grill-tab",
      },
    ]);
    expect(rows.slice(0, builtinKeys.length).map((t) => t.key)).toEqual(builtinKeys);
    expect(rows.at(-1)).toEqual({ key: "grill_tab", label: "Grill Tab", hint: "Tab-to-grill questions" });
  });

  it("falls back to the key when a plugin task arrives without a label, and never duplicates a built-in", () => {
    const rows = auxTaskRows([
      { task: "curator", provider: "auto", model: "", base_url: "" },
      { task: "custom_rag", provider: "auto", model: "", base_url: "" },
    ]);
    expect(rows.filter((t) => t.key === "curator")).toHaveLength(1);
    expect(rows.at(-1)).toEqual({ key: "custom_rag", label: "custom_rag", hint: "" });
    expect(auxTaskLabel(rows.length ? [] : [], "vision")).toBe("Vision");
    expect(auxTaskLabel([{ task: "x_task", provider: "auto", model: "", base_url: "", label: "X" }], "x_task")).toBe("X");
  });

  it("does not offer the retired skills_hub slot (#40703)", () => {
    expect(builtinKeys).not.toContain("skills_hub");
    expect(auxTaskRows(undefined).map((t) => t.key)).not.toContain("skills_hub");
  });

  it("carries a plugin task's inherited base slot", () => {
    const rows = auxTaskRows([
      {
        task: "side_task",
        provider: "auto",
        model: "",
        base_url: "",
        label: "Side",
        inherit_from: "compression",
      },
    ]);
    expect(rows.at(-1)).toEqual({ key: "side_task", label: "Side", hint: "", inheritFrom: "compression" });
  });

  it("formats a built-in task's effective inherited route", () => {
    const effective = { provider: "openrouter", model: "vendor/cheap", base_url: "" };
    const assignment = { task: "compression", provider: "auto", model: "", base_url: "", effective };
    expect(effectiveAuxRoute(assignment)).toBe("openrouter · vendor/cheap");
    expect(effectiveAuxRoute({ ...assignment, effective: { provider: "auto", model: "", base_url: "" } })).toBeNull();
  });

  it("does not treat provider:auto plus an explicit model as default inheritance", () => {
    const assignment = {
      task: "compression", provider: "auto", model: "task/model", base_url: "",
      inherits_default: false,
    };
    expect(auxTaskIsUnpinned(assignment)).toBe(false);
  });

  it("labels a denied plugin base route by its actual Default source", () => {
    const meta = { key: "side_task", label: "Side", hint: "", inheritFrom: "compression" };
    const assignment = {
      task: "side_task",
      provider: "auto",
      model: "",
      base_url: "",
      inherit_from: "compression",
      inherits_default: true,
      effective: { provider: "openrouter", model: "vendor/default", base_url: "" },
    };
    expect(auxTaskRouteLabel(meta, assignment, [assignment])).toBe(
      "inherits Default · openrouter · vendor/default",
    );
  });

  it("shows a standalone plugin's declared effective route", () => {
    const meta = { key: "own_task", label: "Own", hint: "" };
    const assignment = {
      task: "own_task",
      provider: "auto",
      model: "",
      base_url: "",
      inherits_default: false,
      effective: { provider: "openrouter", model: "vendor/own", base_url: "" },
    };
    expect(auxTaskRouteLabel(meta, assignment, [assignment])).toBe(
      "openrouter · vendor/own",
    );
  });

  it("explains auto without claiming it always means main", () => {
    expect(AUTO_ROUTE_EXPLANATION).toBe(
      "auto follows Default when configured, otherwise the main model.",
    );
  });
});
