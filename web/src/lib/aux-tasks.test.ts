import { describe, expect, it } from "vitest";

import { auxTaskLabel, auxTaskMeta, auxTaskRows, BUILTIN_AUX_TASKS } from "./aux-tasks";

const builtinKeys = BUILTIN_AUX_TASKS.map((t) => t.key);

describe("auxTaskRows", () => {
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
});

describe("auxTaskMeta", () => {
  it("labels a built-in from the table and a plugin task from its own metadata", () => {
    expect(auxTaskMeta([], "vision")).toEqual({ label: "Vision", hint: "Image analysis" });
    expect(
      auxTaskMeta(
        [{ task: "rag", provider: "auto", model: "", base_url: "", label: "RAG", hint: "Retrieval" }],
        "rag",
      ),
    ).toEqual({ label: "RAG", hint: "Retrieval" });
  });

  it("title-cases an unlabelled key so a newer backend's task still reads as a name", () => {
    expect(auxTaskMeta([], "custom_rag_task").label).toBe("Custom Rag Task");
    expect(auxTaskMeta([{ task: "custom_rag", provider: "auto", model: "", base_url: "" }], "custom_rag")).toEqual({
      label: "Custom Rag",
      hint: "Auxiliary model task",
    });
  });
});
