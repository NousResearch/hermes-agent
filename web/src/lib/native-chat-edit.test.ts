import { describe, expect, it } from "vitest";

import {
  applyEditedTranscript,
  buildEditSubmitParams,
  parseDurableRowId,
  type EditableTranscriptMessage,
  type EditSubmitResponse,
} from "./native-chat-edit";

describe("native chat true edit", () => {
  it("accepts only finite non-negative durable row ids", () => {
    expect(parseDurableRowId(42)).toBe(42);
    expect(parseDurableRowId("42")).toBe(42);
    expect(parseDurableRowId("42.5")).toBeUndefined();
    expect(parseDurableRowId(-1)).toBeUndefined();
    expect(parseDurableRowId(true)).toBeUndefined();
    expect(parseDurableRowId("row-42")).toBeUndefined();
  });

  it("builds a confirmed row-id truncation payload and survivor rebind list", () => {
    const messages: EditableTranscriptMessage[] = [
      { id: "11", role: "user", text: "first", rowId: 11 },
      { id: "12", role: "assistant", text: "answer" },
      { id: "13", role: "user", text: "second", rowId: 13 },
    ];

    expect(buildEditSubmitParams("runtime-1", messages[2], "revised", messages)).toEqual({
      session_id: "runtime-1",
      text: "revised",
      truncate_before_row_id: 13,
      confirm_truncate: true,
      rebind_survivor_row_ids: [11, 13],
    });
  });

  it("adds an explicit empty-truncation confirmation for the first user turn", () => {
    const target: EditableTranscriptMessage = { id: "11", role: "user", text: "first", rowId: 11 };
    expect(buildEditSubmitParams("runtime-1", target, "revised", [target])).toMatchObject({
      confirm_truncate: true,
      confirm_empty_truncate: true,
    });
  });

  it("preserves existing row ids when an older gateway omits rebind fields", () => {
    const messages: EditableTranscriptMessage[] = [
      { id: "11", role: "user", text: "first", rowId: 11 },
      { id: "12", role: "assistant", text: "answer" },
      { id: "13", role: "user", text: "second", rowId: 13 },
    ];
    expect(applyEditedTranscript(messages, "13", "revised", {}, "edited-1")).toEqual([
      { id: "11", role: "user", text: "first", rowId: 11 },
      { id: "12", role: "assistant", text: "answer" },
      { id: "edited-1", role: "user", text: "revised" },
    ]);
  });

  it("keeps the prefix, rebinds survivor rows, and replaces the selected tail", () => {
    const messages: EditableTranscriptMessage[] = [
      { id: "11", role: "user", text: "first", rowId: 11 },
      { id: "12", role: "assistant", text: "answer" },
      { id: "13", role: "user", text: "second", rowId: 13 },
      { id: "14", role: "assistant", text: "later" },
    ];
    const response: EditSubmitResponse = {
      survivor_user_row_ids: [101],
      survivor_row_id_map: { "11": 101, "13": null },
    };

    expect(applyEditedTranscript(messages, "13", "revised", response, "edited-1")).toEqual([
      { id: "101", role: "user", text: "first", rowId: 101 },
      { id: "12", role: "assistant", text: "answer" },
      { id: "edited-1", role: "user", text: "revised" },
    ]);
  });
});
