import { describe, expect, it } from "vitest";

import { filterTranscriptMessages } from "./native-chat-search";

describe("native chat transcript search", () => {
  const messages = [
    { id: "u1", role: "user" as const, text: "Explain Thai combining marks" },
    { id: "a1", role: "assistant" as const, text: "Combining marks stay intact." },
    { id: "u2", role: "user" as const, text: "A different question" },
  ];

  it("matches case-insensitively across user and assistant messages", () => {
    expect(filterTranscriptMessages(messages, "THAI").map((message) => message.id)).toEqual(["u1"]);
    expect(filterTranscriptMessages(messages, "intact").map((message) => message.id)).toEqual(["a1"]);
  });

  it("treats blank queries as an unfiltered copy", () => {
    const result = filterTranscriptMessages(messages, "  ");
    expect(result).toEqual(messages);
    expect(result).not.toBe(messages);
  });

  it("does not mutate the source array when there are no matches", () => {
    const result = filterTranscriptMessages(messages, "missing");
    expect(result).toEqual([]);
    expect(messages).toHaveLength(3);
  });
});
