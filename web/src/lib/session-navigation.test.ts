import { describe, expect, it } from "vitest";

import { buildChatResumePath } from "./session-navigation";

describe("session navigation", () => {
  it("builds a safe chat resume URL for a session id", () => {
    expect(buildChatResumePath("session/with spaces?"))
      .toBe("/chat?resume=session%2Fwith%20spaces%3F");
  });
});
