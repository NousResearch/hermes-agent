import { describe, expect, it } from "vitest";

import { nativeChatSessionCreateParams } from "./native-chat-routing";

describe("nativeChatSessionCreateParams", () => {
  it("keeps a submitted turn server-owned when the dashboard client disconnects", () => {
    const params = nativeChatSessionCreateParams(undefined, { reasoning: "auto" });

    expect(params).toMatchObject({
      close_on_disconnect: false,
      continue_on_disconnect: true,
      source: "dashboard",
    });
  });
});
