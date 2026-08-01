import { describe, expect, it, vi } from "vitest";

import { createPtyCompositionForwarder } from "./pty-composition";

describe("createPtyCompositionForwarder", () => {
  it("forwards committed dead-key text and suppresses xterm's duplicate onData", () => {
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");

    expect(send).toHaveBeenCalledExactlyOnceWith("ä");
    expect(forwarder.shouldForwardTerminalData("ä")).toBe(false);
    expect(forwarder.shouldForwardTerminalData("x")).toBe(true);
  });

  it("does not send an empty cancelled composition", () => {
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("");

    expect(send).not.toHaveBeenCalled();
  });
});
