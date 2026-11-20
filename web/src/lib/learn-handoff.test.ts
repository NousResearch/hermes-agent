import { describe, expect, it, vi } from "vitest";

import {
  buildLearnNavigationState,
  createLearnHandoffController,
  formatLearnCommand,
  hasLearnSeedState,
  type LearnSocketLike,
  normalizeLearnSeed,
  readLearnSeedFromState,
  stripLearnSeedFromState,
} from "./learn-handoff";

describe("normalizeLearnSeed", () => {
  it("keeps ordinary skill descriptions intact", () => {
    expect(normalizeLearnSeed("TypeScript refactor workflow")).toBe(
      "TypeScript refactor workflow",
    );
  });

  it("collapses C0, DEL, and line-separator control characters to a single line", () => {
    expect(normalizeLearnSeed("notes\r/exit\nnext\u0007")).toBe(
      "notes /exit next",
    );
    expect(normalizeLearnSeed("\u0015!whoami")).toBe("!whoami");
    expect(normalizeLearnSeed("\u007f\u007f\u007fnotes\u2028more\u2029end")).toBe(
      "notes more end",
    );
  });

  it("drops 7-bit and 8-bit CSI ANSI sequences without leaving parameters behind", () => {
    expect(normalizeLearnSeed("notes \u001b[31mred\u001b[0m")).toBe("notes red");
    expect(normalizeLearnSeed("notes \u009b31mred\u009b0m")).toBe("notes red");
  });

  it("drops two-byte Fp/Fe/Fs and nF VT escape sequences cleanly", () => {
    expect(normalizeLearnSeed("\u001b7note\u001b8")).toBe("note");
    expect(normalizeLearnSeed("before \u001bc after")).toBe("before after");
    expect(normalizeLearnSeed("keypad \u001b= mode \u001b> done")).toBe(
      "keypad mode done",
    );
    expect(normalizeLearnSeed("charset \u001b(B ok")).toBe("charset ok");
  });

  it("drops OSC and DCS/APC string sequences including their payloads", () => {
    expect(normalizeLearnSeed("before \u001b]52;c;YWJj\u0007 after")).toBe(
      "before after",
    );
    expect(normalizeLearnSeed("before \u001b]0;window title\u001b\\ after")).toBe(
      "before after",
    );
    expect(normalizeLearnSeed("before \u001bP1+q544e\u001b\\ after")).toBe(
      "before after",
    );
    expect(normalizeLearnSeed("before \u001b]52;c;unterminated")).toBe("before");
  });

  it("strips BiDi directional marks, overrides, and isolate controls", () => {
    expect(normalizeLearnSeed("skill \u202eoverride\u202c text")).toBe(
      "skill override text",
    );
    expect(normalizeLearnSeed("ltr\u200e rtl\u200f alm\u061c end")).toBe(
      "ltr rtl alm end",
    );
  });

  it("returns null for empty, nullish, or control-only values", () => {
    expect(normalizeLearnSeed(null)).toBeNull();
    expect(normalizeLearnSeed(undefined)).toBeNull();
    expect(normalizeLearnSeed(" \r\n\t\u001b\u0015\u007f ")).toBeNull();
    expect(normalizeLearnSeed("\u001b[31m\u001b[0m\u001b7\u001bc")).toBeNull();
  });
});

describe("formatLearnCommand", () => {
  it("prefixes normalized text with /learn", () => {
    expect(formatLearnCommand("URL: https://example.com; notes\nline2")).toBe(
      "/learn URL: https://example.com; notes line2",
    );
  });

  it("returns null when normalization leaves nothing to send", () => {
    expect(formatLearnCommand("\u0015\u007f\r\n")).toBeNull();
  });
});

describe("router-state learn handoff helpers", () => {
  it("builds normalized navigation state and rejects empty input", () => {
    expect(buildLearnNavigationState("  local source: ./docs\nnotes  ")).toEqual({
      learnSeed: "local source: ./docs notes",
    });
    expect(buildLearnNavigationState(" \r\n\u0015 ")).toBeNull();
  });

  it("reads and normalizes learnSeed only from plain object router state", () => {
    expect(hasLearnSeedState(null)).toBe(false);
    expect(hasLearnSeedState(undefined)).toBe(false);
    expect(hasLearnSeedState(["learnSeed"])).toBe(false);
    expect(hasLearnSeedState({ other: 1 })).toBe(false);
    expect(hasLearnSeedState({ learnSeed: "test" })).toBe(true);

    expect(readLearnSeedFromState({ learnSeed: "a\u0015b\u001b[31mc\u001b[0m" })).toBe(
      "a b c",
    );
    expect(readLearnSeedFromState({ learnSeed: 42 })).toBeNull();
    expect(readLearnSeedFromState(null)).toBeNull();
  });

  it("strips only learnSeed from router state while preserving sibling keys", () => {
    expect(stripLearnSeedFromState({ learnSeed: "abc" })).toBeNull();
    expect(stripLearnSeedFromState({ learnSeed: "abc", from: "/skills" })).toEqual({
      from: "/skills",
    });
    expect(stripLearnSeedFromState({ from: "/skills" })).toEqual({
      from: "/skills",
    });
    expect(stripLearnSeedFromState(null)).toBeNull();
  });
});

describe("createLearnHandoffController", () => {
  function makeSocket(readyState = 1): {
    socket: LearnSocketLike;
    sent: string[];
    setReadyState: (n: number) => void;
  } {
    const sent: string[] = [];
    let state = readyState;
    const socket: LearnSocketLike = {
      get readyState() {
        return state;
      },
      send(data: string) {
        sent.push(data);
      },
    };
    return {
      socket,
      sent,
      setReadyState(next: number) {
        state = next;
      },
    };
  }

  it("waits for composer readiness on cold start, then flushes once", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const { socket, sent } = makeSocket(1);

      expect(ctrl.enqueue("docs\u001b[31m guide\u001b[0m")).toBe(true);
      ctrl.onSocketOpen(socket);
      vi.advanceTimersByTime(799);
      expect(sent).toEqual([]);

      vi.advanceTimersByTime(1);
      expect(sent).toEqual(["/learn docs guide\r"]);

      // Re-opening later does not replay an already-delivered command.
      ctrl.onSocketOpen(socket);
      vi.advanceTimersByTime(800);
      expect(sent).toEqual(["/learn docs guide\r"]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not inject or queue a handoff when a PTY socket is already open", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const first = makeSocket(1);
      const second = makeSocket(1);

      ctrl.onSocketOpen(first.socket);
      vi.advanceTimersByTime(800);
      expect(first.sent).toEqual([]);

      // Warm return while the live composer may contain an unfinished draft.
      expect(ctrl.enqueue("second skill")).toBe(false);
      vi.advanceTimersByTime(1000);
      expect(first.sent).toEqual([]);

      // Nor should that skipped warm handoff fire unexpectedly on a later reconnect.
      ctrl.onTransportTeardown();
      ctrl.onSocketOpen(second.socket);
      vi.advanceTimersByTime(800);
      expect(second.sent).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("cancels a pending cold-start handoff if the user begins typing", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const { socket, sent } = makeSocket(1);

      expect(ctrl.enqueue("cold skill")).toBe(true);
      ctrl.onSocketOpen(socket);
      vi.advanceTimersByTime(300);
      ctrl.onUserInput();
      vi.advanceTimersByTime(800);
      expect(sent).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("retains an unsent handoff across transient transport teardown and delivers once on retry", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const first = makeSocket(1);
      const second = makeSocket(1);
      const third = makeSocket(1);

      expect(ctrl.enqueue("retry skill")).toBe(true);
      ctrl.onSocketOpen(first.socket);
      vi.advanceTimersByTime(400);
      ctrl.onTransportTeardown();
      vi.advanceTimersByTime(800);
      expect(first.sent).toEqual([]);

      ctrl.onSocketOpen(second.socket);
      vi.advanceTimersByTime(800);
      expect(second.sent).toEqual(["/learn retry skill\r"]);

      // A later reconnect after delivery must never replay the command.
      ctrl.onTransportTeardown();
      ctrl.onSocketOpen(third.socket);
      vi.advanceTimersByTime(800);
      expect(third.sent).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("ignores delayed close callbacks from a retired socket", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const socketA = makeSocket(1);
      const socketB = makeSocket(1);

      expect(ctrl.enqueue("handoff")).toBe(true);
      ctrl.onSocketOpen(socketA.socket);
      ctrl.onSocketOpen(socketB.socket);
      ctrl.onSocketClose(socketA.socket);

      vi.advanceTimersByTime(800);
      expect(socketA.sent).toEqual([]);
      expect(socketB.sent).toEqual(["/learn handoff\r"]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("drops queued handoffs on reset (scope change or unmount)", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const first = makeSocket(1);
      const second = makeSocket(1);

      ctrl.enqueue("pending skill");
      ctrl.onSocketOpen(first.socket);
      vi.advanceTimersByTime(400);
      ctrl.reset();
      vi.advanceTimersByTime(800);
      expect(first.sent).toEqual([]);

      ctrl.onSocketOpen(second.socket);
      vi.advanceTimersByTime(800);
      expect(second.sent).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not flush to a closed socket even if timer callback runs", () => {
    vi.useFakeTimers();
    try {
      const ctrl = createLearnHandoffController({ delayMs: 800 });
      const { socket, sent, setReadyState } = makeSocket(1);

      ctrl.enqueue("skill");
      ctrl.onSocketOpen(socket);
      setReadyState(3);
      vi.advanceTimersByTime(800);
      expect(sent).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });
});
