import { describe, expect, it } from "vitest";
import { classifyLine } from "./log-classify";

const TS = "2026-07-26 13:07:45,228";

describe("classifyLine", () => {
  it("classifies by the structured level token", () => {
    expect(classifyLine(`${TS} INFO run_agent: client created`)).toBe("info");
    expect(classifyLine(`${TS} DEBUG registry: scanned 12 tools`)).toBe(
      "debug",
    );
    expect(classifyLine(`${TS} WARNING hermes_state: WAL disabled`)).toBe(
      "warning",
    );
    expect(classifyLine(`${TS} ERROR gateway: connect failed`)).toBe("error");
    expect(classifyLine(`${TS} CRITICAL agent: giving up`)).toBe("error");
  });

  it("does NOT flag INFO lines whose payload mentions errors", () => {
    // Regression: these rendered red with the old substring heuristic.
    expect(
      classifyLine(
        `${TS} INFO tui_gateway.ws: ws closed peer=127.0.0.1:43212 parse_errors=0 dispatch_crashes=0 send_failures=1`,
      ),
    ).toBe("info");
    expect(classifyLine(`${TS} INFO logs: rotating errors.log`)).toBe("info");
    expect(
      classifyLine(`${TS} INFO retry: recovered from FatalError subclass`),
    ).toBe("info");
  });

  it("does NOT let payload text spoof a higher level", () => {
    expect(
      classifyLine(`${TS} INFO chat: user said "ERROR ERROR ERROR"`),
    ).toBe("info");
    expect(classifyLine(`${TS} DEBUG probe: WARNING string seen`)).toBe(
      "debug",
    );
  });

  it("falls back to word-boundary matching for untimestamped lines", () => {
    expect(classifyLine("ValueError: bad input")).toBe("info"); // no bare token
    expect(classifyLine("ERROR: something broke")).toBe("error");
    expect(classifyLine("Traceback (most recent call last):")).toBe("error");
    expect(classifyLine("  WARNING low disk")).toBe("warning");
    expect(classifyLine("errors=0 all good")).toBe("info");
    expect(classifyLine("wrote to errors.log")).toBe("info");
    expect(classifyLine("just a plain line")).toBe("info");
  });
});
