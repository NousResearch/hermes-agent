import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { en } from "@/i18n/en";
import { ja } from "@/i18n/ja";
import { consoleTruncationNotice } from "@/lib/console-output-truncation";

/**
 * T4.7 — chat truncation notice. HermesConsoleEngine caps console output at
 * `output_limit` (50000 via chat_ws) and silently appended an ASCII marker;
 * users never learned the final size. The `/api/console` output frame now
 * carries `truncated`/`omitted_bytes`, HermesConsoleModal renders a locale
 * notice plus a "Show full output" button, and the server answers
 * `{type:"full"}` with the uncapped buffer (engine.last_full_output).
 *
 * The notice lives in HermesConsoleModal — the `/api/console` socket UI — not
 * ChatPage (which is the PTY/xterm surface where truncation happens inside
 * the TUI); premise adapted in the commit message.
 */

describe("console output truncation notice (T4.7)", () => {
  it("notice text comes from the locale, not a hardcoded string", () => {
    const enNotice = consoleTruncationNotice(en, 1234);
    const jaNotice = consoleTruncationNotice(ja, 1234);
    expect(enNotice).toBe(en.app.consoleOutputTruncated.replace("{count}", "1234"));
    expect(jaNotice).toBe(ja.app.consoleOutputTruncated.replace("{count}", "1234"));
    expect(enNotice).toContain("1234");
    expect(enNotice).not.toBe(jaNotice);
  });

  it("HermesConsoleModal renders the notice and requests the full buffer", () => {
    const modal = readFileSync(
      path.resolve(process.cwd(), "src/components/HermesConsoleModal.tsx"),
      "utf8",
    );
    expect(modal).toContain("consoleTruncationNotice");
    expect(modal).toContain('sendFrame({ type: "full" })');
    expect(modal).toContain("app.consoleShowFullOutput");
  });

  it("chat_ws marks output frames truncated and serves the full buffer", () => {
    const ws = readFileSync(
      path.resolve(process.cwd(), "../hermes_cli/web_routers/chat_ws.py"),
      "utf8",
    );
    expect(ws).toContain('"truncated"');
    expect(ws).toContain('frame_type == "full"');
    const engine = readFileSync(
      path.resolve(process.cwd(), "../hermes_cli/console_engine.py"),
      "utf8",
    );
    expect(engine).toContain("last_full_output");
  });
});
