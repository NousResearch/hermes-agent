import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { SessionInfo } from "@/lib/api";

import { SessionActivityLine } from "./SessionsPage_row_text";

const row = (
  overrides: Partial<SessionInfo> & { last_activity_description?: string | null },
) => ({
  id: "s1", source: "cli", model: null, title: "Busy", started_at: 1, ended_at: null,
  last_active: 1, is_active: false, message_count: 2, tool_call_count: 0,
  input_tokens: 0, output_tokens: 0, preview: null, ...overrides,
});

const html = (session: ReturnType<typeof row>) => renderToStaticMarkup(<SessionActivityLine session={session} />);

describe("SessionActivityLine", () => {
  it("shows a live session's current activity", () => {
    expect(
      html(row({ is_active: true, last_activity_description: "executing tool: terminal" })),
    ).toContain(">executing tool: terminal</p>");
  });

  it("renders nothing for an idle session, even with a stored label", () => {
    expect(html(row({ is_active: false, last_activity_description: "executing tool: terminal" }))).toBe("");
  });
});
