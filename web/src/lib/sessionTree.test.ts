/**
 * Tests for web/src/lib/sessionTree.ts
 *
 * Covers: basic tree, orphan detection, worker separation, empty-field fallbacks.
 */

import { describe, it, expect } from "vitest";
import { buildSessionTree } from "./sessionTree";
import type { SessionInfo } from "./api";

function makeSession(overrides: Partial<SessionInfo> & { id: string }): SessionInfo {
  const { id, ...rest } = overrides;
  return {
    id,
    source: null,
    model: null,
    title: null,
    started_at: 0,
    ended_at: null,
    last_active: 0,
    message_count: 0,
    tool_call_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    preview: null,
    is_active: false,
    ...rest,
  };
}

describe("buildSessionTree", () => {
  it("returns empty buckets for empty input", () => {
    const { roots, workers, orphans } = buildSessionTree([]);
    expect(roots).toHaveLength(0);
    expect(workers).toHaveLength(0);
    expect(orphans).toHaveLength(0);
  });

  it("treats sessions without kind as main roots", () => {
    const s = makeSession({ id: "a" });
    const { roots, workers, orphans } = buildSessionTree([s]);
    expect(roots).toHaveLength(1);
    expect(roots[0].session.id).toBe("a");
    expect(workers).toHaveLength(0);
    expect(orphans).toHaveLength(0);
  });

  it("nests subagent under its parent", () => {
    const parent = makeSession({ id: "p", kind: "main" });
    const child = makeSession({ id: "c", kind: "subagent", parent_session_id: "p" });
    const { roots, workers, orphans } = buildSessionTree([parent, child]);
    expect(roots).toHaveLength(1);
    expect(roots[0].children).toHaveLength(1);
    expect(roots[0].children[0].session.id).toBe("c");
    expect(orphans).toHaveLength(0);
    expect(workers).toHaveLength(0);
  });

  it("puts subagent in orphans when parent is not in list", () => {
    const child = makeSession({ id: "c", kind: "subagent", parent_session_id: "missing" });
    const { roots, workers, orphans } = buildSessionTree([child]);
    expect(orphans).toHaveLength(1);
    expect(orphans[0].id).toBe("c");
    expect(roots).toHaveLength(0);
    expect(workers).toHaveLength(0);
  });

  it("separates kanban_worker sessions into workers bucket", () => {
    const main = makeSession({ id: "m", kind: "main" });
    const worker = makeSession({ id: "w", kind: "kanban_worker" });
    const { roots, workers, orphans } = buildSessionTree([main, worker]);
    expect(roots).toHaveLength(1);
    expect(workers).toHaveLength(1);
    expect(workers[0].id).toBe("w");
    expect(orphans).toHaveLength(0);
  });

  it("handles a mix of roots, subagents, orphans, and workers", () => {
    const sessions: SessionInfo[] = [
      makeSession({ id: "m1", kind: "main" }),
      makeSession({ id: "m2", kind: "main" }),
      makeSession({ id: "s1", kind: "subagent", parent_session_id: "m1" }),
      makeSession({ id: "s2", kind: "subagent", parent_session_id: "gone" }), // orphan
      makeSession({ id: "w1", kind: "kanban_worker" }),
    ];
    const { roots, workers, orphans } = buildSessionTree(sessions);
    expect(roots).toHaveLength(2);
    const m1Node = roots.find((r) => r.session.id === "m1");
    expect(m1Node?.children).toHaveLength(1);
    expect(m1Node?.children[0].session.id).toBe("s1");
    expect(orphans).toHaveLength(1);
    expect(orphans[0].id).toBe("s2");
    expect(workers).toHaveLength(1);
    expect(workers[0].id).toBe("w1");
  });

  it("filter scenario: works when parent was filtered out (all non-main become orphans)", () => {
    // Simulate search result that only returned the child (parent not in list)
    const child = makeSession({ id: "c", kind: "subagent", parent_session_id: "p" });
    const { roots, orphans } = buildSessionTree([child]);
    expect(roots).toHaveLength(0);
    expect(orphans).toHaveLength(1);
  });

  it("parent_session_id=null subagent with kind=subagent goes to orphans (no parent)", () => {
    const child = makeSession({ id: "c", kind: "subagent", parent_session_id: null });
    const { roots, orphans } = buildSessionTree([child]);
    // No parent → orphan
    expect(orphans).toHaveLength(1);
    expect(roots).toHaveLength(0);
  });
});
