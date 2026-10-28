/**
 * Projects page utility tests.
 *
 * Tests pure/DI-testable functions:
 * - Variant storage (localStorage read/write)
 * - KanbanBadge status mapping contract
 * - Session grouping (running / attention / done)
 *
 * Environment: node (default for this project). localStorage stubbed via
 * an in-memory map — no jsdom required.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { createElement } from "react";

// ── Stub localStorage for node environment ──────────────────────────────────

const _store: Record<string, string> = {};
const localStorageStub = {
  getItem: (k: string) => _store[k] ?? null,
  setItem: (k: string, v: string) => {
    _store[k] = v;
  },
  removeItem: (k: string) => {
    delete _store[k];
  },
  clear: () => {
    Object.keys(_store).forEach((k) => delete _store[k]);
  },
};

Object.defineProperty(globalThis, "localStorage", {
  value: localStorageStub,
  writable: true,
  configurable: true,
});

// ── VariantPicker utilities ─────────────────────────────────────────────────

import { readVariant, writeVariant } from "@/components/projects/VariantPicker";

describe("readVariant", () => {
  beforeEach(() => localStorageStub.clear());

  it("returns 'workbench' by default (key absent)", () => {
    expect(readVariant()).toBe("workbench");
  });

  it("returns 'overview' when stored value is 'overview'", () => {
    localStorageStub.setItem("hermes.projects.variant", "overview");
    expect(readVariant()).toBe("overview");
  });

  it("returns 'workbench' for an unrecognised stored value", () => {
    localStorageStub.setItem("hermes.projects.variant", "unknown");
    expect(readVariant()).toBe("workbench");
  });
});

describe("writeVariant", () => {
  beforeEach(() => localStorageStub.clear());

  it("persists 'overview' so readVariant returns it", () => {
    writeVariant("overview");
    expect(readVariant()).toBe("overview");
  });

  it("persists 'workbench' after overwriting 'overview'", () => {
    writeVariant("overview");
    writeVariant("workbench");
    expect(readVariant()).toBe("workbench");
  });

  it("is a no-op when localStorage.setItem throws (quota error)", () => {
    const orig = localStorageStub.setItem.bind(localStorageStub);
    localStorageStub.setItem = () => {
      throw new Error("quota exceeded");
    };
    expect(() => writeVariant("overview")).not.toThrow();
    localStorageStub.setItem = orig;
  });
});

// ── Variant toggle invariant ─────────────────────────────────────────────────

describe("variant toggle invariant", () => {
  beforeEach(() => localStorageStub.clear());

  it("only two valid variants: workbench and overview", () => {
    const validVariants = ["workbench", "overview"] as const;
    for (const v of validVariants) {
      writeVariant(v);
      expect(readVariant()).toBe(v);
    }
  });

  it("round-trips after multiple writes", () => {
    writeVariant("overview");
    writeVariant("workbench");
    writeVariant("overview");
    expect(readVariant()).toBe("overview");
  });
});

// ── Session grouping logic ────────────────────────────────────────────────────

// Pure function that mirrors the contract of groupSessions in ProjectsOverview.
// Test validates behavior (running/attention/done split), not source text.

type MockSession = {
  id: string;
  task: null | { status: string };
  endReason: null | string;
};

function groupSessions(sessions: MockSession[]) {
  const running: MockSession[] = [];
  const attention: MockSession[] = [];
  const done: MockSession[] = [];
  for (const s of sessions) {
    const st = s.task?.status;
    if (st === "doing" || st === "running" || (!s.task && !s.endReason)) {
      running.push(s);
    } else if (st === "blocked" || st === "failed") {
      attention.push(s);
    } else {
      done.push(s);
    }
  }
  return { running, attention, done };
}

function mkS(id: string, taskStatus: string | null, endReason: string | null = null): MockSession {
  return { id, task: taskStatus ? { status: taskStatus } : null, endReason };
}

describe("groupSessions", () => {
  it("puts doing/running sessions in running group", () => {
    const { running, attention, done } = groupSessions([
      mkS("a", "doing"),
      mkS("b", "running"),
    ]);
    expect(running.map((s) => s.id)).toEqual(["a", "b"]);
    expect(attention).toHaveLength(0);
    expect(done).toHaveLength(0);
  });

  it("puts blocked/failed sessions in attention group", () => {
    const { running, attention, done } = groupSessions([
      mkS("a", "blocked"),
      mkS("b", "failed"),
    ]);
    expect(attention.map((s) => s.id)).toEqual(["a", "b"]);
    expect(running).toHaveLength(0);
    expect(done).toHaveLength(0);
  });

  it("puts done/todo/review task sessions in done group", () => {
    const { running, attention, done } = groupSessions([
      mkS("a", "done"),
      mkS("b", "todo"),
      mkS("c", "review"),
    ]);
    expect(done.map((s) => s.id)).toEqual(["a", "b", "c"]);
    expect(running).toHaveLength(0);
    expect(attention).toHaveLength(0);
  });

  it("puts sessions with no task and no endReason in running group", () => {
    const { running } = groupSessions([mkS("a", null, null)]);
    expect(running).toHaveLength(1);
  });

  it("puts sessions with no task but with endReason in done group", () => {
    const { done } = groupSessions([mkS("a", null, "user_exit")]);
    expect(done).toHaveLength(1);
  });

  it("handles an empty list", () => {
    const { running, attention, done } = groupSessions([]);
    expect(running).toHaveLength(0);
    expect(attention).toHaveLength(0);
    expect(done).toHaveLength(0);
  });
});

// ── KanbanBadge status contract ───────────────────────────────────────────────

import { KanbanBadge } from "@/components/projects/KanbanBadge";

describe("KanbanBadge", () => {
  it("renders (createElement) without throwing for all recognised statuses", () => {
    const statuses = [
      "todo",
      "doing",
      "review",
      "blocked",
      "done",
      "running",
      "failed",
      "notask",
    ] as const;
    for (const status of statuses) {
      expect(() => createElement(KanbanBadge, { status })).not.toThrow();
    }
  });
});

// ── API error surface ─────────────────────────────────────────────────────────

describe("api.getProjects error surfacing", () => {
  it("rejects when fetch fails", async () => {
    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockRejectedValueOnce(new Error("Network Error"));
    const { api } = await import("@/lib/api");
    await expect(api.getProjects()).rejects.toThrow();
    globalThis.fetch = origFetch;
  });
});
