/**
 * Live status badge on a plugin's sidebar row.
 *
 * A plugin declares it in its manifest (`tab.badge`), the shell polls that
 * endpoint and paints a chip while the JSON value at `path` is truthy. Kanban
 * points it at `/paused`, which lists every held BOARD: pause is per board, so
 * the chip says "PAUSED" for one and "2 PAUSED" for several instead of
 * implying the board you happen to be looking at (#141451). The shell stays
 * kanban-agnostic — same spirit as the desktop's `sidebar.nav.trailing` slot,
 * data-only because a web manifest is JSON.
 */

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";
import { fetchJSON } from "@/lib/api";
import type { PluginTabBadge } from "@/plugins/types";

const POLL_MS = 30_000;

function readPath(data: unknown, path: string): unknown {
  let node: unknown = data;
  for (const key of path.split(".")) {
    if (!node || typeof node !== "object") return undefined;
    node = (node as Record<string, unknown>)[key];
  }
  return node;
}

type BadgeState = { count: number | null; name: string | null; reason: string | null };

/** Truthy payload → state (count/name/reason for the chip); null → no badge. */
function badgeState(value: unknown, countValue: unknown): BadgeState | null {
  if (!value) return null;

  // A single-entry list (one held board) can still say WHICH board it is, so a
  // lone chip next to a plugin you are not looking at is never ambiguous;
  // several entries name themselves in the tooltip instead.
  const source = Array.isArray(value) ? (value.length === 1 ? value[0] : null) : value;
  const name =
    source && typeof source === "object" && typeof (source as { name?: unknown }).name === "string"
      ? (source as { name: string }).name || null
      : null;
  const reason =
    source && typeof source === "object" && typeof (source as { reason?: unknown }).reason === "string"
      ? (source as { reason: string }).reason || null
      : null;

  return { count: typeof countValue === "number" ? countValue : null, name, reason };
}

export function PluginNavBadge({
  badge,
  collapsed,
  label,
  labelCount,
}: {
  badge: PluginTabBadge;
  collapsed: boolean;
  label: string;
  labelCount?: string;
}) {
  const [state, setState] = useState<BadgeState | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetchJSON<unknown>(badge.endpoint)
        .then((data) => {
          if (cancelled) return;

          setState(
            badgeState(
              readPath(data, badge.path),
              badge.countPath ? readPath(data, badge.countPath) : undefined,
            ),
          );
        })
        .catch(() => {
          // A badge is decoration: a failing endpoint must never look like an
          // alert of its own, and never break the nav.
          if (!cancelled) setState(null);
        });
    };

    load();
    const timer = window.setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [badge.endpoint, badge.path, badge.countPath]);

  if (!state) return null;

  const plural = state.count !== null && state.count > 1 && labelCount;
  const text = plural ? labelCount.replace("{count}", String(state.count)) : label;
  // Name the board the chip is about, so "PAUSED" beside a plugin row is never
  // read as the board currently open.
  const title = [state.name, state.reason].filter(Boolean).join(" — ") || text;

  if (collapsed) {
    return (
      <span
        aria-label={title}
        className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-red-500"
        data-testid="plugin-nav-badge-dot"
        role="status"
        title={title}
      />
    );
  }

  return (
    <span
      className={cn(
        "ml-auto inline-flex shrink-0 items-center gap-1 rounded-full",
        "bg-red-500 px-1.5 py-px font-sans text-[0.6rem] font-semibold tracking-wide text-white normal-case",
      )}
      data-testid="plugin-nav-badge"
      role="status"
      title={title}
    >
      {text}
    </span>
  );
}
