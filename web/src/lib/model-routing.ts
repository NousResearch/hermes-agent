/**
 * Pure logic for the Models → Model routing block.
 *
 * Kept React-free (and network-free) so the fallback-chain reducers and the
 * schema-gating predicates can be unit-tested directly. The card component only
 * wires these to the generic config GET/save path — the block never talks to a
 * bespoke endpoint.
 */

/**
 * One fallback hop. This is the YAML shape the backend accepts for both the
 * top-level `fallback_providers` list and the legacy `fallback_model`
 * single-dict/chain (see `hermes_cli/fallback_config.py::_iter_fallback_entries`):
 * `provider` + `model` required, `base_url` / `api_mode` optional.
 */
export interface FallbackRoute {
  provider: string;
  model: string;
  base_url?: string;
  api_mode?: string;
}

/**
 * Schema keys the routing block reads. Each control is gated on the served
 * config schema exposing its key (same rule as the config pages) — we never
 * invent a control for a key the backend does not serve.
 */
export const ROUTING_KEYS = {
  /** Subagent PREFERRED route — provider half. */
  subagentProvider: "delegation.provider",
  /** Subagent PREFERRED route — model half. */
  subagentModel: "delegation.model",
  /** Subagent fallback chain (null = inherit parent chain, [] = disable). */
  subagentFallback: "delegation.fallback_providers",
  /** MAIN agent fallback — legacy single dict OR chain list. */
  mainFallback: "fallback_model",
  /** Live re-read of the subagent route before each request. */
  hotReload: "delegation.hot_reload_model",
} as const;

/** True when the served config schema knows this key (controls gate on it). */
export function hasSchemaKey(
  schema: Record<string, unknown> | null | undefined,
  key: string,
): boolean {
  return !!schema && key in schema;
}

/**
 * Whether the MAIN-agent fallback control can render.
 *
 * The canonical key the spec names is `fallback_model`, but it is only a
 * commented-out optional in DEFAULT_CONFIG, so a served schema usually exposes
 * just the newer top-level list `fallback_providers`. Either one proves the
 * feature exists, so the block renders when either key is present; it always
 * edits `ROUTING_KEYS.mainFallback` (which the backend accepts as a dict OR a
 * chain list and merges into the effective chain).
 */
export function hasMainFallbackSupport(
  schema: Record<string, unknown> | null | undefined,
): boolean {
  return (
    hasSchemaKey(schema, ROUTING_KEYS.mainFallback) ||
    hasSchemaKey(schema, "fallback_providers")
  );
}

/**
 * Whether the routing block has anything to render for this schema — i.e. at
 * least one of the keys it edits is served. The card uses this to bail out, and
 * the Models page uses the same predicate so its surrounding section (heading /
 * Save button) appears under exactly the same condition.
 */
export function showsRoutingBlock(
  schema: Record<string, unknown> | null | undefined,
): boolean {
  return (
    hasSchemaKey(schema, ROUTING_KEYS.subagentProvider) ||
    hasSchemaKey(schema, ROUTING_KEYS.subagentModel) ||
    hasSchemaKey(schema, ROUTING_KEYS.subagentFallback) ||
    hasMainFallbackSupport(schema) ||
    hasSchemaKey(schema, ROUTING_KEYS.hotReload)
  );
}

function asScalar(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

/**
 * Normalize any served value into an editable route list.
 *
 * Accepts a single dict, a list of dicts, or null/undefined/malformed (→ `[]`).
 * Partial rows are preserved so an in-progress edit survives a re-render; a
 * non-object entry is skipped.
 */
export function parseFallbackRoutes(value: unknown): FallbackRoute[] {
  const candidates = Array.isArray(value)
    ? value
    : value !== null && typeof value === "object"
      ? [value]
      : [];
  const routes: FallbackRoute[] = [];
  for (const entry of candidates) {
    if (entry === null || typeof entry !== "object" || Array.isArray(entry)) continue;
    const record = entry as Record<string, unknown>;
    const route: FallbackRoute = {
      provider: asScalar(record.provider),
      model: asScalar(record.model),
    };
    const baseUrl = asScalar(record.base_url).trim();
    if (baseUrl) route.base_url = baseUrl;
    const apiMode = asScalar(record.api_mode).trim();
    if (apiMode) route.api_mode = apiMode;
    routes.push(route);
  }
  return routes;
}

/**
 * Serialize the editor's rows back to the YAML list shape. All-blank rows are
 * dropped (the backend would discard them anyway); every other row is kept so
 * nothing the user typed is silently lost on save.
 */
export function serializeFallbackRoutes(routes: FallbackRoute[]): Array<Record<string, string>> {
  const out: Array<Record<string, string>> = [];
  for (const route of routes) {
    const provider = (route.provider ?? "").trim();
    const model = (route.model ?? "").trim();
    const baseUrl = (route.base_url ?? "").trim();
    const apiMode = (route.api_mode ?? "").trim();
    if (!provider && !model && !baseUrl && !apiMode) continue;
    const entry: Record<string, string> = { provider, model };
    if (baseUrl) entry.base_url = baseUrl;
    if (apiMode) entry.api_mode = apiMode;
    out.push(entry);
  }
  return out;
}

/** Append a blank row for the user to fill in. */
export function addFallbackRoute(routes: FallbackRoute[]): FallbackRoute[] {
  return [...routes, { provider: "", model: "" }];
}

/** Patch one row in place (never mutates the input list). */
export function updateFallbackRoute(
  routes: FallbackRoute[],
  index: number,
  patch: Partial<FallbackRoute>,
): FallbackRoute[] {
  if (index < 0 || index >= routes.length) return routes;
  return routes.map((route, i) => (i === index ? { ...route, ...patch } : route));
}

/** Remove one row (never mutates the input list). */
export function removeFallbackRoute(routes: FallbackRoute[], index: number): FallbackRoute[] {
  if (index < 0 || index >= routes.length) return routes;
  return routes.filter((_, i) => i !== index);
}

/** Which neighbour a row should swap with. */
export type FallbackMoveDirection = "up" | "down";

/**
 * Move one row one position up or down by swapping it with its neighbour.
 *
 * The chain is positional — first entry is tried first, last is the last-resort
 * fallback — so reordering is exactly this swap. Returns the input list
 * unchanged when the row or its neighbour would fall outside the list, which
 * lets callers wire it to buttons that are disabled at the ends (never mutates
 * the input).
 */
export function moveFallbackRoute(
  routes: FallbackRoute[],
  index: number,
  direction: FallbackMoveDirection,
): FallbackRoute[] {
  if (index < 0 || index >= routes.length) return routes;
  const target = direction === "up" ? index - 1 : index + 1;
  if (target < 0 || target >= routes.length) return routes;
  const next = [...routes];
  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

/** A route is usable once it names both halves of the provider:model pair. */
export function isRouteComplete(route: FallbackRoute): boolean {
  return route.provider.trim().length > 0 && route.model.trim().length > 0;
}
