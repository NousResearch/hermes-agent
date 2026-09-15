const UNSAFE_PATH_SEGMENTS = new Set([
  "__proto__",
  "constructor",
  "prototype",
]);

function pathSegments(path: string): string[] {
  const segments = path.split(".");
  if (
    segments.some(
      (segment) => !segment || UNSAFE_PATH_SEGMENTS.has(segment),
    )
  ) {
    throw new Error("Unsafe nested configuration path");
  }
  return segments;
}

export function getNestedValue(
  obj: Record<string, unknown>,
  path: string,
): unknown {
  let parts: string[];
  try {
    parts = pathSegments(path);
  } catch {
    return undefined;
  }
  let cur: unknown = obj;
  for (const part of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    if (!Object.prototype.hasOwnProperty.call(cur, part)) return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

export function setNestedValue(
  obj: Record<string, unknown>,
  path: string,
  value: unknown,
): Record<string, unknown> {
  const clone = structuredClone(obj);
  const parts = pathSegments(path);
  let cur: Record<string, unknown> = clone;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (
      !Object.prototype.hasOwnProperty.call(cur, part) ||
      cur[part] == null ||
      typeof cur[part] !== "object"
    ) {
      cur[part] = {};
    }
    cur = cur[part] as Record<string, unknown>;
  }
  cur[parts[parts.length - 1]] = value;
  return clone;
}
