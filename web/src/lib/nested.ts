// Path segments that would let a crafted config key walk into the prototype
// chain (e.g. "__proto__.polluted") and mutate the global Object.prototype for
// the whole SPA session. Mirrors the hardening already shipped in the desktop
// twin of this helper (apps/desktop/src/app/settings/helpers.ts).
const POLLUTING_PATH_PARTS = new Set(["__proto__", "constructor", "prototype"]);

function isSafePart(part: string): boolean {
  return part.length > 0 && !POLLUTING_PATH_PARTS.has(part);
}

function safePathParts(path: string): string[] {
  const parts = path.split(".");

  if (!parts.every(isSafePart)) {
    throw new Error(`Unsafe config path: ${path}`);
  }

  return parts;
}

// Assign via an own-property definition rather than bracket assignment so a
// "constructor"/"prototype" leaf can never reach the prototype chain even if a
// future caller bypasses safePathParts.
function safeSet(target: Record<string, unknown>, key: string, value: unknown): void {
  if (!isSafePart(key)) {
    throw new Error(`Unsafe config key: ${key}`);
  }

  Object.defineProperty(target, key, {
    value,
    writable: true,
    enumerable: true,
    configurable: true,
  });
}

export function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  let cur: unknown = obj;
  for (const part of safePathParts(path)) {
    if (cur == null || typeof cur !== "object") return undefined;
    if (!Object.prototype.hasOwnProperty.call(cur, part)) return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

export function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const clone = structuredClone(obj);
  const parts = safePathParts(path);
  let cur: Record<string, unknown> = clone;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    const existing = Object.prototype.hasOwnProperty.call(cur, part) ? cur[part] : undefined;
    if (existing == null || typeof existing !== "object") {
      safeSet(cur, part, {});
    }
    cur = cur[part] as Record<string, unknown>;
  }
  safeSet(cur, parts[parts.length - 1], value);
  return clone;
}
