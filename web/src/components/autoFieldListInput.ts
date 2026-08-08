export function parseListInput(raw: string): string[] {
  // Keep empty segments while the field is focused. Removing the trailing
  // empty item immediately would make a just-typed comma disappear on the
  // controlled re-render, preventing entry of the next value.
  return raw.split(",").map((item) => item.trim());
}

export function compactListInput(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function normalizeListFieldValue(value: unknown): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return compactListInput(parseListInput(value));
  return null;
}

function normalizeListAtPath(value: unknown, path: string[]): unknown {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return value;
  const [key, ...rest] = path;
  if (!key || !(key in value)) return value;

  const record = value as Record<string, unknown>;
  if (rest.length === 0) {
    if (typeof record[key] !== "string") return value;
    return { ...record, [key]: normalizeListFieldValue(record[key]) ?? [] };
  }

  const normalizedChild = normalizeListAtPath(record[key], rest);
  return normalizedChild === record[key] ? value : { ...record, [key]: normalizedChild };
}

export function normalizeBuzzAllowedUsersConfig(value: unknown): unknown {
  return normalizeListAtPath(value, [
    "gateway",
    "platforms",
    "buzz",
    "extra",
    "allowed_users",
  ]);
}

const BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

function bech32Polymod(values: number[]): number {
  const generators = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
  let checksum = 1;
  for (const value of values) {
    const top = checksum >>> 25;
    checksum = ((checksum & 0x1ffffff) << 5) ^ value;
    for (let bit = 0; bit < 5; bit += 1) {
      if ((top >>> bit) & 1) checksum ^= generators[bit];
    }
  }
  return checksum;
}

function isValidNpub(value: string): boolean {
  const normalized = value.toLowerCase();
  if (!normalized.startsWith("npub1")) return false;
  const data = [...normalized.slice(5)].map((char) => BECH32_CHARSET.indexOf(char));
  if (data.length < 7 || data.some((part) => part < 0)) return false;
  const hrp = [..."npub"];
  const expandedHrp = [
    ...hrp.map((char) => char.charCodeAt(0) >>> 5),
    0,
    ...hrp.map((char) => char.charCodeAt(0) & 31),
  ];
  if (bech32Polymod([...expandedHrp, ...data]) !== 1) return false;

  let accumulator = 0;
  let bits = 0;
  const decoded: number[] = [];
  for (const part of data.slice(0, -6)) {
    accumulator = (accumulator << 5) | part;
    bits += 5;
    while (bits >= 8) {
      bits -= 8;
      decoded.push((accumulator >>> bits) & 0xff);
    }
  }
  return decoded.length === 32 && bits < 5 && ((accumulator << (8 - bits)) & 0xff) === 0;
}

export function validateBuzzAllowedUsers(value: unknown): string | null {
  const normalized = normalizeListFieldValue(value);
  if (!normalized) return "Allowed Users must be a list.";
  for (const [index, rawItem] of normalized.entries()) {
    if (typeof rawItem !== "string") return "Each public key must be text.";
    const item = rawItem.trim();
    if (/\s/.test(item)) return "Separate public keys with commas.";
    if (!/^[0-9a-f]{64}$/i.test(item) && !isValidNpub(item)) {
      return `Invalid public key at item ${index + 1}; use one npub or 64-character hex key.`;
    }
  }
  return null;
}

export function getBuzzAllowedUsersValidationError(config: unknown): string | null {
  const paths = [
    ["gateway", "platforms", "buzz", "extra", "allowed_users"],
    ["buzz", "extra", "allowed_users"],
  ];
  for (const path of paths) {
    let current: unknown = config;
    let found = true;
    for (const key of path) {
      if (typeof current !== "object" || current === null || !(key in current)) {
        found = false;
        break;
      }
      current = (current as Record<string, unknown>)[key];
    }
    if (found) {
      return validateBuzzAllowedUsers(current);
    }
  }
  return null;
}

export function updateListInputDraft(raw: string): { draft: string; value: string[] } {
  return {
    draft: raw,
    value: compactListInput(parseListInput(raw)),
  };
}
