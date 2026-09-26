import type { PluginManifest } from "./types";
import { SDK_CONTRACT_VERSION } from "./registry";

const component = "(?:0|[1-9][0-9]*)";
const version = `${component}\\.${component}(?:\\.${component})?`;
const exactVersion = new RegExp(`^(?:${version})$`);
const maxVersion = new RegExp(`^(?:${version}|${component}\\.x)$`);

function parts(value: string): bigint[] {
  const values = value.split(".").map(BigInt);
  return [...values, ...Array<bigint>(3 - values.length).fill(0n)];
}

function compare(a: bigint[], b: bigint[]): number {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

const alphabet = "[A-Za-z0-9+/_-]";
const digest = `(?:sha256-${alphabet}{43}=?|sha384-${alphabet}{64}|sha512-${alphabet}{86}(?:==)?)`;
const token = `${digest}(?:\\?[\\x21-\\x7e]*)?`;
const whitespace = "[ \\t\\n\\f\\r]";
// Match the entire value, including a terminal newline (JS `$` alone does not).
const jsIntegrity = new RegExp(`^${whitespace}*${token}(?:${whitespace}+${token})*${whitespace}*(?![\\s\\S])`);

/** Validate supplied declarations, never treating malformed metadata as absent. */
export function pluginAdmissionError(manifest: PluginManifest): string | null {
  if ("integrity" in manifest
    && (typeof manifest.integrity !== "string" || !jsIntegrity.test(manifest.integrity))) {
    return "Invalid JavaScript integrity declaration";
  }
  if ("css_integrity" in manifest
    && (typeof manifest.css_integrity !== "string"
      || !/^sha384-[A-Za-z0-9+/]{64}(?![\s\S])/.test(manifest.css_integrity))) {
    return "Invalid CSS integrity declaration";
  }
  if ("sdk" in manifest) {
    const sdk = manifest.sdk;
    if (!sdk || typeof sdk !== "object" || Array.isArray(sdk)
      || Object.keys(sdk).sort().join(",") !== "max,min"
      || typeof sdk.min !== "string" || !exactVersion.test(sdk.min)
      || typeof sdk.max !== "string" || !maxVersion.test(sdk.max)
      || /\s/.test(sdk.min + sdk.max)) return "Invalid SDK declaration";
    const lower = parts(sdk.min);
    const host = parts(SDK_CONTRACT_VERSION);
    const wildcard = sdk.max.endsWith(".x");
    const upper = wildcard ? [BigInt(sdk.max.split(".")[0])] : parts(sdk.max);
    if ((wildcard ? lower[0] > upper[0] : compare(lower, upper) > 0)) {
      return "Invalid SDK range";
    }
    if (compare(host, lower) < 0
      || (wildcard ? host[0] > upper[0] : compare(host, upper) > 0)) {
      return `Incompatible SDK (host ${SDK_CONTRACT_VERSION})`;
    }
  }
  return null;
}
