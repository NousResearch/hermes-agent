import type { ManagedFileEntry } from "./api";

export function joinPath(base: string, name: string): string {
  const cleanName = name.trim().replace(/^[\\/]+/, "");
  if (!cleanName) return base;
  const separator = base.includes("\\") && !base.includes("/") ? "\\" : "/";
  if (!base || base.endsWith("/") || base.endsWith("\\")) return `${base}${cleanName}`;
  return `${base}${separator}${cleanName}`;
}

export function formatBytes(size: number | null): string {
  if (size === null) return "-";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export function displayPath(path: string | null | undefined): string {
  return path?.trim() || "Files";
}

export function extractManagedFileErrorPath(message: string): string | null {
  const text = message.trim();
  const quotedTail = text.match(/['"]([^'"]+)['"]\s*}?\s*$/);
  if (quotedTail?.[1]?.startsWith("/")) return quotedTail[1];

  const absolutePath = text.match(/((?:\/[\w .:@%+=,~#-]+)+)/);
  return absolutePath?.[1] ?? null;
}

export function parentPathOf(path: string | null | undefined): string | null {
  const value = path?.trim();
  if (!value || value === "/") return null;
  const separator = value.includes("\\") && !value.includes("/") ? "\\" : "/";
  const trimmed = value.replace(/[\\/]+$/, "");
  const index = trimmed.lastIndexOf(separator);
  if (index <= 0) return separator === "/" ? "/" : null;
  return trimmed.slice(0, index);
}

export function filterManagedEntries(
  entries: ManagedFileEntry[],
  query: string,
): ManagedFileEntry[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return entries;
  return entries.filter((entry) =>
    `${entry.name}\n${entry.path}`.toLowerCase().includes(needle),
  );
}
