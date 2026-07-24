export const MAX_PREVIEW_BYTES = 2 * 1024 * 1024;

export type PreviewKind = "markdown" | "text" | "image" | "binary" | "large";

const MARKDOWN_EXTENSIONS = new Set([".md", ".markdown", ".mdown", ".mkd"]);
const TEXT_EXTENSIONS = new Set([
  ".css",
  ".csv",
  ".env",
  ".ex",
  ".exs",
  ".go",
  ".h",
  ".html",
  ".ini",
  ".java",
  ".js",
  ".json",
  ".jsx",
  ".log",
  ".mdx",
  ".php",
  ".properties",
  ".py",
  ".rb",
  ".rs",
  ".sh",
  ".sql",
  ".toml",
  ".ts",
  ".tsx",
  ".txt",
  ".xml",
  ".yaml",
  ".yml",
]);

function extension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

export function previewKind(name: string, mimeType: string | null, size: number): PreviewKind {
  if (size > MAX_PREVIEW_BYTES) return "large";
  if (MARKDOWN_EXTENSIONS.has(extension(name))) return "markdown";
  if (mimeType?.toLowerCase().startsWith("image/")) return "image";
  if (mimeType?.toLowerCase().startsWith("text/") || TEXT_EXTENSIONS.has(extension(name))) {
    return "text";
  }
  return "binary";
}

export function decodeTextDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  if (comma < 0 || !dataUrl.slice(0, comma).includes(";base64")) {
    throw new Error("Unsupported file payload");
  }

  const binary = atob(dataUrl.slice(comma + 1));
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export interface Breadcrumb {
  label: string;
  path: string;
}

export function buildBreadcrumbs(root: string, currentPath: string): Breadcrumb[] {
  const separator = root.includes("\\") && !root.includes("/") ? "\\" : "/";
  const trimEnd = (path: string) => {
    if (path === separator) return path;
    return path.replace(/[\\/]+$/, "");
  };
  const normalizedRoot = trimEnd(root);
  const normalizedCurrent = trimEnd(currentPath);
  const rootLabel = normalizedRoot.split(/[\\/]/).filter(Boolean).at(-1) ?? normalizedRoot;
  const breadcrumbs: Breadcrumb[] = [{ label: rootLabel, path: normalizedRoot }];

  if (normalizedCurrent === normalizedRoot) return breadcrumbs;

  const prefix = `${normalizedRoot}${separator}`;
  if (!normalizedCurrent.startsWith(prefix)) {
    return [{ label: normalizedCurrent, path: normalizedCurrent }];
  }

  let path = normalizedRoot;
  for (const part of normalizedCurrent.slice(prefix.length).split(separator).filter(Boolean)) {
    path = `${path}${separator}${part}`;
    breadcrumbs.push({ label: part, path });
  }
  return breadcrumbs;
}

export function pathName(path: string): string {
  return path.replace(/[\\/]+$/, "").split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}
