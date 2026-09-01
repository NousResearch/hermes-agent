export type ArtifactKind = "code" | "html" | "svg";

export type ArtifactDetection = {
  kind: ArtifactKind;
  language: string;
  title: string;
};

const HTML_LANGUAGES = new Set(["html", "htm", "xhtml"]);
const NON_ARTIFACT_LANGUAGES = new Set(["", "console", "diff", "log", "logs", "markdown", "md", "mermaid", "output", "patch", "plain", "plaintext", "shell-session", "stdout", "text", "txt"]);
const HTML_DOCUMENT_RE = /<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>]/i;
const HTML_TAG_RE = /<[a-z][a-z0-9-]*(\s[^>]*)?>/i;
const FILENAME_COMMENT_RE = /^\s*(?:\/\/|#|--|<!--|\/\*)\s*([\w./-]+\.[a-z0-9]{1,8})\b/i;
const DECLARATION_RE = /(?:^|\n)\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|struct|interface|enum|trait|impl|def|fn)\s+([A-Za-z_$][\w$]*)/;
const EXTENSIONS: Record<string, string> = {
  bash: ".sh", c: ".c", cpp: ".cpp", csharp: ".cs", css: ".css", go: ".go", htm: ".html", html: ".html", java: ".java", javascript: ".js", js: ".js", json: ".json", jsx: ".jsx", kotlin: ".kt", php: ".php", py: ".py", python: ".py", rb: ".rb", rs: ".rs", ruby: ".rb", rust: ".rs", sh: ".sh", sql: ".sql", svg: ".svg", swift: ".swift", toml: ".toml", ts: ".ts", tsx: ".tsx", typescript: ".ts", xhtml: ".html", xml: ".xml", yaml: ".yaml", yml: ".yaml",
};

function cleanLanguage(language: string | undefined): string {
  return (language ?? "").trim().toLowerCase().replace(/[^a-z0-9+#.-]/g, "");
}

function titleFromTag(content: string, tag: "title" | "h1"): string {
  const match = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, "i").exec(content);
  return match?.[1]?.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, 80) ?? "";
}

function codeTitle(language: string, content: string): string {
  const head = content.slice(0, 2000);
  return FILENAME_COMMENT_RE.exec(head)?.[1] ?? DECLARATION_RE.exec(head)?.[1] ?? language ?? "Code";
}

function looksLikeProse(code: string): boolean {
  const words = code.trim().split(/\s+/).filter(Boolean);
  if (words.length < 20) return false;
  const codeSignals = (code.match(/[{};=()[\]<>]|=>|import\s|export\s|const\s|function\s/g) ?? []).length;
  return codeSignals < Math.max(2, Math.floor(words.length / 12));
}

export function detectArtifact(language: string | undefined, code: string | undefined): ArtifactDetection | null {
  const content = (code ?? "").trim();
  if (!content) return null;
  const clean = cleanLanguage(language);

  if (HTML_LANGUAGES.has(clean)) {
    const isDocument = HTML_DOCUMENT_RE.test(content);
    if ((isDocument && content.length >= 160) || (!isDocument && content.length >= 1200 && HTML_TAG_RE.test(content))) {
      return { kind: "html", language: clean, title: titleFromTag(content, "title") || titleFromTag(content, "h1") || "HTML" };
    }
    return null;
  }

  if (clean === "svg") {
    return content.length >= 2000 && /<svg[\s>]/i.test(content)
      ? { kind: "svg", language: clean, title: titleFromTag(content, "title") || "SVG" }
      : null;
  }

  if (NON_ARTIFACT_LANGUAGES.has(clean) || looksLikeProse(content)) return null;
  const lineCount = content.split("\n").length;
  if (content.length < 3000 && lineCount < 48) return null;
  return { kind: "code", language: clean, title: codeTitle(clean, content) };
}

export function artifactDownloadName(kind: ArtifactKind, language: string, title: string): string {
  const base = title.replace(/[^\p{L}\p{N}._ -]+/gu, "").trim().replace(/\s+/g, "-").slice(0, 60) || "artifact";
  if (/\.[a-z0-9]{1,8}$/i.test(base)) return base;
  const extension = kind === "html" ? ".html" : kind === "svg" ? ".svg" : EXTENSIONS[language.toLowerCase()] ?? ".txt";
  return `${base}${extension}`;
}
