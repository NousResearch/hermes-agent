import { useState, type ReactNode } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";

type CodeBlockProps = {
  code: string;
  language?: string;
};

type SyntaxTokenKind =
  | "plain"
  | "comment"
  | "string"
  | "number"
  | "keyword"
  | "property"
  | "function"
  | "value"
  | "tag"
  | "operator";

type SyntaxToken = {
  kind: SyntaxTokenKind;
  value: string;
};

const LANGUAGE_ALIASES: Record<string, string> = {
  bash: "shell",
  css: "css",
  html: "html",
  javascript: "javascript",
  js: "javascript",
  json: "json",
  jsx: "javascript",
  less: "css",
  markdown: "markdown",
  md: "markdown",
  plain: "plain",
  plaintext: "plain",
  py: "python",
  python: "python",
  scss: "css",
  shell: "shell",
  sh: "shell",
  text: "plain",
  ts: "javascript",
  tsx: "javascript",
  typescript: "javascript",
  xml: "html",
  yaml: "yaml",
  yml: "yaml",
  zsh: "shell",
};

const COMMON_KEYWORDS = new Set([
  "as", "async", "await", "break", "case", "catch", "class", "const", "continue", "default",
  "delete", "do", "else", "export", "extends", "finally", "for", "from", "function", "if",
  "import", "in", "interface", "let", "new", "of", "return", "switch", "throw", "try", "type",
  "var", "while", "with", "yield",
]);

const LANGUAGE_KEYWORDS: Record<string, Set<string>> = {
  javascript: COMMON_KEYWORDS,
  json: new Set(["false", "null", "true"]),
  python: new Set([
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "False",
    "None", "True",
  ]),
  shell: new Set(["case", "do", "done", "elif", "else", "esac", "fi", "for", "function", "if", "in", "then", "until", "while"]),
  yaml: new Set(["false", "null", "true"]),
};

const CSS_VALUES = new Set([
  "block", "border-box", "currentcolor", "flex", "grid", "important", "inherit", "initial", "inline",
  "none", "relative", "solid", "srgb", "transparent", "unset",
]);

function normalizedLanguage(language?: string): string | null {
  const value = language?.trim().toLowerCase();
  if (!value) return null;
  return LANGUAGE_ALIASES[value] ?? null;
}

function pushToken(tokens: SyntaxToken[], kind: SyntaxTokenKind, value: string): void {
  if (!value) return;
  const previous = tokens.at(-1);
  if (kind === "plain" && previous?.kind === "plain") {
    previous.value += value;
  } else {
    tokens.push({ kind, value });
  }
}

function consumeQuoted(code: string, start: number, quote: string): number {
  const triple = code.startsWith(quote.repeat(3), start);
  let index = start + (triple ? 3 : 1);
  while (index < code.length) {
    if (code[index] === "\\") {
      index += 2;
      continue;
    }
    if (triple ? code.startsWith(quote.repeat(3), index) : code[index] === quote) {
      return index + (triple ? 3 : 1);
    }
    index += 1;
  }
  return code.length;
}

function consumeLineComment(code: string, start: number): number {
  const newline = code.indexOf("\n", start);
  return newline < 0 ? code.length : newline;
}

function nextNonWhitespace(code: string, start: number): number {
  let index = start;
  while (index < code.length && /\s/.test(code[index])) index += 1;
  return index;
}

function consumeIdentifier(code: string, start: number): number {
  let index = start + 1;
  while (index < code.length && /[A-Za-z0-9_$-]/.test(code[index])) index += 1;
  return index;
}

function classifyWord(word: string, language: string, code: string, end: number): SyntaxTokenKind {
  const lower = word.toLowerCase();
  const next = nextNonWhitespace(code, end);
  if (lower === "important") return "keyword";
  if (code[next] === "(") return "function";
  if (code[next] === ":" && (language === "css" || language === "json" || language === "javascript" || language === "python" || language === "yaml")) return "property";
  if (LANGUAGE_KEYWORDS[language]?.has(word) || LANGUAGE_KEYWORDS[language]?.has(lower) || COMMON_KEYWORDS.has(lower)) return "keyword";
  if (language === "css" && CSS_VALUES.has(lower)) return "value";
  return "plain";
}

function tokenizeCode(code: string, language?: string): SyntaxToken[] {
  const normalized = normalizedLanguage(language);
  if (!normalized || normalized === "plain" || normalized === "markdown") return [{ kind: "plain", value: code }];

  const tokens: SyntaxToken[] = [];
  const lineComments = normalized === "javascript" || normalized === "python" || normalized === "shell" || normalized === "yaml";
  let index = 0;
  while (index < code.length) {
    if (normalized === "html" && code[index] === "<" && /[!/A-Za-z]/.test(code[index + 1] ?? "")) {
      const end = code.indexOf(">", index + 1);
      const next = end < 0 ? code.length : end + 1;
      pushToken(tokens, "tag", code.slice(index, next));
      index = next;
      continue;
    }
    if (code.startsWith("/*", index)) {
      const end = code.indexOf("*/", index + 2);
      const next = end < 0 ? code.length : end + 2;
      pushToken(tokens, "comment", code.slice(index, next));
      index = next;
      continue;
    }
    if (lineComments && code.startsWith("//", index)) {
      const next = consumeLineComment(code, index);
      pushToken(tokens, "comment", code.slice(index, next));
      index = next;
      continue;
    }
    if ((normalized === "python" || normalized === "shell" || normalized === "yaml") && code[index] === "#") {
      const next = consumeLineComment(code, index);
      pushToken(tokens, "comment", code.slice(index, next));
      index = next;
      continue;
    }
    if (/["'`]/.test(code[index]) && (normalized !== "python" || code[index] !== "`")) {
      const next = consumeQuoted(code, index, code[index]);
      pushToken(tokens, "string", code.slice(index, next));
      index = next;
      continue;
    }
    if (normalized === "css" && code[index] === "#" && /[0-9a-f]/i.test(code[index + 1] ?? "")) {
      const match = code.slice(index).match(/^#[0-9a-f]{3,8}/i);
      if (match) {
        pushToken(tokens, "number", match[0]);
        index += match[0].length;
        continue;
      }
    }
    if (/\d/.test(code[index]) || (code[index] === "." && /\d/.test(code[index + 1] ?? ""))) {
      const match = code.slice(index).match(/^(?:\d+(?:\.\d+)?|\.\d+)(?:%|[a-zA-Z]+)?/);
      if (match) {
        pushToken(tokens, "number", match[0]);
        index += match[0].length;
        continue;
      }
    }
    if (/[A-Za-z_$]/.test(code[index])) {
      const end = consumeIdentifier(code, index);
      const word = code.slice(index, end);
      pushToken(tokens, classifyWord(word, normalized, code, end), word);
      index = end;
      continue;
    }
    const operator = code.slice(index).match(/^(?:=>|===|!==|==|!=|<=|>=|&&|\|\||\+\+|--|[{}[\]();,:.=+\-*/%!<>?])/);
    if (operator) {
      pushToken(tokens, "operator", operator[0]);
      index += operator[0].length;
      continue;
    }
    pushToken(tokens, "plain", code[index]);
    index += 1;
  }
  return tokens;
}

function highlightedCode(code: string, language?: string): ReactNode {
  return tokenizeCode(code, language).map((token, index) => token.kind === "plain"
    ? token.value
    : <span key={`${token.kind}-${index}`} data-syntax-token={token.kind} className={`code-token-${token.kind}`}>{token.value}</span>);
}

export function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    try {
      if (await copyTextToClipboard(code, true)) {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      } else {
        setCopied(false);
      }
    } catch {
      setCopied(false);
    }
  }

  return (
    <div data-slot="code-block" className="overflow-hidden rounded-md border border-midground/40 bg-secondary/60 text-secondary-foreground text-midground">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 text-xs text-secondary-foreground text-midground">
        <span data-code-language>{language || "Code"}</span>
        <button
          type="button"
          aria-label="Copy code"
          className="rounded px-2 py-1 hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring"
          onClick={copyCode}
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="max-w-full overflow-x-auto p-3 text-xs leading-relaxed text-secondary-foreground text-midground whitespace-pre-wrap break-words sm:whitespace-pre sm:break-normal">
        <code data-slot="code-content" className="bg-transparent text-midground">{highlightedCode(code, language)}</code>
      </pre>
    </div>
  );
}
