import { useMemo, useState, type ReactNode } from "react";

import { Check as CheckIcon, Copy as CopyIcon } from "lucide-react";

import { copyTextToClipboard } from "@/lib/clipboard";

/**
 * Lightweight markdown renderer for LLM output.
 * Handles: code blocks, inline code, bold, italic, headers, links, lists, horizontal rules.
 * NOT a full CommonMark parser — optimized for typical assistant message patterns.
 *
 * `streaming` renders a blinking caret at the tail of the last block so it
 * appears to hug the final character instead of wrapping onto a new line
 * after a block element (paragraph/list/code/…).
 */
export function Markdown({
  content,
  highlightTerms,
  streaming,
  codeCopy = false,
}: {
  content: string;
  highlightTerms?: string[];
  streaming?: boolean;
  /** Show a copy button on fenced code blocks (mobile chat uses this). */
  codeCopy?: boolean;
}) {
  const blocks = useMemo(() => parseBlocks(content), [content]);
  const caret = streaming ? <StreamingCaret /> : null;

  return (
    <div className="text-sm text-foreground leading-relaxed space-y-2">
      {blocks.map((block, i) => (
        <Block
          key={i}
          block={block}
          highlightTerms={highlightTerms}
          caret={caret && i === blocks.length - 1 ? caret : null}
          codeCopy={codeCopy}
        />
      ))}
      {blocks.length === 0 && caret}
    </div>
  );
}

function StreamingCaret() {
  return (
    <span
      aria-hidden
      className="inline-block w-[0.5em] h-[1em] ml-0.5 align-[-0.15em] bg-foreground/50 animate-pulse"
    />
  );
}

/* ------------------------------------------------------------------ */
/*  Code block + copy button                                           */
/* ------------------------------------------------------------------ */

/**
 * Fenced code block. With `codeCopy` the block gains a header row holding
 * the language label and a copy button; the button lives in that header
 * (not absolutely positioned over the code) so it never interferes with
 * the horizontal scrolling of wide code — the scroll container is the
 * <pre> only, and the header stays fixed above it.
 *
 * Without `codeCopy` (desktop) the rendering is byte-identical to the
 * original bare <pre>.
 */
function CodeBlock({
  block,
  caret,
  codeCopy,
}: {
  block: { type: "code"; lang: string; content: string };
  caret?: ReactNode;
  codeCopy?: boolean;
}) {
  const pre = (
    <pre className="px-3 py-2.5 text-xs font-mono leading-relaxed overflow-x-auto">
      <code>
        {block.content}
        {caret}
      </code>
    </pre>
  );

  if (!codeCopy) {
    return (
      <pre className="bg-secondary/60 border border-border px-3 py-2.5 text-xs font-mono leading-relaxed overflow-x-auto">
        <code>
          {block.content}
          {caret}
        </code>
      </pre>
    );
  }

  return (
    <div className="border border-border bg-secondary/60">
      <div className="flex items-center justify-between gap-2 border-b border-border/60 bg-card/80 pl-3 pr-1.5 py-1">
        <span className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground/70 select-none">
          {block.lang || "code"}
        </span>
        <CopyCodeButton text={block.content} />
      </div>
      {pre}
    </div>
  );
}

function CopyCodeButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    void copyTextToClipboard(text).then((ok) => {
      if (!ok) return;
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : "Copy code"}
      onClick={handleCopy}
      className="flex items-center gap-1 rounded border border-border/60 bg-card/80 px-1.5 py-0.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors shrink-0"
    >
      {copied ? (
        <CheckIcon className="h-3 w-3" />
      ) : (
        <CopyIcon className="h-3 w-3" />
      )}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type BlockNode =
  | { type: "code"; lang: string; content: string }
  | { type: "heading"; level: number; content: string }
  | { type: "hr" }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "paragraph"; content: string };

/* ------------------------------------------------------------------ */
/*  Block parser                                                       */
/* ------------------------------------------------------------------ */

function parseBlocks(text: string): BlockNode[] {
  const lines = text.split("\n");
  const blocks: BlockNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    const fenceMatch = line.match(/^```(\w*)/);
    if (fenceMatch) {
      const lang = fenceMatch[1] || "";
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push({ type: "code", lang, content: codeLines.join("\n") });
      continue;
    }

    // Heading
    const headingMatch = line.match(/^(#{1,4})\s+(.+)/);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        content: headingMatch[2],
      });
      i++;
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }

    // Unordered list
    if (/^[-*+]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*+]\s/, ""));
        i++;
      }
      blocks.push({ type: "list", ordered: false, items });
      continue;
    }

    // Ordered list
    if (/^\d+[.)]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+[.)]\s/, ""));
        i++;
      }
      blocks.push({ type: "list", ordered: true, items });
      continue;
    }

    // Empty line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraph — collect consecutive non-empty, non-special lines
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].match(/^```/) &&
      !lines[i].match(/^#{1,4}\s/) &&
      !lines[i].match(/^[-*+]\s/) &&
      !lines[i].match(/^\d+[.)]\s/) &&
      !lines[i].match(/^[-*_]{3,}\s*$/)
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "paragraph", content: paraLines.join("\n") });
    }
  }

  return blocks;
}

/* ------------------------------------------------------------------ */
/*  Block renderer                                                     */
/* ------------------------------------------------------------------ */

function Block({
  block,
  highlightTerms,
  caret,
  codeCopy,
}: {
  block: BlockNode;
  highlightTerms?: string[];
  caret?: ReactNode;
  codeCopy?: boolean;
}) {
  switch (block.type) {
    case "code":
      return <CodeBlock block={block} caret={caret} codeCopy={codeCopy} />;

    case "heading": {
      const Tag = `h${Math.min(block.level, 4)}` as "h1" | "h2" | "h3" | "h4";
      const sizes: Record<string, string> = {
        h1: "text-base font-bold",
        h2: "text-sm font-bold",
        h3: "text-sm font-semibold",
        h4: "text-sm font-medium",
      };
      return (
        <Tag className={sizes[Tag]}>
          <InlineContent text={block.content} highlightTerms={highlightTerms} />
          {caret}
        </Tag>
      );
    }

    case "hr":
      return (
        <>
          <hr className="border-border" />
          {caret}
        </>
      );

    case "list": {
      const Tag = block.ordered ? "ol" : "ul";
      const last = block.items.length - 1;
      return (
        <Tag
          className={`space-y-0.5 ${block.ordered ? "list-decimal" : "list-disc"} pl-5 text-sm`}
        >
          {block.items.map((item, i) => (
            <li key={i}>
              <InlineContent text={item} highlightTerms={highlightTerms} />
              {i === last ? caret : null}
            </li>
          ))}
        </Tag>
      );
    }

    case "paragraph":
      return (
        <p>
          <InlineContent text={block.content} highlightTerms={highlightTerms} />
          {caret}
        </p>
      );
  }
}

/* ------------------------------------------------------------------ */
/*  Inline parser + renderer                                           */
/* ------------------------------------------------------------------ */

type InlineNode =
  | { type: "text"; content: string }
  | { type: "code"; content: string }
  | { type: "bold"; content: string }
  | { type: "italic"; content: string }
  | { type: "link"; text: string; href: string }
  | { type: "br" };

function parseInline(text: string): InlineNode[] {
  const nodes: InlineNode[] = [];
  // Pattern priority: code > link > bold > italic > bare URL > line break
  const pattern =
    /(`[^`]+`)|(\[([^\]]+)\]\(([^)]+)\))|(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(\bhttps?:\/\/[^\s<>)\]]+)|(\n)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push({ type: "text", content: text.slice(lastIndex, match.index) });
    }

    if (match[1]) {
      // Inline code
      nodes.push({ type: "code", content: match[1].slice(1, -1) });
    } else if (match[2]) {
      // [text](url) link
      nodes.push({ type: "link", text: match[3], href: match[4] });
    } else if (match[5]) {
      // **bold**
      nodes.push({ type: "bold", content: match[6] });
    } else if (match[7]) {
      // *italic*
      nodes.push({ type: "italic", content: match[8] });
    } else if (match[9]) {
      // Bare URL
      nodes.push({ type: "link", text: match[9], href: match[9] });
    } else if (match[10]) {
      // Line break within paragraph
      nodes.push({ type: "br" });
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    nodes.push({ type: "text", content: text.slice(lastIndex) });
  }

  return nodes;
}

function InlineContent({
  text,
  highlightTerms,
}: {
  text: string;
  highlightTerms?: string[];
}) {
  const nodes = useMemo(() => parseInline(text), [text]);

  return (
    <>
      {nodes.map((node, i) => {
        switch (node.type) {
          case "text":
            return (
              <HighlightedText
                key={i}
                text={node.content}
                terms={highlightTerms}
              />
            );
          case "code":
            return (
              <code
                key={i}
                className="bg-secondary/60 px-1.5 py-0.5 text-xs font-mono text-primary/90"
              >
                {node.content}
              </code>
            );
          case "bold":
            return (
              <strong key={i} className="font-semibold">
                <HighlightedText text={node.content} terms={highlightTerms} />
              </strong>
            );
          case "italic":
            return (
              <em key={i}>
                <HighlightedText text={node.content} terms={highlightTerms} />
              </em>
            );
          case "link": {
            // Security: only render http(s)/mailto links. Other schemes
            // (javascript:, data:, vbscript:) are dropped to plain text so a
            // crafted link in agent/message content can't execute on click.
            const href = node.href.trim();
            if (!/^(https?:|mailto:)/i.test(href)) {
              return (
                <HighlightedText
                  key={i}
                  text={node.text}
                  terms={highlightTerms}
                />
              );
            }
            return (
              <a
                key={i}
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-primary underline underline-offset-2 decoration-primary/30 hover:decoration-primary/60 transition-colors"
              >
                {node.text}
              </a>
            );
          }
          case "br":
            return <br key={i} />;
        }
      })}
    </>
  );
}

/** Highlight search terms within a plain text string. */
function HighlightedText({ text, terms }: { text: string; terms?: string[] }) {
  if (!terms || terms.length === 0) return <>{text}</>;

  // Build a regex that matches any of the search terms (case-insensitive)
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark key={i} className="bg-warning/30 text-warning px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}
