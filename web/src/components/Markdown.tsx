import { useMemo, useState, type ReactNode } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";

/**
 * Lightweight markdown renderer for LLM output.
 * Handles: code blocks (with a copy button), inline code, bold, italic,
 * headers, links, horizontal rules, lists (incl. nested levels and task
 * list checkboxes), and GFM pipe tables.
 * NOT a full CommonMark parser — optimized for typical assistant message
 * patterns.
 *
 * `streaming` renders a blinking caret at the tail of the last block so it
 * appears to hug the final character instead of wrapping onto a new line
 * after a block element (paragraph/list/code/table/…).
 */
export function Markdown({
  content,
  highlightTerms,
  streaming,
}: {
  content: string;
  highlightTerms?: string[];
  streaming?: boolean;
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
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type TaskState = "open" | "done";

type ListItem = {
  text: string;
  task: TaskState | null;
  orderedChildren: boolean;
  children: ListItem[];
};

type BlockNode =
  | { type: "code"; lang: string; content: string }
  | { type: "heading"; level: number; content: string }
  | { type: "hr" }
  | { type: "list"; ordered: boolean; items: ListItem[] }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "paragraph"; content: string };

/* ------------------------------------------------------------------ */
/*  Block parser                                                       */
/* ------------------------------------------------------------------ */

const LIST_ITEM_RE = /^(\s*)([-*+]|\d+[.)])\s+(.+)$/;
const TABLE_ROW_RE = /^\s*\|.*\|\s*$/;
const TABLE_SEP_RE = /^\s*\|?[\s:|-]*\|[\s:|-]*$/;

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

    // GFM pipe table: a pipe row followed by a `---|---` separator row.
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      TABLE_SEP_RE.test(lines[i + 1]) &&
      lines[i + 1].includes("-")
    ) {
      const header = splitTableRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].includes("|") && TABLE_ROW_RE.test(lines[i])) {
        rows.push(splitTableRow(lines[i]));
        i++;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    // List — contiguous list lines, nesting by indentation.
    if (LIST_ITEM_RE.test(line)) {
      const entries: { indent: number; ordered: boolean; text: string; task: TaskState | null }[] = [];
      while (i < lines.length) {
        const m = lines[i].match(LIST_ITEM_RE);
        if (!m) break;
        const indent = m[1].replace(/\t/g, "  ").length;
        const ordered = /\d+[.)]/.test(m[2]);
        let text = m[3];
        let task: TaskState | null = null;
        const tm = text.match(/^\[([ xX])\]\s+(.*)$/);
        if (tm) {
          task = tm[1] === " " ? "open" : "done";
          text = tm[2];
        }
        entries.push({ indent, ordered, text, task });
        i++;
      }
      blocks.push(parseListTree(entries));
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
      !LIST_ITEM_RE.test(lines[i]) &&
      !lines[i].match(/^[-*_]{3,}\s*$/) &&
      !(lines[i].includes("|") && TABLE_ROW_RE.test(lines[i]))
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

/** Split a `| a | b |` row into trimmed cells (edge pipes dropped). */
function splitTableRow(line: string): string[] {
  const cells = line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|");
  return cells.map((cell) => cell.trim());
}

/** Flatten indented list entries into a nested tree, one list block. */
function parseListTree(
  entries: { indent: number; ordered: boolean; text: string; task: TaskState | null }[],
): BlockNode {
  const baseIndent = entries[0].indent;
  const rootItems: ListItem[] = [];
  const ordered = entries[0].ordered;
  // Stack of (indent, items-array, ordered flag for that level).
  const stack: { indent: number; items: ListItem[]; ordered: boolean }[] = [
    { indent: baseIndent, items: rootItems, ordered },
  ];

  for (const entry of entries) {
    // Deeper than the current level → nest under that level's last item.
    if (entry.indent > stack[stack.length - 1].indent) {
      const parent = stack[stack.length - 1].items[stack[stack.length - 1].items.length - 1];
      if (parent) {
        parent.children = [];
        parent.orderedChildren = entry.ordered;
        stack.push({ indent: entry.indent, items: parent.children, ordered: entry.ordered });
      } else {
        // No parent to nest into — keep this entry at the current level.
        stack[stack.length - 1].items.push({ text: entry.text, task: entry.task, orderedChildren: false, children: [] });
        continue;
      }
    }
    // Shallower than a nested level → unwind back to a matching level.
    while (stack.length > 1 && entry.indent < stack[stack.length - 1].indent) {
      stack.pop();
    }
    stack[stack.length - 1].items.push({
      text: entry.text,
      task: entry.task,
      orderedChildren: false,
      children: [],
    });
  }

  return { type: "list", ordered, items: rootItems };
}

/* ------------------------------------------------------------------ */
/*  Block renderer                                                     */
/* ------------------------------------------------------------------ */

function Block({
  block,
  highlightTerms,
  caret,
}: {
  block: BlockNode;
  highlightTerms?: string[];
  caret?: ReactNode;
}) {
  switch (block.type) {
    case "code":
      return (
        <div>
          <div className="flex justify-end border border-border border-b-0 bg-secondary/60 px-2 py-1">
            <CodeCopyButton code={block.content} />
          </div>
          <pre className="bg-secondary/60 border border-border px-3 py-2.5 text-xs font-mono leading-relaxed overflow-x-auto">
            <code>
              {block.content}
              {caret}
            </code>
          </pre>
        </div>
      );

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

    case "list":
      return (
        <>
          <ListItems items={block.items} ordered={block.ordered} highlightTerms={highlightTerms} />
          {caret}
        </>
      );

    case "table":
      return (
        <>
          <div className="overflow-x-auto">
            <table className="border-collapse border border-border text-sm">
              <thead>
                <tr>
                  {block.header.map((cell, i) => (
                    <th
                      key={i}
                      className="border border-border px-2 py-1 text-left font-semibold"
                    >
                      <InlineContent text={cell} highlightTerms={highlightTerms} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c} className="border border-border px-2 py-1">
                        <InlineContent text={cell} highlightTerms={highlightTerms} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {caret}
        </>
      );

    case "paragraph":
      return (
        <p>
          <InlineContent text={block.content} highlightTerms={highlightTerms} />
          {caret}
        </p>
      );
  }
}

function ListItems({
  items,
  ordered,
  highlightTerms,
  caret,
}: {
  items: ListItem[];
  ordered: boolean;
  highlightTerms?: string[];
  caret?: ReactNode;
}) {
  const Tag = ordered ? "ol" : "ul";
  const last = items.length - 1;
  return (
    <Tag
      className={`space-y-0.5 ${ordered ? "list-decimal" : "list-disc"} pl-5 text-sm`}
    >
      {items.map((item, i) => (
        <li key={i}>
          {item.task && <TaskCheckbox done={item.task === "done"} />}
          <InlineContent text={item.text} highlightTerms={highlightTerms} />
          {item.children.length > 0 && (
            <ListItems
              items={item.children}
              ordered={item.orderedChildren}
              highlightTerms={highlightTerms}
            />
          )}
          {i === last ? caret : null}
        </li>
      ))}
    </Tag>
  );
}

function TaskCheckbox({ done }: { done: boolean }) {
  return (
    <input
      type="checkbox"
      checked={done}
      disabled
      aria-label={done ? "Task completed" : "Task not completed"}
      className="mr-1 align-middle"
    />
  );
}

function CodeCopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label="Copy code block"
      className="text-[0.6875rem] uppercase tracking-[0.08em] text-text-secondary hover:text-midground"
      onClick={() => {
        void copyTextToClipboard(code).then((ok) => {
          if (!ok) return;
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
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
