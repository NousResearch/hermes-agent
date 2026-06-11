import { useMemo, Fragment, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Full GFM-compatible markdown renderer for LLM output.
 * Uses react-markdown + remark-gfm for proper table, blockquote, strikethrough,
 * and task list support. Tailwind Typography provides the prose styling.
 *
 * `streaming` renders a blinking caret at the tail of the content.
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
  // Strip tool-call summary lines that the agent injects inline
  // e.g. "> 2 skill_view", "> 1 terminal" — these are rendered by
  // ToolGroup components, not inside the markdown body.
  const cleaned = useMemo(() => {
    if (!content) return "";
    return content
      .split("\n")
      .filter((line) => !/^>\s*\d+\s+\w/.test(line))
      .join("\n")
      .trim();
  }, [content]);

  if (!cleaned && !streaming) return null;

  const caret = streaming ? <StreamingCaret /> : null;

  return (
    <div className="prose prose-invert prose-sm max-w-none
      prose-headings:text-foreground prose-headings:font-semibold
      prose-p:text-foreground/90 prose-p:leading-relaxed
      prose-a:text-primary prose-a:no-underline hover:prose-a:underline
      prose-strong:text-foreground prose-strong:font-semibold
      prose-code:text-primary/90 prose-code:bg-secondary/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:before:content-none prose-code:after:content-none prose-code:font-normal
      prose-pre:bg-secondary/60 prose-pre:border prose-pre:border-border prose-pre:rounded-xl prose-pre:text-xs
      prose-blockquote:border-l-primary/50 prose-blockquote:text-foreground/80
      prose-th:text-foreground prose-th:font-semibold prose-th:border-border
      prose-td:border-border prose-td:text-foreground/90
      prose-img:rounded-xl
      prose-hr:border-border
      prose-li:text-foreground/90 prose-li:marker:text-muted-foreground
      text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Highlight search terms in text nodes
          p: ({ children }) => (
            <p>
              {highlightTerms ? highlightChildren(children, highlightTerms) : children}
            </p>
          ),
          li: ({ children }) => (
            <li>
              {highlightTerms ? highlightChildren(children, highlightTerms) : children}
            </li>
          ),
          // Open links in new tab
          a: ({ href, children, ...props }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              {...props}
            >
              {children}
            </a>
          ),
          // Table wrapper for overflow
          table: ({ children }) => (
            <div className="overflow-x-auto rounded-lg border border-border/30">
              <table className="min-w-full">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-muted/50 text-xs uppercase tracking-wide">
              {children}
            </thead>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-medium text-foreground">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-1.5 border-t border-border/20 text-sm">
              {children}
            </td>
          ),
        }}
      >
        {cleaned}
      </ReactMarkdown>
      {caret}
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

/**
 * Walk React children and wrap text nodes that match search terms with <mark>.
 */
function highlightChildren(children: ReactNode, terms: string[]): ReactNode {
  if (!terms || terms.length === 0) return children;

  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`(${escaped.join("|")})`, "gi");

  const processNode = (node: ReactNode): ReactNode => {
    if (typeof node === "string") {
      const parts = node.split(regex);
      if (parts.length === 1) return node;
      return parts.map((part, i) =>
        regex.test(part) ? (
          <mark key={i} className="bg-warning/30 text-warning px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      );
    }
    return node;
  };

  if (Array.isArray(children)) {
    return children.map((child, i) => (
      <Fragment key={i}>{processNode(child)}</Fragment>
    ));
  }
  return processNode(children);
}
