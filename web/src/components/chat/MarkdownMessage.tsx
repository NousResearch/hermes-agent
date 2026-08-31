import { useMemo, type ElementType, type ReactNode } from "react";

import { CodeBlock } from "./CodeBlock";

type Block =
  | { type: "code"; code: string; language?: string }
  | { type: "heading"; level: number; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "quote"; text: string }
  | { type: "paragraph"; text: string };

export type MarkdownMessageProps = {
  content: string;
  className?: string;
};

/** Safe, intentionally small Markdown renderer for assistant messages.
 * HTML is never interpreted; only a conservative set of elements is emitted.
 */
export function MarkdownMessage({ content, className = "" }: MarkdownMessageProps) {
  const blocks = useMemo(() => parseBlocks(content), [content]);
  return (
    <div className={`space-y-3 text-sm leading-relaxed text-foreground ${className}`.trim()}>
      {blocks.map((block, index) => <BlockView key={index} block={block} />)}
    </div>
  );
}

function BlockView({ block }: { block: Block }) {
  if (block.type === "code") return <CodeBlock code={block.code} language={block.language} />;
  if (block.type === "heading") {
    const Tag = `h${block.level}` as ElementType;
    return <Tag className="font-semibold">{inline(block.text)}</Tag>;
  }
  if (block.type === "quote") {
    return <blockquote className="border-l-2 border-primary/50 pl-3 text-muted-foreground">{inline(block.text)}</blockquote>;
  }
  if (block.type === "list") {
    const Tag = block.ordered ? "ol" : "ul";
    return <Tag className={`space-y-1 pl-5 ${block.ordered ? "list-decimal" : "list-disc"}`}>
      {block.items.map((item, index) => <li key={index}>{inline(item)}</li>)}
    </Tag>;
  }
  return <p className="whitespace-pre-wrap break-words">{inline(block.text)}</p>;
}

function parseBlocks(content: string): Block[] {
  const normalized = content.replaceAll("\r\n", "\n");
  // A message without block syntax is kept as one pre-wrapped paragraph so
  // leading, trailing, and repeated newlines are not discarded.
  if (!/(^|\n)\s*(?:`{3,}|~{3,}|#{1,6}\s|>\s?|[-+*]\s+|\d+[.)]\s+)/m.test(normalized)) {
    return normalized ? [{ type: "paragraph", text: normalized }] : [];
  }
  const lines = normalized.split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    if (!lines[i].trim()) { i++; continue; }
    const fence = lines[i].match(/^\s*(`{3,}|~{3,})\s*([^\s`]*)\s*$/);
    if (fence) {
      const marker = fence[1][0];
      const close = new RegExp(`^\\s*${marker}{${fence[1].length},}\\s*$`);
      const code: string[] = [];
      i++;
      while (i < lines.length && !close.test(lines[i])) code.push(lines[i++]);
      if (i < lines.length) i++;
      blocks.push({ type: "code", code: code.join("\n"), language: fence[2] || undefined });
      continue;
    }
    const heading = lines[i].match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) { blocks.push({ type: "heading", level: heading[1].length, text: heading[2] }); i++; continue; }
    const quote: string[] = [];
    while (i < lines.length && /^\s*> ?/.test(lines[i])) quote.push(lines[i++].replace(/^\s*> ?/, ""));
    if (quote.length) { blocks.push({ type: "quote", text: quote.join("\n") }); continue; }
    const list = lines[i].match(/^\s*([-+*])\s+(.+)$/) || lines[i].match(/^\s*(\d+)[.)]\s+(.+)$/);
    if (list) {
      const ordered = /^\d/.test(list[1]);
      const items: string[] = [];
      while (i < lines.length) {
        const item = lines[i].match(ordered ? /^\s*\d+[.)]\s+(.+)$/ : /^\s*[-+*]\s+(.+)$/);
        if (!item) break;
        items.push(item[1]); i++;
      }
      blocks.push({ type: "list", ordered, items }); continue;
    }
    const paragraph: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(`{3,}|~{3,})/.test(lines[i]) &&
      !/^\s*#{1,6}\s+/.test(lines[i]) && !/^\s*>/.test(lines[i]) &&
      !/^\s*(?:[-+*]\s+|\d+[.)]\s+)/.test(lines[i])) paragraph.push(lines[i++]);
    if (paragraph.length) blocks.push({ type: "paragraph", text: paragraph.join("\n") });
  }
  return blocks;
}

function safeHref(value: string): string | undefined {
  const href = value.trim();
  return /^(?:https?:|mailto:)/i.test(href) ? href : undefined;
}

function inline(text: string): ReactNode {
  const pattern = /(\n)|(`[^`\n]+`)|(\[([^\]]+)\]\(([^)\s]+(?:\s[^)]*)?)\))|(\*\*([^*]+)\*\*|__([^_]+)__)|(\*([^*\n]+)\*|_([^_\n]+)_)/g;
  const output: ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text))) {
    if (match.index > last) output.push(text.slice(last, match.index));
    if (match[1]) output.push("\n", <br key={match.index} />);
    else if (match[2]) output.push(<code key={match.index} className="rounded bg-secondary px-1 font-mono text-xs">{match[2].slice(1, -1)}</code>);
    else if (match[3]) {
      const href = safeHref(match[5]);
      output.push(href ? <a key={match.index} href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">{match[4]}</a> : match[4]);
    } else if (match[6]) output.push(<strong key={match.index}>{match[7] || match[8]}</strong>);
    else if (match[9]) output.push(<em key={match.index}>{match[10] || match[11]}</em>);
    last = match.index + match[0].length;
  }
  if (last < text.length) output.push(text.slice(last));
  return output;
}
