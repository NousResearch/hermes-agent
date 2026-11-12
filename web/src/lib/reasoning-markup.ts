export type ReasoningMarkupSegment =
  | { type: "prose"; content: string }
  | { type: "action"; content: string }
  | { type: "result"; content: string };

type KnownTag = "thinking" | "reflection" | "action" | "result";

const KNOWN_TAGS = new Set<KnownTag>([
  "thinking",
  "reflection",
  "action",
  "result",
]);
const FENCE_RE = /^```/;
const TAG_RE = /<\/?(thinking|reflection|action|result)>/gi;

export function hasReasoningMarkup(content: string): boolean {
  return splitFenceAware(content).some((chunk) => {
    if (chunk.kind === "code") return false;
    return findFirstCompleteKnownTag(chunk.content) !== null;
  });
}

/**
 * Structured reasoning UI is for assistant output only. User prompts and tool
 * results may contain literal known tags that should stay on the Markdown path.
 */
export function shouldRenderStructuredReasoning(
  role: string,
  content: string,
): boolean {
  return role === "assistant" && hasReasoningMarkup(content);
}

export function parseReasoningMarkup(content: string): ReasoningMarkupSegment[] {
  return mergeAdjacentProse(
    splitFenceAware(content).flatMap((chunk) =>
      chunk.kind === "code"
        ? [{ type: "prose" as const, content: chunk.content }]
        : parseKnownTags(chunk.content),
    ),
  );
}

function parseKnownTags(content: string): ReasoningMarkupSegment[] {
  const segments: ReasoningMarkupSegment[] = [];
  let outputCursor = 0;
  let searchCursor = 0;

  for (;;) {
    const opening = findNextOpeningTag(content, searchCursor);
    if (!opening) break;

    const closing = findMatchingCloseTag(content, opening.tag, opening.end);
    if (!closing) {
      searchCursor = opening.end;
      continue;
    }

    if (opening.start > outputCursor) {
      segments.push({
        type: "prose",
        content: content.slice(outputCursor, opening.start),
      });
    }

    const inner = content.slice(opening.end, closing.start);
    if (opening.tag === "action" || opening.tag === "result") {
      segments.push({ type: opening.tag, content: inner.trim() });
    } else {
      segments.push(...parseKnownTags(inner));
    }
    outputCursor = closing.end;
    searchCursor = closing.end;
  }

  if (outputCursor < content.length) {
    segments.push({ type: "prose", content: content.slice(outputCursor) });
  }

  return segments.filter((segment) => segment.content.length > 0);
}

function findFirstCompleteKnownTag(content: string): KnownTag | null {
  let cursor = 0;
  for (;;) {
    const opening = findNextOpeningTag(content, cursor);
    if (!opening) return null;
    if (findMatchingCloseTag(content, opening.tag, opening.end)) {
      return opening.tag;
    }
    cursor = opening.end;
  }
}

function findNextOpeningTag(
  content: string,
  from: number,
): { tag: KnownTag; start: number; end: number } | null {
  TAG_RE.lastIndex = from;
  let match: RegExpExecArray | null;
  while ((match = TAG_RE.exec(content)) !== null) {
    const raw = match[0];
    const tag = match[1].toLowerCase() as KnownTag;
    if (!raw.startsWith("</") && KNOWN_TAGS.has(tag)) {
      return { tag, start: match.index, end: TAG_RE.lastIndex };
    }
  }
  return null;
}

function findMatchingCloseTag(
  content: string,
  tag: KnownTag,
  from: number,
): { start: number; end: number } | null {
  const tagRe = new RegExp(`</?${tag}>`, "gi");
  tagRe.lastIndex = from;
  let depth = 1;
  let match: RegExpExecArray | null;

  while ((match = tagRe.exec(content)) !== null) {
    if (match[0].startsWith("</")) {
      depth -= 1;
      if (depth === 0) {
        return { start: match.index, end: tagRe.lastIndex };
      }
    } else {
      depth += 1;
    }
  }

  return null;
}

function splitFenceAware(
  content: string,
): Array<{ kind: "prose" | "code"; content: string }> {
  const chunks: Array<{ kind: "prose" | "code"; content: string }> = [];
  const lines = content.split("\n");
  let buffer: string[] = [];
  let inCode = false;

  for (const line of lines) {
    if (FENCE_RE.test(line)) {
      if (!inCode && buffer.length > 0) {
        chunks.push({ kind: "prose", content: buffer.join("\n") });
        buffer = [];
      }
      buffer.push(line);

      if (inCode) {
        chunks.push({ kind: "code", content: buffer.join("\n") });
        buffer = [];
      }
      inCode = !inCode;
      continue;
    }

    buffer.push(line);
  }

  if (buffer.length > 0) {
    chunks.push({
      kind: inCode ? "code" : "prose",
      content: buffer.join("\n"),
    });
  }

  return chunks;
}

function mergeAdjacentProse(
  segments: ReasoningMarkupSegment[],
): ReasoningMarkupSegment[] {
  const merged: ReasoningMarkupSegment[] = [];

  for (const segment of segments) {
    const prev = merged.at(-1);
    if (prev?.type === "prose" && segment.type === "prose") {
      prev.content += `\n${segment.content}`;
    } else {
      merged.push({ ...segment });
    }
  }

  return merged;
}
