// Lightweight fuzzy subsequence scorer for picker filtering.
//
// Matches a query as an ordered subsequence of the target (so `g4o` matches
// `gpt-4o`) and scores by match quality so callers can rank results. Higher
// score is a better match. Returns the matched character indices so callers
// can highlight them.
//
// The scoring favours, in rough order: exact full match, prefix match, matches
// that start on a word boundary (after `-`, `_`, `/`, `.`, space, or a
// lower→upper case transition), contiguous runs, and earlier matches. This is
// intentionally simple — no external dependency — but good enough to make
// `son4` rank `claude-sonnet-4` above an incidental scattered hit.
//
// This is a logically identical copy of ui-tui/src/lib/fuzzy.ts (only prettier
// formatting differs); keep the two in sync. The TUI copy carries the vitest
// suite (this `web` package has no test runner), so behavioural changes should
// be validated there.

export interface FuzzyMatch {
  /** Total score; higher is better. */
  score: number;
  /** Indices into the original (non-lowercased) target that were matched. */
  positions: number[];
}

const WORD_BOUNDARY = /[-_/.\s]/;

function isBoundary(target: string, index: number): boolean {
  if (index === 0) {
    return true;
  }

  const prev = target[index - 1];

  if (WORD_BOUNDARY.test(prev)) {
    return true;
  }

  // camelCase / lower→upper transition (e.g. the `O` in `gptO`).
  const cur = target[index];

  return (
    prev === prev.toLowerCase() &&
    cur !== cur.toLowerCase() &&
    cur === cur.toUpperCase()
  );
}

/**
 * Score a single query token against a target. Returns null when the token is
 * not a subsequence of the target. An empty query scores 0 with no positions.
 */
export function fuzzyScore(target: string, query: string): FuzzyMatch | null {
  if (!query) {
    return { score: 0, positions: [] };
  }

  const lowerTarget = target.toLowerCase();
  const lowerQuery = query.toLowerCase();

  const positions: number[] = [];
  let score = 0;
  let prevIndex = -1;
  let searchFrom = 0;

  for (const ch of lowerQuery) {
    const idx = lowerTarget.indexOf(ch, searchFrom);

    if (idx < 0) {
      return null;
    }

    positions.push(idx);

    // Base point for the matched character.
    score += 1;

    // Contiguous with the previous match → strong bonus.
    if (prevIndex >= 0 && idx === prevIndex + 1) {
      score += 5;
    } else if (prevIndex >= 0) {
      // Penalise the gap we had to skip (capped), so contiguous beats scattered.
      score -= Math.min(idx - prevIndex - 1, 3);
    }

    // Word-boundary / start-of-string matches are meaningful.
    if (isBoundary(target, idx)) {
      score += 3;
    }

    // Matching the very first character of the target is the strongest signal.
    if (idx === 0) {
      score += 5;
    }

    prevIndex = idx;
    searchFrom = idx + 1;
  }

  // Prefix bonus: the query matched a contiguous prefix of the target.
  if (
    positions.length &&
    positions[0] === 0 &&
    positions[positions.length - 1] === positions.length - 1
  ) {
    score += 8;
  }

  // Exact full match dominates everything else.
  if (lowerTarget === lowerQuery) {
    score += 20;
  }

  // Slightly prefer shorter targets when scores are otherwise close, so a
  // query that fully prefixes a short id beats the same prefix on a long one.
  score -= lowerTarget.length * 0.01;

  return { score, positions };
}

/**
 * Score a target against a whitespace-separated, multi-token query. Every token
 * must match (AND semantics); the result aggregates per-token scores and the
 * union of matched positions. Returns null if any token fails to match.
 */
export function fuzzyScoreMulti(
  target: string,
  query: string,
): FuzzyMatch | null {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);

  if (!tokens.length) {
    return { score: 0, positions: [] };
  }

  let score = 0;
  const positionSet = new Set<number>();

  for (const token of tokens) {
    const match = fuzzyScore(target, token);

    if (!match) {
      return null;
    }

    score += match.score;

    for (const pos of match.positions) {
      positionSet.add(pos);
    }
  }

  return { score, positions: [...positionSet].sort((a, b) => a - b) };
}

export interface RankedItem<T> {
  item: T;
  score: number;
  positions: number[];
}

/**
 * Filter + rank a list by a fuzzy query against a derived text key. Non-matching
 * items are dropped; matches are sorted by score (descending), ties broken by
 * the original index so ordering is stable for equal scores. An empty query
 * returns every item in original order with no positions.
 */
export function fuzzyRank<T>(
  items: readonly T[],
  query: string,
  toText: (item: T) => string,
): RankedItem<T>[] {
  const trimmed = query.trim();

  if (!trimmed) {
    return items.map((item) => ({ item, score: 0, positions: [] }));
  }

  const ranked: Array<RankedItem<T> & { index: number }> = [];

  items.forEach((item, index) => {
    const match = fuzzyScoreMulti(toText(item), trimmed);

    if (match) {
      ranked.push({ item, score: match.score, positions: match.positions, index });
    }
  });

  ranked.sort((a, b) => b.score - a.score || a.index - b.index);

  return ranked.map(({ item, score, positions }) => ({ item, score, positions }));
}
