export type CharacterJoinRange = [number, number];

interface CharacterJoinerTerminal {
  registerCharacterJoiner(
    handler: (text: string) => CharacterJoinRange[],
  ): number;
  deregisterCharacterJoiner(joinerId: number): void;
}

// Join only virama-linked consonant clusters (plus their final vowel mark).
// Joining whole Bengali runs makes xterm cursor movement and selection treat
// an entire word or pasted paragraph as one unit.
const BENGALI_CONJUNCT_RE =
  /[\u0995-\u09b9\u09dc-\u09df]\u09bc?(?:\u09cd[\u200c\u200d]?[\u0995-\u09b9\u09dc-\u09df]\u09bc?)+(?:[\u0981-\u0983\u09be-\u09c4\u09c7\u09c8\u09cb\u09cc\u09d7])*/gu;

export const DASHBOARD_CHAT_TERMINAL_FONT_FAMILY = [
  "'JetBrains Mono'",
  "'Noto Sans Bengali'",
  "'Noto Serif Bengali'",
  "'Nirmala UI'",
  "'Bangla Sangam MN'",
  "'Bangla MN'",
  "'Kohinoor Bangla'",
  "'Vrinda'",
  "'Mukti'",
  "'SolaimanLipi'",
  "'Cascadia Mono'",
  "'Fira Code'",
  "'MesloLGS NF'",
  "'Source Code Pro'",
  "Menlo",
  "Consolas",
  "'DejaVu Sans Mono'",
  "monospace",
].join(", ");

export function getBengaliCharacterJoinRanges(
  text: string,
): CharacterJoinRange[] {
  const ranges: CharacterJoinRange[] = [];

  for (const match of text.matchAll(BENGALI_CONJUNCT_RE)) {
    const value = match[0];
    const start = match.index;
    ranges.push([start, start + value.length]);
  }

  return ranges;
}

export function registerBengaliCharacterJoiner(
  term: CharacterJoinerTerminal,
): () => void {
  const joinerId = term.registerCharacterJoiner(getBengaliCharacterJoinRanges);
  return () => term.deregisterCharacterJoiner(joinerId);
}
