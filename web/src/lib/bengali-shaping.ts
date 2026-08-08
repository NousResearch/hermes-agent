export type CharacterJoinRange = [number, number];

interface CharacterJoinerTerminal {
  registerCharacterJoiner(
    handler: (text: string) => CharacterJoinRange[],
  ): number;
  deregisterCharacterJoiner(joinerId: number): void;
}

const BENGALI_BLOCK_RE = /[\u0980-\u09ff]/u;
const BENGALI_JOINABLE_RUN_RE = /[\u0980-\u09ff\u200c\u200d]+/gu;

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

  for (const match of text.matchAll(BENGALI_JOINABLE_RUN_RE)) {
    const value = match[0];
    if (value.length <= 1 || !BENGALI_BLOCK_RE.test(value)) {
      continue;
    }

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
