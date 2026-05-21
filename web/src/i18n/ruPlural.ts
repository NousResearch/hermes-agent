export function formatRussianCount(
  count: number,
  one: string,
  few: string,
  many: string,
): string {
  const abs = Math.abs(count);
  const mod10 = abs % 10;
  const mod100 = abs % 100;

  let noun = many;
  if (mod10 === 1 && mod100 !== 11) {
    noun = one;
  } else if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    noun = few;
  }

  return `${count} ${noun}`;
}
