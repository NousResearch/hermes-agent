// The existing {s} suffix is for English-like plurals. A locale may instead
// supply the full {count} label (e.g. Russian "Полей: {count}") to avoid
// attaching an English suffix or an incorrectly declined noun after the number.
export function countLabel(template: string, count: number): string {
  const text = template.replace("{count}", String(count)).replace("{s}", count === 1 ? "" : "s");
  return template.includes("{count}") ? text : `${count} ${text}`;
}
