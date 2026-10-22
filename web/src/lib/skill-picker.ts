/** Filter/sort helpers for the dashboard chat skill picker. */

export interface SkillPickerEntry {
  name: string;
  description?: string;
  usage?: number;
}

function usageCount(usage: unknown): number {
  return typeof usage === "number" && Number.isFinite(usage) ? usage : 0;
}

/** Hyphenate only when the API name still contains spaces. */
function commandToken(name: string): string {
  return /\s/.test(name) ? name.replace(/\s+/g, "-") : name;
}

export function filterSkills<T extends SkillPickerEntry>(
  skills: readonly T[],
  query: string,
): T[] {
  const q = query.trim().toLowerCase();
  const visible = skills.filter((skill) => {
    if (!(skill.name ?? "").trim()) return false;
    if (!q) return true;
    const name = skill.name.toLowerCase();
    const description = (skill.description ?? "").toLowerCase();
    return name.includes(q) || description.includes(q);
  });

  return visible.sort((a, b) => {
    const byUsage = usageCount(b.usage) - usageCount(a.usage);
    if (byUsage !== 0) return byUsage;
    return a.name.localeCompare(b.name);
  });
}

export function launchSkillCommand(name: string | null | undefined): string | null {
  const trimmed = (name ?? "").trim();
  if (!trimmed) return null;
  return `/${commandToken(trimmed)}`;
}
