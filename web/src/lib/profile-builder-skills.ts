type NamedSkill = { name: string };

export function profileSkillSelectionSummary(
  skills: readonly NamedSkill[],
  keptSkills: ReadonlySet<string>,
): { selected: number; total: number } {
  return {
    selected: skills.reduce(
      (count, skill) => count + Number(keptSkills.has(skill.name)),
      0,
    ),
    total: skills.length,
  };
}

export function buildKeepSkillsPayload(
  keepAll: boolean,
  keptSkills: ReadonlySet<string>,
): string[] | undefined {
  return keepAll ? undefined : Array.from(keptSkills);
}
