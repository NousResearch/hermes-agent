export interface ProfileModelChoice {
  provider: string;
  model: string;
  label: string;
}

export function profileModelKey(choice: ProfileModelChoice): string {
  return `${choice.provider}\u0000${choice.model}`;
}

export function filterProfileModelChoices(
  choices: readonly ProfileModelChoice[],
  query: string,
): ProfileModelChoice[] {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return [...choices];

  return choices.filter((choice) =>
    [choice.provider, choice.model, choice.label].some((value) =>
      value.toLowerCase().includes(normalizedQuery),
    ),
  );
}
