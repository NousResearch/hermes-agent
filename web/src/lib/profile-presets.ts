export interface ProfileExample {
  name: string;
  title: string;
  description: string;
  bestFor: string;
}

export const PROFILE_FIELD_GUIDANCE = {
  name: {
    hint: "Use a short role name you would actually summon.",
    tooltip: "Good names are concrete and lowercase: coder, repo-researcher, support-triage.",
  },
  description: {
    hint:
      "Say what this profile is great at, what context it should prefer, and what good output looks like.",
    tooltip:
      "One focused sentence beats a vague personality. Example: Great at reading unfamiliar codebases, finding the smallest safe fix, and explaining tradeoffs.",
  },
  cloneFrom: {
    hint:
      "Clone when this profile should share keys, tools, and setup with an existing agent.",
    tooltip:
      "Start blank only when you want a profile to build its own habits, skills, and configuration from scratch.",
  },
  model: {
    hint:
      "Pick for the job style: stronger reasoning for coding or research, faster models for routine drafting.",
    tooltip:
      "You can leave this unset and choose later from the Models page or with the profile command.",
  },
  skills: {
    hint:
      "Start with the skills this profile will actually use; focused capability makes behavior easier to predict.",
    tooltip:
      "Coder profiles usually want repo, test, and review skills. Research profiles usually want search, docs, and summarization skills.",
  },
  mcp: {
    hint:
      "Add external systems only when they are part of this profile's job.",
    tooltip:
      "Examples: GitHub for code review, Jira for delivery work, Drive for docs-heavy research.",
  },
  soul: {
    hint:
      "Use SOUL.md for stable preferences: tone, decision rules, boundaries, and when to ask before acting.",
    tooltip:
      "Keep it durable. Session-specific instructions belong in the conversation, not the profile identity.",
  },
} as const;

export const PROFILE_EXAMPLES: ProfileExample[] = [
  {
    name: "coder",
    title: "Coder",
    description:
      "Great at reading unfamiliar codebases, identifying the smallest safe fix, and explaining tradeoffs before changing files.",
    bestFor: "Implementation, debugging, refactors",
  },
  {
    name: "repo-researcher",
    title: "Researcher",
    description:
      "Great at scanning docs, issues, and prior decisions, then turning messy context into concise implementation plans.",
    bestFor: "Discovery, specs, technical briefs",
  },
  {
    name: "ops-assistant",
    title: "Ops Assistant",
    description:
      "Great at watching recurring workflows, triaging exceptions, and producing clear next actions with owners.",
    bestFor: "Triage, status, follow-through",
  },
];
