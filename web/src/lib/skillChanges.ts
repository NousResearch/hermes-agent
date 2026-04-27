import type { SkillChangeEvent } from "@/lib/api";

export function reasonKindLabel(kind: string): string {
  switch (kind) {
    case "explicit":
      return "Explicit reason";
    case "system":
      return "System-detected change";
    case "unattributed":
      return "No reason captured";
    case "model_summary":
      return "Model-generated summary";
    default:
      return "Unknown reason source";
  }
}

export function reviewStatusLabel(status: string): string {
  switch (status) {
    case "unreviewed":
      return "Unreviewed";
    case "reviewed":
      return "Reviewed";
    case "needs_followup":
      return "Needs follow-up";
    default:
      return status;
  }
}

export function latestChangeBySkill<T extends Pick<SkillChangeEvent, "skill" | "event_id">>(
  changes: readonly T[],
): Map<string, T> {
  const bySkill = new Map<string, T>();
  for (const change of changes) {
    if (!bySkill.has(change.skill)) {
      bySkill.set(change.skill, change);
    }
  }
  return bySkill;
}

export async function loadSkillChangesBestEffort<T>(
  loadChanges: () => Promise<T[]>,
  onError?: (error: unknown) => void,
): Promise<T[]> {
  try {
    return await loadChanges();
  } catch (error) {
    onError?.(error);
    return [];
  }
}

export function formatSkillChangeTime(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return date.toLocaleString();
}
