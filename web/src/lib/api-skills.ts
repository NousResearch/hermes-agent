// Skill list/editor API types for the dashboard Skills page. Kept out of api.ts so
// that facade stays under its code-health line cap.

export interface SkillInfo {
  name: string;
  description: string;
  category: string;
  enabled: boolean;
  /** "hub" | "bundled" | "external" | "agent" — agent covers agent-authored + hand-made. */
  provenance?: string;
}

export interface SkillContent {
  name: string;
  content: string;
  path: string;
}

export interface SkillWriteResult {
  success: boolean;
  message?: string;
  path?: string;
  error?: string;
}

/** DELETE /api/skills archives the skill; ``message`` names the restore command. */
export interface SkillDeleteResult {
  ok: boolean;
  message: string;
}
