import type { SkillInfo } from "@/lib/api-skills";

/** Whether the Skills page offers Delete (archive) for a row. Local/agent-authored skills
 *  only: the backend refuses bundled, hub and external-dir skills (hub skills uninstall via
 *  the Hub tab). A path-form name (`a/dup`) is a same-tier duplicate listed by its relative
 *  path; the archive is keyed by frontmatter name, which two dirs share, so it cannot target
 *  that row and would 404. */
export function isArchivableSkill(skill: Pick<SkillInfo, "name" | "provenance">): boolean {
  return skill.provenance === "agent" && !skill.name.includes("/");
}
