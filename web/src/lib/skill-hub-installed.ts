import { skillCatalogInstallIdentifier } from "@hermes/shared";

import type { SkillHubInstalledEntry, SkillHubResult } from "@/lib/api";

/**
 * The installed lock entry a hub search or featured hit names. The backend keys
 * a ClawHub install by its lock id (`@owner/slug`) and by the `clawhub/<slug>`
 * id a catalog card installs, but a ClawHub hit carries its bare slug. So a hit
 * is looked up under its own id, then under the id a catalog card would install
 * for it, which differs only for ClawHub.
 */
export function installedHubEntry(
  installed: Record<string, SkillHubInstalledEntry>,
  hit: Pick<SkillHubResult, "identifier" | "name" | "source">,
): SkillHubInstalledEntry | undefined {
  return (
    installed[hit.identifier] ??
    installed[skillCatalogInstallIdentifier(hit) ?? hit.identifier]
  );
}
