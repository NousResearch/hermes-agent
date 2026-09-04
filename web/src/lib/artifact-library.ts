import type { ArtifactKind } from "@/lib/artifact-detect";
import type { StoredArtifact } from "@/lib/artifact-storage";

export type ArtifactKindFilter = ArtifactKind | "all";

export type ArtifactLibraryFilters = {
  query?: string;
  kind?: ArtifactKindFilter;
  sessionId?: string;
};

function normalized(value: string | undefined): string {
  return (value ?? "").trim().toLocaleLowerCase();
}

export function filterStoredArtifacts(
  artifacts: StoredArtifact[],
  filters: ArtifactLibraryFilters = {},
): StoredArtifact[] {
  const query = normalized(filters.query);
  const kind = filters.kind ?? "all";
  const sessionId = filters.sessionId ?? "all";

  return artifacts.filter((artifact) => {
    if (kind !== "all" && artifact.kind !== kind) return false;
    if (sessionId !== "all" && artifact.sessionId !== sessionId) return false;
    if (!query) return true;

    return [
      artifact.title,
      artifact.kind,
      artifact.language,
      artifact.sessionId,
      artifact.code,
    ].some((field) => normalized(field).includes(query));
  });
}

export function sortStoredArtifacts(artifacts: StoredArtifact[]): StoredArtifact[] {
  return [...artifacts].sort((left, right) =>
    right.createdAt - left.createdAt || right.id.localeCompare(left.id),
  );
}

export function artifactSessionIds(artifacts: StoredArtifact[]): string[] {
  return Array.from(new Set(artifacts.map((artifact) => artifact.sessionId))).sort((a, b) =>
    a.localeCompare(b),
  );
}
