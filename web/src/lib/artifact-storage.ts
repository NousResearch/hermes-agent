import type { ArtifactKind } from "@/lib/artifact-detect";

export const ARTIFACT_STORAGE_KEY = "hermes.native-chat.artifacts.v1";
export const MAX_PERSISTED_ARTIFACT_BYTES = 200_000;

export type StoredArtifact = {
  id: string;
  sessionId: string;
  kind: ArtifactKind;
  language: string;
  title: string;
  code: string;
  createdAt: number;
};

export function makeArtifactId(
  sessionId: string,
  kind: ArtifactKind,
  language: string,
  title: string,
  code: string,
): string {
  return [sessionId, kind, language, title, code.length, code.slice(0, 80), code.slice(-80)]
    .map((part) => encodeURIComponent(String(part)))
    .join("|");
}

function defaultStorage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

export function getArtifactStorage(): Storage | null {
  return defaultStorage();
}

function isStoredArtifact(value: unknown): value is StoredArtifact {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<StoredArtifact>;
  return typeof candidate.id === "string"
    && typeof candidate.sessionId === "string"
    && (candidate.kind === "code" || candidate.kind === "html" || candidate.kind === "svg")
    && typeof candidate.language === "string"
    && typeof candidate.title === "string"
    && typeof candidate.code === "string"
    && typeof candidate.createdAt === "number";
}

export function readStoredArtifacts(storage: Storage | null = defaultStorage()): StoredArtifact[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(ARTIFACT_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isStoredArtifact) : [];
  } catch {
    return [];
  }
}

export function isArtifactPinned(storage: Storage | null, id: string): boolean {
  return readStoredArtifacts(storage).some((artifact) => artifact.id === id);
}

export function setArtifactPinned(
  storage: Storage | null,
  artifact: StoredArtifact,
  pinned: boolean,
): boolean {
  if (!storage) return false;
  if (pinned && new TextEncoder().encode(artifact.code).byteLength > MAX_PERSISTED_ARTIFACT_BYTES) return false;
  const current = readStoredArtifacts(storage);
  const next = pinned
    ? [...current.filter((entry) => entry.id !== artifact.id), artifact]
    : current.filter((entry) => entry.id !== artifact.id);
  try {
    storage.setItem(ARTIFACT_STORAGE_KEY, JSON.stringify(next));
    return true;
  } catch {
    return false;
  }
}
