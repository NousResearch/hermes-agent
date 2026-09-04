import type { ArtifactKind } from "@/lib/artifact-detect";

/** Legacy unscoped key kept for migration and source compatibility. */
export const LEGACY_ARTIFACT_STORAGE_KEY = "hermes.native-chat.artifacts.v1";
export const ARTIFACT_STORAGE_KEY = LEGACY_ARTIFACT_STORAGE_KEY;
export const ARTIFACT_STORAGE_PREFIX = "hermes.native-chat.artifacts.v3.";
export const LEGACY_PROFILE_ARTIFACT_STORAGE_PREFIX = "hermes.native-chat.artifacts.v2.";
export const ARTIFACT_PROFILE_IDENTITIES_STORAGE_KEY = "hermes.native-chat.artifact-profile-identities.v1";
export const ARTIFACT_STORAGE_CHANGE_EVENT = "hermes-native-chat-artifacts-change";
export const MAX_PERSISTED_ARTIFACT_BYTES = 200_000;
export const MAX_PERSISTED_ARTIFACT_STORAGE_BYTES = 2_000_000;

const volatileProfileIdentities = new Map<string, string>();
let volatileIdentityCounter = 0;

export type StoredArtifact = {
  id: string;
  sessionId: string;
  kind: ArtifactKind;
  language: string;
  title: string;
  code: string;
  createdAt: number;
};

export type ArtifactStorageFailureReason =
  | "storage-unavailable"
  | "artifact-too-large"
  | "storage-limit"
  | "storage-quota";

export type ArtifactStorageWriteResult =
  | { ok: true; artifacts: StoredArtifact[]; bytes: number }
  | {
      ok: false;
      reason: ArtifactStorageFailureReason;
      artifacts: StoredArtifact[];
      bytes: number;
    };

function hashArtifactCode(code: string): string {
  // Keep IDs compact and independent from artifact source text while hashing
  // every code unit. Two independent 32-bit lanes distinguish middle-only
  // edits without requiring async crypto.subtle in render paths.
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < code.length; index += 1) {
    const char = code.charCodeAt(index);
    first = Math.imul(first ^ char, 0x01000193);
    second = Math.imul(second ^ (char + index), 0x85ebca6b);
    second = (second << 13) | (second >>> 19);
  }
  return `${(first >>> 0).toString(16).padStart(8, "0")}${(second >>> 0).toString(16).padStart(8, "0")}`;
}

export function makeArtifactId(
  sessionId: string,
  kind: ArtifactKind,
  language: string,
  title: string,
  code: string,
): string {
  return [sessionId, kind, language, title, code.length, hashArtifactCode(code)]
    .map((part) => encodeURIComponent(String(part)))
    .join("|");
}

function canonicalizeArtifact(artifact: StoredArtifact): StoredArtifact {
  const id = makeArtifactId(
    artifact.sessionId,
    artifact.kind,
    artifact.language,
    artifact.title,
    artifact.code,
  );
  return artifact.id === id ? artifact : { ...artifact, id };
}

function defaultStorage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function profileScopeName(profile?: string): string {
  return profile || "default";
}

function createProfileIdentity(): string {
  const cryptoApi = typeof globalThis !== "undefined" ? globalThis.crypto : undefined;
  if (cryptoApi?.randomUUID) return cryptoApi.randomUUID();
  if (cryptoApi?.getRandomValues) {
    const values = new Uint32Array(4);
    cryptoApi.getRandomValues(values);
    return Array.from(values, (value) => value.toString(16).padStart(8, "0")).join("");
  }
  volatileIdentityCounter += 1;
  return `volatile-${Date.now().toString(36)}-${volatileIdentityCounter.toString(36)}`;
}

function readProfileIdentities(storage: Storage | null): Record<string, string> {
  if (!storage) return {};
  try {
    const raw = storage.getItem(ARTIFACT_PROFILE_IDENTITIES_STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
    return Object.fromEntries(
      Object.entries(parsed).filter((entry): entry is [string, string] =>
        typeof entry[0] === "string" && typeof entry[1] === "string" && entry[1].length >= 8,
      ),
    );
  } catch {
    return {};
  }
}

function getProfileIdentity(storage: Storage | null, profile?: string): string {
  const scope = profileScopeName(profile);
  const persisted = readProfileIdentities(storage);
  const persistedIdentity = persisted[scope];
  if (persistedIdentity) return persistedIdentity;

  const existing = volatileProfileIdentities.get(scope);
  if (existing) {
    if (storage) {
      try {
        storage.setItem(ARTIFACT_PROFILE_IDENTITIES_STORAGE_KEY, JSON.stringify({ ...persisted, [scope]: existing }));
      } catch {
        // Keep the identity stable for this page even when persistence is full.
      }
    }
    return existing;
  }

  const identity = createProfileIdentity();
  volatileProfileIdentities.set(scope, identity);
  if (storage) {
    try {
      storage.setItem(ARTIFACT_PROFILE_IDENTITIES_STORAGE_KEY, JSON.stringify({ ...persisted, [scope]: identity }));
    } catch {
      // Keep the identity stable for this page even when persistence is full.
    }
  }
  return identity;
}

export function getArtifactStorage(): Storage | null {
  return defaultStorage();
}

export function getArtifactStorageKey(profile?: string, storage: Storage | null = defaultStorage()): string {
  const scope = profileScopeName(profile);
  return `${ARTIFACT_STORAGE_PREFIX}${encodeURIComponent(scope)}.${getProfileIdentity(storage, scope)}`;
}

export function getLegacyArtifactStorageKey(profile?: string): string {
  const legacyScope = !profile || profile === "default" ? "" : profile;
  return `${LEGACY_PROFILE_ARTIFACT_STORAGE_PREFIX}${encodeURIComponent(legacyScope)}`;
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
    && typeof candidate.createdAt === "number"
    && Number.isSafeInteger(candidate.createdAt)
    && candidate.createdAt >= -8_640_000_000_000_000
    && candidate.createdAt <= 8_640_000_000_000_000;
}

function readKey(storage: Storage | null, key: string): StoredArtifact[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(key);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    const artifacts = Array.isArray(parsed)
      ? parsed
        .filter(isStoredArtifact)
        .filter((artifact) => utf8ByteLength(artifact.code) <= MAX_PERSISTED_ARTIFACT_BYTES)
        .map(canonicalizeArtifact)
      : [];
    return fitArtifactsWithinStorageLimit(artifacts);
  } catch {
    return [];
  }
}

function migrateScopedArtifacts(storage: Storage | null, profile = ""): boolean {
  if (!storage) return false;
  const legacyKey = getLegacyArtifactStorageKey(profile);
  let raw: string | null;
  try {
    raw = storage.getItem(legacyKey);
  } catch {
    return false;
  }
  if (!raw) return false;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return false;
  }
  if (!Array.isArray(parsed)) return false;

  const targetKey = getArtifactStorageKey(profile, storage);
  const current = readKey(storage, targetKey);
  const legacy = readKey(storage, legacyKey);
  const merged = [...current];
  for (const artifact of legacy) {
    if (!merged.some((entry) => entry.id === artifact.id)) merged.push(artifact);
  }
  const bounded = fitArtifactsWithinStorageLimit(merged);
  try {
    storage.setItem(targetKey, JSON.stringify(bounded));
    storage.removeItem(legacyKey);
    notifyArtifactStorageChanged(storage, profile);
    return true;
  } catch {
    return false;
  }
}

/**
 * Move v1's unscoped records into the default management-profile scope.
 * Named profiles deliberately do not inherit these records: the old key had
 * no profile identity, so copying it to every profile would leak local data.
 */
export function migrateLegacyArtifacts(storage: Storage | null): boolean {
  if (!storage) return false;
  const changed = migrateScopedArtifacts(storage, "");
  let raw: string | null;
  try {
    raw = storage.getItem(LEGACY_ARTIFACT_STORAGE_KEY);
  } catch {
    return changed;
  }
  if (!raw) return changed;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Keep malformed data untouched so a transient storage/parser issue cannot
    // silently destroy the user's only copy.
    return changed;
  }
  if (!Array.isArray(parsed)) return changed;
  const legacy = Array.isArray(parsed)
    ? parsed
      .filter(isStoredArtifact)
      .filter((artifact) => utf8ByteLength(artifact.code) <= MAX_PERSISTED_ARTIFACT_BYTES)
      .map(canonicalizeArtifact)
    : [];
  const targetKey = getArtifactStorageKey("", storage);
  const current = readKey(storage, targetKey);
  const merged = [...current];
  for (const artifact of legacy) {
    if (!merged.some((entry) => entry.id === artifact.id)) merged.push(artifact);
  }
  const bounded = fitArtifactsWithinStorageLimit(merged);

  try {
    storage.setItem(targetKey, JSON.stringify(bounded));
    storage.removeItem(LEGACY_ARTIFACT_STORAGE_KEY);
    notifyArtifactStorageChanged(storage, "");
    return true;
  } catch {
    return changed;
  }
}

export function readStoredArtifacts(
  storage: Storage | null = defaultStorage(),
  profile = "",
): StoredArtifact[] {
  if (!storage) return [];
  if (profile === "" || profile === "default") migrateLegacyArtifacts(storage);
  else migrateScopedArtifacts(storage, profile);
  return readKey(storage, getArtifactStorageKey(profile, storage));
}

export function isArtifactPinned(
  storage: Storage | null,
  id: string,
  profile = "",
): boolean {
  return readStoredArtifacts(storage, profile).some((artifact) => artifact.id === id);
}

function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function fitArtifactsWithinStorageLimit(artifacts: StoredArtifact[]): StoredArtifact[] {
  const bounded: StoredArtifact[] = [];
  for (const artifact of artifacts) {
    if (utf8ByteLength(artifact.code) > MAX_PERSISTED_ARTIFACT_BYTES) continue;
    const candidate = [...bounded, artifact];
    if (artifactStorageBytes(candidate) <= MAX_PERSISTED_ARTIFACT_STORAGE_BYTES) bounded.push(artifact);
  }
  return bounded;
}

function notifyArtifactStorageChanged(storage: Storage | null, profile: string, allProfiles = false): void {
  if (typeof window === "undefined" || typeof window.dispatchEvent !== "function") return;
  window.dispatchEvent(new CustomEvent(ARTIFACT_STORAGE_CHANGE_EVENT, {
    detail: allProfiles ? { profile, allProfiles: true } : { key: getArtifactStorageKey(profile, storage) },
  }));
}

export function artifactStorageBytes(artifacts: StoredArtifact[]): number {
  return artifacts.length === 0 ? 0 : utf8ByteLength(JSON.stringify(artifacts));
}

function writeArtifacts(
  storage: Storage | null,
  profile: string,
  artifacts: StoredArtifact[],
): ArtifactStorageWriteResult {
  const bytes = artifactStorageBytes(artifacts);
  if (!storage) return { ok: false, reason: "storage-unavailable", artifacts, bytes };
  if (bytes > MAX_PERSISTED_ARTIFACT_STORAGE_BYTES) {
    return { ok: false, reason: "storage-limit", artifacts, bytes };
  }
  try {
    storage.setItem(getArtifactStorageKey(profile, storage), JSON.stringify(artifacts));
    notifyArtifactStorageChanged(storage, profile);
    return { ok: true, artifacts, bytes };
  } catch {
    return { ok: false, reason: "storage-quota", artifacts, bytes };
  }
}

export function setArtifactPinnedWithResult(
  storage: Storage | null,
  artifact: StoredArtifact,
  pinned: boolean,
  profile = "",
): ArtifactStorageWriteResult {
  if (!storage) return { ok: false, reason: "storage-unavailable", artifacts: [], bytes: 0 };
  if (pinned && utf8ByteLength(artifact.code) > MAX_PERSISTED_ARTIFACT_BYTES) {
    const current = readStoredArtifacts(storage, profile);
    return {
      ok: false,
      reason: "artifact-too-large",
      artifacts: current,
      bytes: artifactStorageBytes(current),
    };
  }

  // Re-read immediately before writing so a pin made by another tab between
  // the first read and this mutation is retained rather than clobbered.
  const before = readStoredArtifacts(storage, profile);
  const latest = readStoredArtifacts(storage, profile);
  const base = JSON.stringify(before) === JSON.stringify(latest) ? before : latest;
  const reconciled = pinned
    ? [...base.filter((entry) => entry.id !== artifact.id), artifact]
    : base.filter((entry) => entry.id !== artifact.id);
  return writeArtifacts(storage, profile, reconciled);
}

export function setArtifactPinned(
  storage: Storage | null,
  artifact: StoredArtifact,
  pinned: boolean,
  profile = "",
): boolean {
  return setArtifactPinnedWithResult(storage, artifact, pinned, profile).ok;
}

export function clearStoredArtifacts(
  storage: Storage | null,
  profile = "",
): ArtifactStorageWriteResult {
  const current = readStoredArtifacts(storage, profile);
  if (!storage) return { ok: false, reason: "storage-unavailable", artifacts: current, bytes: 0 };
  try {
    storage.removeItem(getArtifactStorageKey(profile, storage));
    storage.removeItem(getLegacyArtifactStorageKey(profile));
    if (profile === "" || profile === "default") storage.removeItem(LEGACY_ARTIFACT_STORAGE_KEY);
    notifyArtifactStorageChanged(storage, profile);
    return { ok: true, artifacts: [], bytes: 0 };
  } catch {
    return { ok: false, reason: "storage-quota", artifacts: current, bytes: artifactStorageBytes(current) };
  }
}

function removeProfileArtifactKeys(storage: Storage, scope: string): void {
  const prefix = `${ARTIFACT_STORAGE_PREFIX}${encodeURIComponent(scope)}.`;
  const keys: string[] = [];
  for (let index = storage.length - 1; index >= 0; index -= 1) {
    const key = storage.key(index);
    if (key?.startsWith(prefix)) keys.push(key);
  }
  for (const key of keys) storage.removeItem(key);
  storage.removeItem(getLegacyArtifactStorageKey(scope));
  if (scope === "default") storage.removeItem(LEGACY_ARTIFACT_STORAGE_KEY);
}

/**
 * Rotate a profile's local identity and remove its old artifact records.
 * Rotating before removal means a delete/rename cannot expose old records if
 * the browser refuses the later removeItem call.
 */
export function forgetArtifactProfileScope(storage: Storage | null, profile = ""): boolean {
  if (!storage) return false;
  const scope = profileScopeName(profile);
  getArtifactStorageKey(profile, storage);
  const identities = readProfileIdentities(storage);
  const replacement = createProfileIdentity();
  let rotated = false;

  try {
    storage.setItem(
      ARTIFACT_PROFILE_IDENTITIES_STORAGE_KEY,
      JSON.stringify({ ...identities, [scope]: replacement }),
    );
    rotated = true;
  } catch {
    // Removal below can still succeed when the storage quota is full.
  }

  try {
    removeProfileArtifactKeys(storage, scope);
    volatileProfileIdentities.delete(scope);
    notifyArtifactStorageChanged(storage, profile, true);
    return true;
  } catch {
    // A persisted replacement keeps old records unreachable even if cleanup
    // is refused. Keep the current page on that replacement as well.
    if (rotated) volatileProfileIdentities.set(scope, replacement);
    return false;
  }
}

/**
 * Forget identities for profiles no longer returned by the profile API. Their
 * old v3/v2 keys become unreachable even when deletion happened elsewhere.
 */
export function pruneArtifactProfileIdentities(storage: Storage | null, profileNames: string[]): boolean {
  if (!storage) return false;
  const known = new Set(profileNames.map(profileScopeName));
  known.add("default");
  const identities = readProfileIdentities(storage);
  const stale = Object.entries(identities).filter(([scope]) => !known.has(scope));
  if (stale.length === 0) return true;

  const next = { ...identities };
  for (const [scope] of stale) delete next[scope];
  let rotated = false;
  try {
    storage.setItem(ARTIFACT_PROFILE_IDENTITIES_STORAGE_KEY, JSON.stringify(next));
    rotated = true;
  } catch {
    // Continue with physical cleanup; removeItem can work when setItem is at
    // quota, and an empty old key is safer than an accessible old profile.
  }

  try {
    for (const [scope] of stale) {
      removeProfileArtifactKeys(storage, scope);
      volatileProfileIdentities.delete(scope);
    }
    notifyArtifactStorageChanged(storage, "", true);
    return rotated;
  } catch {
    return false;
  }
}
