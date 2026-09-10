import { authedFetch } from "@/lib/api";

export interface ChatFileUploadResult {
  path: string;
  name: string;
  bytes: number;
  mime_type: string;
}

function fileKey(file: File): string {
  return `${file.name}\0${file.type}\0${file.size}\0${file.lastModified}`;
}

/** Consume mirrored items/files one-for-one, preserving real multiplicity. */
export function filesFromTransfer(data: DataTransfer | null): File[] {
  if (!data) return [];
  const files: File[] = [];
  const mirroredItems = new Map<string, number>();
  for (const item of Array.from(data.items ?? [])) {
    if (item.kind !== "file") continue;
    const file = item.getAsFile();
    if (!file) continue;
    files.push(file);
    const key = fileKey(file);
    mirroredItems.set(key, (mirroredItems.get(key) ?? 0) + 1);
  }
  for (const file of Array.from(data.files ?? [])) {
    const key = fileKey(file);
    const remaining = mirroredItems.get(key) ?? 0;
    if (remaining) mirroredItems.set(key, remaining - 1);
    else files.push(file);
  }
  return files;
}

export function transferMayContainFiles(data: DataTransfer | null): boolean {
  return !!data && (
    Array.from(data.items ?? []).some(item => item.kind === "file") ||
    !!data.files?.length || Array.from(data.types ?? []).includes("Files")
  );
}

/** Browser bytes only; native draft insertion is a separate acknowledged action. */
export async function uploadChatFile(
  file: File,
  profile: string,
  signal?: AbortSignal,
): Promise<ChatFileUploadResult> {
  const body = new FormData();
  body.append("file", file, file.name);
  const qs = profile ? `?profile=${encodeURIComponent(profile)}` : "";
  const response = await authedFetch(`/api/chat/file-upload${qs}`, {
    method: "POST", body, signal,
  });
  if (!response.ok) throw new Error("Upload failed.");
  const uploaded = await response.json() as ChatFileUploadResult;
  // Preserve the staging identifier literally; malformed responses are not
  // attachment success and must never become native draft requests.
  // eslint-disable-next-line no-control-regex -- staging paths cannot contain control bytes
  if (typeof uploaded?.path !== "string" || !uploaded.path.startsWith("/") || /[\x00-\x1f\x7f-\x9f]/.test(uploaded.path)) {
    throw new Error("Upload failed.");
  }
  return uploaded;
}
