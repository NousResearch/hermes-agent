import { authedFetch } from "@/lib/api";

// Clipboard image MIME → file extension. Mirrors the set the TUI's /image
// attach path and the gateway's image sniffer accept.
const IMAGE_MIME_EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
};

// Anthropic caps a single vision image at ~25 MB; reject earlier client-side
// with a clear message rather than round-tripping a doomed upload.
const MAX_IMAGE_BYTES = 25 * 1024 * 1024;

export interface ChatImageUploadResult {
  /** Absolute path under HERMES_HOME/images the gateway wrote. */
  path: string;
  /** Byte size of the uploaded image. */
  bytes: number;
  /** Basename written on the server. */
  name: string;
  mime_type: string;
}

function imageFileKey(file: File): string {
  return `${file.name}\0${file.type}\0${file.size}\0${file.lastModified}`;
}

function addImageFile(files: File[], seen: Set<string>, file: File | null) {
  if (!file || !file.type.startsWith("image/")) return;
  const key = imageFileKey(file);
  if (seen.has(key)) return;
  seen.add(key);
  files.push(file);
}

/** Pull every image file out of a DataTransfer (clipboard or drop). */
export function imageFilesFromTransfer(
  data: DataTransfer | null,
): File[] {
  if (!data) return [];
  const files: File[] = [];
  const seen = new Set<string>();

  if (data.items?.length) {
    for (let i = 0; i < data.items.length; i++) {
      const item = data.items[i];
      if (item.kind === "file" && item.type.startsWith("image/")) {
        addImageFile(files, seen, item.getAsFile());
      }
    }
  }

  if (data.files?.length) {
    for (let i = 0; i < data.files.length; i++) {
      addImageFile(files, seen, data.files[i]);
    }
  }

  return files;
}

/** Pull the first image blob out of a DataTransfer, or null if none present. */
export function firstImageFromClipboard(
  data: DataTransfer | null,
): File | null {
  return imageFilesFromTransfer(data)[0] ?? null;
}

// Non-image chat uploads (PDFs, docs, archives) — server cap mirrored in
// /api/chat/file-upload.
const MAX_FILE_BYTES = 50 * 1024 * 1024;

/** Pull every non-image file out of a DataTransfer (clipboard or drop). */
export function nonImageFilesFromTransfer(
  data: DataTransfer | null,
): File[] {
  if (!data) return [];
  const files: File[] = [];
  const seen = new Set<string>();
  const add = (file: File | null) => {
    if (!file || file.type.startsWith("image/")) return;
    const key = imageFileKey(file);
    if (seen.has(key)) return;
    seen.add(key);
    files.push(file);
  };

  if (data.items?.length) {
    for (let i = 0; i < data.items.length; i++) {
      const item = data.items[i];
      if (item.kind === "file" && !item.type.startsWith("image/")) {
        add(item.getAsFile());
      }
    }
  }

  if (data.files?.length) {
    for (let i = 0; i < data.files.length; i++) {
      add(data.files[i]);
    }
  }

  return files;
}

/** True when a drag payload may contain any file (for dragover preventDefault). */
export function transferMayContainFile(data: DataTransfer | null): boolean {
  if (!data) return false;
  if (data.items?.length) {
    for (let i = 0; i < data.items.length; i++) {
      if (data.items[i].kind === "file") return true;
    }
    return false;
  }
  return (data.files?.length ?? 0) > 0;
}

/**
 * Upload a non-image chat file to ``HERMES_HOME/uploads`` via the generic
 * chat upload endpoint and return the absolute gateway path. The caller
 * types the path into the TUI input so the agent can read the file with
 * its file tools.
 */
export async function uploadChatFile(
  file: File,
  profile = "",
): Promise<ChatImageUploadResult> {
  if (file.size === 0) throw new Error("file is empty");
  if (file.size > MAX_FILE_BYTES) {
    const mb = Math.round(MAX_FILE_BYTES / (1024 * 1024));
    throw new Error(`file too large (max ${mb} MB)`);
  }

  const dataUrl = await fileToDataUrl(file);
  const qs = profile ? `?profile=${encodeURIComponent(profile)}` : "";
  const res = await authedFetch(`/api/chat/file-upload${qs}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data_url: dataUrl,
      filename: file.name,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }

  const uploaded = (await res.json()) as ChatImageUploadResult;
  if (!uploaded?.path) {
    throw new Error("file upload did not return a path");
  }
  return uploaded;
}

/** True when a drag payload may contain an image (for dragover preventDefault). */
export function transferMayContainImage(data: DataTransfer | null): boolean {
  if (!data) return false;
  if (data.items?.length) {
    for (let i = 0; i < data.items.length; i++) {
      const item = data.items[i];
      if (
        item.kind === "file" &&
        (!item.type || item.type.startsWith("image/"))
      ) {
        return true;
      }
    }
    return false;
  }
  if (data.files?.length) {
    for (let i = 0; i < data.files.length; i++) {
      if (data.files[i].type.startsWith("image/")) return true;
    }
  }
  return false;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(reader.error ?? new Error("image read failed"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        resolve(result);
      } else {
        reject(new Error("image read failed"));
      }
    };
    reader.readAsDataURL(file);
  });
}

/**
 * Upload a browser clipboard/drop image to ``HERMES_HOME/images`` via the
 * dedicated chat upload endpoint and return the absolute gateway path.
 *
 * The dashboard Chat tab is an xterm mirror of a TUI running INSIDE the
 * gateway. The container has no access to the browser's clipboard, so the
 * server-side ``clipboard.paste`` path can never see a pasted image.
 * Upload the bytes the browser already holds, then hand the path to the
 * TUI's ``/image`` command.
 */
export async function uploadChatImage(
  blob: Blob,
  profile = "",
): Promise<ChatImageUploadResult> {
  if (blob.size === 0) throw new Error("clipboard image is empty");
  if (blob.size > MAX_IMAGE_BYTES) {
    const mb = Math.round(MAX_IMAGE_BYTES / (1024 * 1024));
    throw new Error(`image too large (max ${mb} MB)`);
  }

  const mime = blob.type || "image/png";
  const ext = IMAGE_MIME_EXT[mime] || "png";
  const filename =
    blob instanceof File && blob.name
      ? blob.name
      : `clipboard.${ext}`;
  const file =
    blob instanceof File
      ? blob
      : new File([blob], filename, { type: mime });

  const dataUrl = await fileToDataUrl(file);
  const qs = profile ? `?profile=${encodeURIComponent(profile)}` : "";
  const res = await authedFetch(`/api/chat/image-upload${qs}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data_url: dataUrl,
      filename,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }

  const uploaded = (await res.json()) as ChatImageUploadResult;
  if (!uploaded?.path) {
    throw new Error("image upload did not return a path");
  }
  return uploaded;
}
