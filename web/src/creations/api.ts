import { authedFetch, fetchJSON } from "@/lib/api";
import type { Approval, LibraryItem, LibraryPage, Plan, PlanInput, Run, RunsResponse, RuntimeStatus, SubmitResult, Workflow } from "./contracts";

const path = (suffix: string) => `/api/creations${suffix}`;
export const creationsApi = {
  getStatus: (signal?: AbortSignal) => fetchJSON<RuntimeStatus>(path("/status"), { signal }),
  getWorkflows: (signal?: AbortSignal) => fetchJSON<{ spaces?: Workflow[]; workflows?: Workflow[] }>(path("/workflows"), { signal }),
  createPlan: (input: PlanInput, signal?: AbortSignal) => fetchJSON<Plan>(path("/plans"), { method: "POST", body: JSON.stringify(input), headers: { "Content-Type": "application/json" }, signal }),
  submitRun: (planId: string, requestId: string, approval: Approval, signal?: AbortSignal) => fetchJSON<SubmitResult>(path("/runs"), { method: "POST", body: JSON.stringify({ planId, requestId, approval }), headers: { "Content-Type": "application/json" }, signal }),
  getRuns: (requestId?: string, signal?: AbortSignal) => fetchJSON<RunsResponse>(`${path("/runs")}${requestId ? `?requestId=${encodeURIComponent(requestId)}` : ""}`, { signal }),
  getRun: (id: string, signal?: AbortSignal) => fetchJSON<Run>(path(`/runs/${encodeURIComponent(id)}`), { signal }),
  retryRun: (id: string, nodeId: string, requestId: string, approval: Approval, signal?: AbortSignal) => fetchJSON<SubmitResult>(path(`/runs/${encodeURIComponent(id)}/retry`), { method: "POST", body: JSON.stringify({ nodeId, requestId, approval }), headers: { "Content-Type": "application/json" }, signal }),
  uploadAsset: async (file: Blob, signal?: AbortSignal) => { const res = await authedFetch(path("/uploads"), { method: "POST", body: file, headers: { "Content-Type": file.type }, signal }); if (!res.ok) throw new Error(`Upload failed (${res.status})`); return (await res.json()) as { assetUrl: string }; },
  getLibrary: (kind?: "image" | "video", cursor?: string, signal?: AbortSignal) => { const q = new URLSearchParams(); if (kind) q.set("kind", kind); if (cursor) q.set("cursor", cursor); return fetchJSON<LibraryPage>(`${path("/library")}?${q}`, { signal }); },
  reuseItem: (id: string, signal?: AbortSignal) => fetchJSON<{ assetUrl: string }>(path(`/library/${encodeURIComponent(id)}/reuse`), { method: "POST", signal }),
  retryCopy: (id: string, signal?: AbortSignal) => fetchJSON<LibraryItem>(path(`/library/${encodeURIComponent(id)}/copy-retry`), { method: "POST", signal }),
  media: (id: string, signal?: AbortSignal) => authedFetch(path(`/library/${encodeURIComponent(id)}/content`), { signal }),
};
