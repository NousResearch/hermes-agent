export type Kind = "image" | "video";
export type Readiness = "disabled" | "offline" | "incompatible" | "ready";
export type RunState = "queued" | "running" | "succeeded" | "failed" | "reconciliation_required";
export type StepState = "queued" | "submitting" | "pending" | "done" | "error" | "blocked" | "reconciliation_required";
export interface PlanInput { spaceId: string; outputNodeIds: string[]; overrides: Record<string, { prompt?: string; assetUrl?: string }>; providers: Record<string, "codex" | "kie">; }
export interface Approval { fingerprint: string; maxImages: number; maxVideos: number; maxVideoSeconds: number; allowPremium: boolean; allowPaid: boolean; acknowledgement: string; }
export interface Output { id: string; nodeId: string; kind: Kind; url: string; mimeType: string; }
export interface PlanStep { nodeId: string; kind: Kind; model: string; provider: "codex" | "kie"; dependsOn: string[]; payloadPreview: Record<string, unknown>; }
export interface Plan { planId: string; fingerprint: string; spaceId: string; spaceName: string; steps: PlanStep[]; warnings: string[]; expiresAt: string; }
export interface RunStep extends PlanStep { attempt: number; taskId: string | null; state: StepState; error: string | null; outputs: Output[]; creditBefore: number | null; creditAfter: number | null; startedAt?: string | null; finishedAt?: string | null; }
export interface Run { runId: string; requestId: string; planId: string; fingerprint: string; spaceId: string; spaceName: string; state: RunState; createdAt: string; updatedAt: string; steps: RunStep[]; outputs: Output[]; }
export interface Workflow { id: string; name: string; nodes?: Array<Record<string, unknown>>; edges?: Array<Record<string, unknown>>; readiness?: Readiness; reason?: string; }
export interface RuntimeStatus { protocolVersion: number; readiness: Readiness; capabilities?: { nodeTypes: string[]; targetHandles: string[] }; imageModels?: Array<Record<string, unknown>>; videoModels?: Array<Record<string, unknown>>; anvilUrl?: string; reason?: string; }
export interface LibraryItem { id: string; runId: string; nodeId: string; kind: Kind; mimeType: string; filename: string; sha256: string; bytes: number; createdAt: string; model: string; provider: string; sourceUrl?: string; storageState: "copying" | "local" | "copy_failed"; }
export interface LibraryPage { items: LibraryItem[]; nextCursor: string | null; }
export type SubmitResult = Run | { requestId: string; state: "reconciling" };
export interface RunsResponse { runs: Run[]; pendingRequests: Array<{ requestId: string; state: string }>; }
