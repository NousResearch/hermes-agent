import type { Plan, PlanInput, Workflow } from "./contracts";
export interface CreationDraft extends Omit<PlanInput, "spaceId"> { spaceId: string | null; plan: Plan | null; requestId: string | null; }
export interface WorkflowWithStatus extends Workflow { generationNodes?: Array<{ id: string; kind?: string; model?: string; provider?: string; label?: string }>; }
