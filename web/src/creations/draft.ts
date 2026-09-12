import type { CreationDraft } from "./types";
export const emptyDraft = (): CreationDraft => ({ spaceId: null, outputNodeIds: [], overrides: {}, providers: {}, plan: null, requestId: null });
export function editDraft(draft: CreationDraft, change: Partial<Pick<CreationDraft, "spaceId" | "outputNodeIds" | "overrides" | "providers">>): CreationDraft { return { ...draft, ...change, plan: null, requestId: null }; }
