/**
 * useMemory — thin re-export wrapper around useMemoryData.
 *
 * Exists so callers can import from `@/hooks/useMemory` (the name that
 * matches the hook trio described in the acceptance criteria) without
 * having to change `useMemoryData` internals.
 *
 * @example
 *   const { data, loading, error, refetch, save } = useMemory("memory");
 *   const charCount = data?.char_count ?? 0;
 */
export {
  useMemoryData as useMemory,
  type MemoryDataClient as MemoryClient,
  type MemoryDataState as MemoryState,
  type MemoryEntry,
  type MemorySnapshot,
  type MemoryTarget,
} from "@/hooks/useMemoryData";
