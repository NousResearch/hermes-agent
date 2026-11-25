import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { api, type SessionStoreStats } from "@/lib/api";
import { useProfileScope } from "@/contexts/useProfileScope";
import {
  sourceBelongsToCategory,
  sourceLabel,
  type SessionFilterCategory,
  type SourceSelectionsByCategory,
} from "@/pages/SessionsPage_sources";

/**
 * Global session-filter state for the dashboard.
 *
 * The Sessions page's filter (category chats/automation/all + per-category
 * source subset) lives here instead of the page so the sidebar's Filters
 * dropdown — rendered next to the profile switcher — can drive it from any
 * page. The provider also owns the session-store stats (per-source counts)
 * that feed the source rows, refreshing them when the managed profile
 * changes.
 *
 * Page-local side effects (reset to page 0, collapse expansion, clear the
 * bulk selection) are NOT part of this state; the Sessions page reacts to
 * changes itself.
 */
interface SessionFilterContextValue {
  /** Filter category: chats (default) | automation | all. */
  category: SessionFilterCategory;
  setCategory: (category: SessionFilterCategory) => void;
  /** Explicit per-category source subsets; null = the category default. */
  sourceSelections: SourceSelectionsByCategory;
  /** The explicit subset for the current category (null = default). */
  selectedSources: string[] | null;
  toggleSource: (source: string) => void;
  /** Clears the explicit subset for the current category (back to default). */
  clearSources: () => void;
  /** Session-store stats (per-source counts, archived, messages). */
  stats: SessionStoreStats | null;
  /** Re-fetches stats (call after mutations: import, delete, prune). */
  refreshStats: () => void;
  /** Every known source with its count, most-used first (pinned kept). */
  allSourceOptions: [string, number][];
  /** Sources visible for the current category (plus pinned selections). */
  sourceOptions: [string, number][];
  /** Set of effectively selected sources (explicit subset or default). */
  selectedSourceSet: Set<string>;
  /**
   * Number of active filter dimensions: 0 when the list is unfiltered
   * (chats + default sources), 1-2 when a category and/or a source subset
   * is applied. Drives the sidebar badge and the page status chip.
   */
  activeCount: number;
}

const SessionFilterContext = createContext<SessionFilterContextValue | null>(
  null,
);

const DEFAULT_SOURCE_SELECTIONS: SourceSelectionsByCategory = {
  chats: null,
  automation: null,
  all: null,
};

/**
 * Standalone fallback for a consumer rendered without the provider (the
 * Sessions page mounted on its own, or an isolated page test). The filter
 * itself is provider-owned, so with no provider the page behaves as the
 * unfiltered default: chats category, no explicit source subset.
 */
const STANDALONE_SESSION_FILTERS: SessionFilterContextValue = {
  category: "chats",
  setCategory: () => {},
  sourceSelections: DEFAULT_SOURCE_SELECTIONS,
  selectedSources: null,
  toggleSource: () => {},
  clearSources: () => {},
  stats: null,
  refreshStats: () => {},
  allSourceOptions: [],
  sourceOptions: [],
  selectedSourceSet: new Set<string>(),
  activeCount: 0,
};

export function SessionFilterProvider({ children }: { children: ReactNode }) {
  const { profile } = useProfileScope();
  const [category, setCategoryState] =
    useState<SessionFilterCategory>("chats");
  const [sourceSelections, setSourceSelections] =
    useState<SourceSelectionsByCategory>(DEFAULT_SOURCE_SELECTIONS);
  const [stats, setStats] = useState<SessionStoreStats | null>(null);

  // Profile-scoped: the api layer resolves the management profile for each
  // call, so a profile switch re-fetches the counts that feed the rows.
  const refreshStats = useCallback(() => {
    api
      .getSessionStats()
      .then(setStats)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats, profile]);

  const setCategory = useCallback(
    (next: SessionFilterCategory) => setCategoryState(next),
    [],
  );

  const selectedSources = sourceSelections[category];

  const pinnedSourceSelections = useMemo(
    () =>
      Object.values(sourceSelections).flatMap(
        (selection) => selection ?? [],
      ),
    [sourceSelections],
  );

  const allSourceOptions = useMemo<[string, number][]>(() => {
    const entries = Object.entries(stats?.by_source ?? {}).sort(
      ([aSource, aCount], [bSource, bCount]) =>
        bCount - aCount ||
        sourceLabel(aSource).localeCompare(sourceLabel(bSource)),
    );
    const seen = new Set(entries.map(([source]) => source));
    for (const source of pinnedSourceSelections) {
      if (!seen.has(source)) {
        entries.unshift([source, 0]);
        seen.add(source);
      }
    }
    return entries;
  }, [stats, pinnedSourceSelections]);

  const categoryDefaultSources = useMemo(
    () =>
      allSourceOptions
        .map(([source]) => source)
        .filter((source) => sourceBelongsToCategory(source, category)),
    [allSourceOptions, category],
  );

  const sourceOptions = useMemo<[string, number][]>(() => {
    const selected = new Set(selectedSources ?? []);
    return allSourceOptions.filter(
      ([source]) =>
        sourceBelongsToCategory(source, category) || selected.has(source),
    );
  }, [allSourceOptions, selectedSources, category]);

  const effectiveSelectedSources = selectedSources ?? categoryDefaultSources;

  const selectedSourceSet = useMemo(
    () => new Set(effectiveSelectedSources),
    [effectiveSelectedSources],
  );

  const toggleSource = useCallback(
    (source: string) => {
      setSourceSelections((current) => {
        const next = new Set(current[category] ?? categoryDefaultSources);
        if (next.has(source)) {
          next.delete(source);
        } else {
          next.add(source);
        }
        const nextSelection = sourceOptions
          .map(([optionSource]) => optionSource)
          .filter((optionSource) => next.has(optionSource));
        return {
          ...current,
          [category]: nextSelection,
        };
      });
    },
    [category, categoryDefaultSources, sourceOptions],
  );

  const clearSources = useCallback(() => {
    setSourceSelections((current) => ({
      ...current,
      [category]: null,
    }));
  }, [category]);

  const activeCount =
    (category !== "chats" ? 1 : 0) + (selectedSources !== null ? 1 : 0);

  const value = useMemo<SessionFilterContextValue>(
    () => ({
      category,
      setCategory,
      sourceSelections,
      selectedSources,
      toggleSource,
      clearSources,
      stats,
      refreshStats,
      allSourceOptions,
      sourceOptions,
      selectedSourceSet,
      activeCount,
    }),
    [
      category,
      setCategory,
      sourceSelections,
      selectedSources,
      toggleSource,
      clearSources,
      stats,
      refreshStats,
      allSourceOptions,
      sourceOptions,
      selectedSourceSet,
      activeCount,
    ],
  );

  return (
    <SessionFilterContext.Provider value={value}>
      {children}
    </SessionFilterContext.Provider>
  );
}

export function useSessionFilters(): SessionFilterContextValue {
  return useContext(SessionFilterContext) ?? STANDALONE_SESSION_FILTERS;
}
