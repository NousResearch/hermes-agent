import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { SessionSearchResult } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";

type SessionQueryOptions = Parameters<typeof api.searchSessions>[1];

/**
 * Debounced full-text session search for the Sessions page. ``results`` is null while the box is
 * empty or a search is pending; ``error`` says the search failed, so the page never shows the
 * unfiltered list as if every session matched; ``retry`` runs the same query again.
 */
export function useSessionSearch(search: string, options: SessionQueryOptions) {
  const [results, setResults] = useState<SessionSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null);
  const requestRef = useRef(0);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    // Every run supersedes the search in flight, a cleared box included, so a
    // late answer to an older query never lands under the current one.
    const requestId = ++requestRef.current;
    const current = () => requestId === requestRef.current;

    if (!search.trim()) {
      debounceRef.current = setTimeout(() => {
        setResults(null);
        setError(null);
        setSearching(false);
      }, 0);
      return;
    }

    debounceRef.current = setTimeout(() => {
      setSearching(true);
      setResults(null);
      setError(null);
      api
        .searchSessions(search.trim(), options)
        .then((resp) => {
          if (current()) setResults(resp.results);
        })
        .catch((err) => {
          if (current()) setError(errorMessage(err));
        })
        .finally(() => {
          if (current()) setSearching(false);
        });
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [search, options, attempt]);

  return { results, searching, error, retry: () => setAttempt((n) => n + 1) };
}
