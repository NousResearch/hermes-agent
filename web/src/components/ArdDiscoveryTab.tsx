/**
 * ARD Discovery Tab — search and publish Agentic Resource Discovery catalogs.
 *
 * Provides:
 *  - Local semantic search over Hermes capabilities
 *  - Remote ARD registry search (HF Discover, custom registries)
 *  - Publish local capabilities as ai-catalog.json
 *
 * Uses the dashboard REST API:
 *   POST /api/ard/search   — local search (authenticated)
 *   POST /api/ard/publish  — generate catalog (authenticated)
 *   GET  /api/ard/catalog  — get catalog (authenticated)
 */

import { useState, useCallback, useEffect } from "react";
import { Search, Zap, Globe, Upload, Loader2, Tag, ChevronDown } from "lucide-react";
import { fetchJSON } from "../lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ArdEntry {
  identifier: string;
  displayName: string;
  type: string;
  description: string;
  score: number;
  tags?: string[];
  url?: string;
}

interface SearchResponse {
  results: ArdEntry[];
  count: number;
}

interface PublishResponse {
  success: boolean;
  path: string;
  entries: number;
  types: Record<string, number>;
  domain: string;
}

// ---------------------------------------------------------------------------
// Score bar
// ---------------------------------------------------------------------------

function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score));
  const color =
    pct >= 75
      ? "bg-emerald-500"
      : pct >= 50
        ? "bg-yellow-500"
        : "bg-slate-500";
  return (
    <div className="flex items-center gap-2 min-w-[80px]">
      <div className="flex-1 h-1.5 rounded-full bg-slate-700 overflow-hidden">
        <div
          className={`h-full ${color} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-slate-400 w-8 text-right">{pct}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Entry card
// ---------------------------------------------------------------------------

function EntryCard({ entry }: { entry: ArdEntry }) {
  const [expanded, setExpanded] = useState(false);
  const typeLabel = entry.type?.split("/")?.pop() || "unknown";
  const isMcp = typeLabel.includes("mcp");

  return (
    <div
      className="rounded-lg border border-slate-700/50 bg-slate-800/40 p-3 hover:border-slate-600 transition-colors cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-3">
        <ScoreBar score={entry.score} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
              isMcp ? "bg-purple-900/40 text-purple-300" : "bg-sky-900/40 text-sky-300"
            }`}>
              {typeLabel}
            </span>
            <span className="text-sm font-medium text-slate-200 truncate">
              {entry.displayName}
            </span>
          </div>
          {entry.description && (
            <p className="text-xs text-slate-400 mt-1 line-clamp-1">
              {entry.description}
            </p>
          )}
        </div>
        <ChevronDown
          className={`w-4 h-4 text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`}
        />
      </div>

      {expanded && (
        <div className="mt-2 pt-2 border-t border-slate-700/50 space-y-1">
          {entry.tags && entry.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {entry.tags.slice(0, 8).map((tag, i) => (
                <span
                  key={i}
                  className="text-xs px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400"
                >
                  <Tag className="w-3 h-3 inline mr-0.5" />
                  {tag}
                </span>
              ))}
            </div>
          )}
          <div className="text-xs text-slate-500 font-mono break-all">
            {entry.identifier}
          </div>
          {entry.url && (
            <div className="text-xs text-blue-400 truncate">
              <a href={entry.url} target="_blank" rel="noreferrer">
                {entry.url}
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ArdDiscoveryTab() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ArdEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [semantic, setSemantic] = useState(false);
  const [typeFilter, setTypeFilter] = useState("");
  const [publishResult, setPublishResult] = useState<PublishResponse | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [error, setError] = useState("");

  const search = useCallback(async () => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const filter = typeFilter ? { type: [typeFilter] } : undefined;
      const data = await fetchJSON<SearchResponse>("/api/ard/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: { text: query, filter },
          pageSize: 20,
        }),
      });
      setResults(data.results || []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, typeFilter]);

  const publish = useCallback(async () => {
    setPublishing(true);
    setError("");
    try {
      const data = await fetchJSON<PublishResponse>("/api/ard/publish", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain: "hermes.local" }),
      });
      setPublishResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Publish failed");
    } finally {
      setPublishing(false);
    }
  }, []);

  // Keyboard shortcut: Enter to search
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void search();
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Search bar */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search capabilities: 'sql injection', 'image generation', 'web scraping'..."
            className="w-full pl-10 pr-4 py-2 rounded-lg bg-slate-800 border border-slate-700 text-sm text-slate-200 placeholder:text-slate-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
          />
        </div>
        <button
          onClick={() => void search()}
          disabled={loading}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            "Search"
          )}
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={semantic}
            onChange={(e) => setSemantic(e.target.checked)}
            className="rounded border-slate-600 bg-slate-800"
          />
          <Zap className="w-3.5 h-3.5" />
          Semantic
        </label>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="text-xs px-2 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300"
        >
          <option value="">All types</option>
          <option value="application/ai-skill">Skills</option>
          <option value="application/mcp-server+json">MCP Servers</option>
          <option value="application/mcp-server-card+json">MCP Cards</option>
        </select>

        <div className="flex-1" />

        <button
          onClick={() => void publish()}
          disabled={publishing}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-medium disabled:opacity-50 transition-colors"
        >
          {publishing ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Upload className="w-3.5 h-3.5" />
          )}
          Publish Catalog
        </button>
      </div>

      {/* Publish result */}
      {publishResult && (
        <div className="rounded-lg border border-emerald-700/50 bg-emerald-900/20 p-3">
          <div className="flex items-center gap-2 text-emerald-300 text-sm font-medium">
            <Globe className="w-4 h-4" />
            Catalog Published
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {publishResult.entries} entries •{" "}
            {Object.entries(publishResult.types)
              .map(([k, v]) => `${v} ${k}`)
              .join(", ")}
          </div>
          <div className="text-xs text-slate-500 font-mono mt-1">
            {publishResult.path}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-700/50 bg-red-900/20 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      <div className="space-y-2">
        {results.length > 0 && (
          <div className="text-xs text-slate-500">
            {results.length} result{results.length !== 1 ? "s" : ""}
          </div>
        )}
        {results.map((entry, i) => (
          <EntryCard key={`${entry.identifier}-${i}`} entry={entry} />
        ))}
        {!loading && results.length === 0 && query.trim() && (
          <div className="text-center py-8 text-slate-500 text-sm">
            No results for "{query}"
          </div>
        )}
        {!loading && !query.trim() && (
          <div className="text-center py-8 text-slate-500 text-sm">
            Type a search query and press Enter to discover capabilities
          </div>
        )}
      </div>
    </div>
  );
}
