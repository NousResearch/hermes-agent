import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ListFilter } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";

import { useSessionFilters } from "@/contexts/session-filter-context";
import {
  SOURCE_CONFIG,
  sourceLabel,
  type SessionFilterCategory,
} from "@/pages/SessionsPage_sources";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Sidebar Filters dropdown — the session filter (type + sources) rendered
 * next to the profile switcher, so scoping controls sit together: which
 * profile, then which of its sessions to show.
 *
 * Replaces the Sessions page toolbar controls; the page keeps a slim
 * status chip so the active filter stays visible next to the data it
 * filters. State lives in SessionFilterContext (see that file).
 */
export function SessionFilterMenu({ collapsed }: SessionFilterMenuProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const {
    category,
    setCategory,
    sourceOptions,
    selectedSourceSet,
    toggleSource,
    clearSources,
    activeCount,
    stats,
  } = useSessionFilters();

  // Close on outside click / Escape — same behaviour as the toolbar menu
  // this replaces.
  useEffect(() => {
    if (!open) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const categoryOptions: { value: SessionFilterCategory; label: string }[] = [
    { value: "chats", label: t.sessions.filterChats },
    { value: "automation", label: t.sessions.filterAutomation },
    { value: "all", label: t.sessions.filterAll },
  ];

  const filtersLabel = t.app.filters ?? "Filters";
  const sessionFiltersLabel = t.app.sessionFilters ?? "Session filters";
  const filterTypeLabel = t.sessions.filterType ?? "Type";
  const filterSourcesLabel = t.sessions.filterSources ?? "Sources";

  const clearAll = () => {
    setCategory("chats");
    clearSources();
  };

  return (
    <div
      ref={rootRef}
      className={cn(
        "relative border-b border-current/10 px-3 py-2",
        collapsed && "lg:px-0 lg:py-1.5",
      )}
    >
      {collapsed ? (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label={filtersLabel}
          aria-haspopup="menu"
          aria-expanded={open}
          title={filtersLabel}
          className={cn(
            "relative mx-auto flex h-7 w-7 items-center justify-center rounded",
            "text-text-secondary hover:bg-midground/5 hover:text-foreground",
            activeCount > 0 && "text-primary hover:text-primary",
          )}
        >
          <ListFilter className="h-3.5 w-3.5" />
          {activeCount > 0 && (
            <span className="absolute -right-1 -top-1 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary px-0.5 text-[0.5625rem] font-semibold text-primary-foreground tabular-nums">
              {activeCount}
            </span>
          )}
        </button>
      ) : (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label={filtersLabel}
          aria-haspopup="menu"
          aria-expanded={open}
          className={cn(
            "flex h-7 w-full items-center gap-2 rounded border px-2 text-xs",
            "border-border bg-background font-sans normal-case tracking-normal",
            "hover:bg-midground/5",
            activeCount > 0 && "border-primary/40",
          )}
        >
          <ListFilter
            className={cn(
              "h-3.5 w-3.5 shrink-0",
              activeCount > 0 ? "text-primary" : "text-text-tertiary",
            )}
          />
          <span
            className={cn(
              "flex-1 truncate text-left",
              activeCount > 0 ? "text-foreground" : "text-text-secondary",
            )}
          >
            {filtersLabel}
          </span>
          {activeCount > 0 && (
            <Badge tone="secondary" className="h-4 min-w-4 px-1 text-[0.5625rem] tabular-nums">
              {activeCount}
            </Badge>
          )}
          <ChevronDown
            className={cn(
              "h-3.5 w-3.5 shrink-0 text-text-tertiary transition-transform",
              open && "rotate-180",
            )}
          />
        </button>
      )}

      {open && (
        <div
          role="menu"
          aria-label={sessionFiltersLabel}
          className={cn(
            "absolute top-full z-50 mt-1 border border-border bg-background-base shadow-lg",
            collapsed
              ? "left-1/2 w-64 -translate-x-1/2 lg:left-auto lg:right-0 lg:translate-x-0"
              : "left-0 right-0 w-full min-w-[16rem]",
          )}
        >
          <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
            <span className="text-xs font-medium text-muted-foreground">
              {sessionFiltersLabel}
            </span>
            {activeCount > 0 && (
              <Button
                ghost
                size="xs"
                onClick={clearAll}
                className="shrink-0"
              >
                {t.common.clear}
              </Button>
            )}
          </div>

          <div className="p-1">
            <div className="px-2 pb-1 pt-1.5 text-[0.625rem] font-semibold uppercase tracking-wider text-text-tertiary">
              {filterTypeLabel}
            </div>
            {categoryOptions.map((option) => {
              const selected = category === option.value;
              return (
                <div
                  key={option.value}
                  role="menuitemradio"
                  aria-checked={selected}
                  tabIndex={0}
                  className={cn(
                    "flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-xs",
                    "hover:bg-secondary/40",
                    selected && "text-foreground",
                  )}
                  onClick={() => {
                    setCategory(option.value);
                    setOpen(false);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setCategory(option.value);
                      setOpen(false);
                    }
                  }}
                >
                  {selected ? (
                    <Check className="h-3.5 w-3.5 shrink-0 text-primary" />
                  ) : (
                    <span className="h-3.5 w-3.5 shrink-0" aria-hidden />
                  )}
                  <span>{option.label}</span>
                </div>
              );
            })}

            <div className="my-1 border-t border-border" />

            <div className="px-2 pb-1 pt-1.5 text-[0.625rem] font-semibold uppercase tracking-wider text-text-tertiary">
              {filterSourcesLabel}
            </div>
            <div className="max-h-64 overflow-y-auto p-1">
              {sourceOptions.length === 0 ? (
                <div className="px-2 py-2 text-xs text-muted-foreground">
                  {stats ? t.sessions.noSessions : t.common.loading}
                </div>
              ) : (
                sourceOptions.map(([source, count]) => {
                  const selected = selectedSourceSet.has(source);
                  const SourceIcon = SOURCE_CONFIG[source]?.icon;
                  const sourceColor =
                    SOURCE_CONFIG[source]?.color ?? "text-muted-foreground";
                  return (
                    <div
                      key={source}
                      role="menuitemcheckbox"
                      aria-checked={selected}
                      tabIndex={0}
                      className={cn(
                        "flex cursor-pointer items-center gap-2 rounded px-2 py-1.5",
                        "hover:bg-secondary/40",
                      )}
                      onClick={() => toggleSource(source)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          toggleSource(source);
                        }
                      }}
                    >
                      <Checkbox checked={selected} />
                      {SourceIcon && (
                        <SourceIcon
                          className={cn("h-3.5 w-3.5 shrink-0", sourceColor)}
                        />
                      )}
                      <span className="min-w-0 flex-1 truncate text-xs">
                        {sourceLabel(source)}
                      </span>
                      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                        {count}
                      </span>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface SessionFilterMenuProps {
  collapsed?: boolean;
}
