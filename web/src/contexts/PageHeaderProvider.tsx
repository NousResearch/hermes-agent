import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { PageHeaderContext } from "./page-header-context";
import { resolvePageTitle } from "@/lib/resolve-page-title";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";

export function PageHeaderProvider({
  children,
  pluginTabs,
}: {
  children: ReactNode;
  pluginTabs: { path: string; label: string }[];
}) {
  const { pathname } = useLocation();
  const { t } = useI18n();
  const [titleOverride, setTitleOverride] = useState<string | null>(null);
  const [afterTitle, setAfterTitle] = useState<ReactNode>(null);
  const [end, setEnd] = useState<ReactNode>(null);
  const prevPathnameRef = useRef(pathname);

  // Clear per-page slots on navigation only (not the initial mount).
  // Toolbar actions must register via useEffect so they run after this
  // layout-phase reset — see SessionsPage / CronPage.
  /* eslint-disable react-hooks/set-state-in-effect */
  useLayoutEffect(() => {
    if (prevPathnameRef.current === pathname) return;
    prevPathnameRef.current = pathname;
    setTitleOverride(null);
    setAfterTitle(null);
    setEnd(null);
  }, [pathname]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const defaultTitle = useMemo(
    () => resolvePageTitle(pathname, t, pluginTabs),
    [pathname, t, pluginTabs],
  );
  const displayTitle = titleOverride ?? defaultTitle;

  const isChatRoute = pathname === "/chat" || pathname === "/chat/";
  const isEnvRoute =
    pathname === "/env" || pathname.startsWith("/env/");

  const value = useMemo(
    () => ({
      setAfterTitle,
      setEnd,
      setTitle: setTitleOverride,
    }),
    [],
  );

  return (
    <PageHeaderContext.Provider value={value}>
      {/*
        CSS grid: row 1 = page chrome (fixed height), row 2 = scrollable
        outlet. Grid avoids flex min-height bugs where main content paints
        over the header band on narrow viewports.
      */}
      <div className="grid min-h-0 w-full min-w-0 flex-1 grid-rows-[auto_minmax(0,1fr)] overflow-hidden">
        <header
          className={cn(
            "relative z-10 col-span-full w-full isolate",
            "box-border border-b border-current/20",
            "min-h-14 px-3 py-2 sm:px-6 sm:py-0",
          )}
          style={{ backgroundColor: "var(--background-base)" }}
          role="banner"
        >
          <div
            className={cn(
              "flex w-full min-w-0 gap-2 sm:min-h-14 sm:gap-3",
              isChatRoute
                ? "min-h-10 flex-row items-center"
                : isEnvRoute
                  ? "flex-col justify-center gap-2 sm:flex-row sm:items-center"
                  : "min-h-10 flex-row items-center justify-between",
            )}
          >
            <div
              className={cn(
                "flex min-w-0 gap-2 sm:gap-3",
                afterTitle && isEnvRoute
                  ? "flex-1 flex-col items-start sm:flex-row sm:items-center"
                  : afterTitle
                    ? "min-w-0 flex-1 flex-row flex-wrap items-center"
                    : "min-w-0 shrink items-center",
              )}
            >
              <h1
                className={cn(
                  "font-expanded min-w-0 text-sm font-bold tracking-[0.08em] text-midground",
                  afterTitle && isEnvRoute
                    ? "max-w-full sm:min-w-0 sm:shrink sm:truncate"
                    : afterTitle
                      ? "shrink truncate"
                      : "truncate",
                )}
                style={{ mixBlendMode: "plus-lighter" }}
              >
                {displayTitle}
              </h1>
              {afterTitle ? (
                <div
                  className={cn(
                    "min-w-0 scrollbar-none",
                    isEnvRoute
                      ? "w-full overflow-x-auto sm:flex-1 sm:overflow-x-auto"
                      : "shrink-0 overflow-visible",
                  )}
                >
                  {afterTitle}
                </div>
              ) : null}
            </div>

            {end ? (
              <div
                className={cn(
                  "flex shrink-0 items-center",
                  isChatRoute
                    ? "justify-end"
                    : isEnvRoute
                      ? "w-full justify-start sm:w-auto sm:justify-end"
                      : "justify-end",
                )}
              >
                {end}
              </div>
            ) : null}
          </div>
        </header>

        <main
          className={cn(
            "relative z-0 col-span-full flex min-h-0 w-full min-w-0 flex-col",
            isChatRoute
              ? "overflow-hidden"
              : "overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable]",
          )}
        >
          {children}
        </main>
      </div>
    </PageHeaderContext.Provider>
  );
}
