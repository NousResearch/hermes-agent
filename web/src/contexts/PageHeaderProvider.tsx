import { useLayoutEffect, useMemo, useState, type ReactNode } from "react";
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

  // Clear any per-page title / toolbar slots when the path changes. Child routes
  // re-fill these on mount via usePageHeader.
  /* eslint-disable react-hooks/set-state-in-effect */
  useLayoutEffect(() => {
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
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden">
        <header
          className={cn(
            "z-1 w-full shrink-0",
            "box-border h-14 min-h-14",
            "border-b border-current/20",
            "bg-background-base/40 backdrop-blur-sm",
            "overflow-hidden",
            "sm:min-h-0",
          )}
          role="banner"
        >
          <div
            className={cn(
              "flex h-full w-full min-w-0 flex-1 gap-2 px-3 py-2 sm:gap-3 sm:px-6 sm:py-0",
              isChatRoute
                ? "flex-row items-center"
                : "flex-col justify-center sm:flex-row sm:items-center",
            )}
          >
            <div className="flex min-w-0 flex-1 items-center gap-2 sm:gap-3">
              <h1
                className="font-expanded min-w-0 truncate text-sm font-bold tracking-[0.08em] text-midground"
                style={{ mixBlendMode: "plus-lighter" }}
              >
                {displayTitle}
              </h1>
              {afterTitle}
            </div>

            {end ? (
              <div
                className={cn(
                  "flex min-w-0 justify-end sm:max-w-md sm:flex-1",
                  isChatRoute ? "w-auto shrink-0" : "w-full",
                )}
              >
                {end}
              </div>
            ) : null}
          </div>
        </header>

        <main
          className={cn(
            "min-h-0 w-full min-w-0 flex-1 flex flex-col",
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
