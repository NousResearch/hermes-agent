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
  const isDocsRoute = pathname === "/documentation" || pathname === "/documentation/";

  const value = useMemo(
    () => ({ setAfterTitle, setEnd, setTitle: setTitleOverride }),
    [],
  );

  return (
    <PageHeaderContext.Provider value={value}>
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden">
        {!isDocsRoute && (
          <header
            className={cn(
              "z-10 w-full shrink-0 border-b border-border bg-background/65 backdrop-blur",
              isChatRoute ? "h-12" : "min-h-14",
            )}
            role="banner"
          >
            <div
              className={cn(
                "flex h-full w-full min-w-0 gap-2 px-3 py-2 sm:px-4",
                isChatRoute ? "items-center" : "flex-col justify-center sm:flex-row sm:items-center",
              )}
            >
              <div className="flex min-w-0 flex-1 items-center gap-2 sm:gap-3">
                <h1 className="min-w-0 truncate text-sm font-semibold tracking-tight text-foreground sm:text-base">
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
        )}

        <main
          className={cn(
            "min-h-0 w-full min-w-0 flex-1 flex flex-col",
            isChatRoute || isDocsRoute
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
