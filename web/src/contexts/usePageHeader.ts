import { useContext } from "react";
import { PageHeaderContext, type PageHeaderContextValue } from "./page-header-context";

/**
 * Fill header slots from `useLayoutEffect` and clear them in its cleanup. A
 * `useEffect` cleanup on the outgoing page runs after the incoming page's
 * layout effects, so it would erase the new page's header.
 */
export function usePageHeader(): PageHeaderContextValue {
  const ctx = useContext(PageHeaderContext);
  if (!ctx) {
    throw new Error("usePageHeader must be used within a PageHeaderProvider");
  }
  return ctx;
}
