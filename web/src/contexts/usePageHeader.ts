import { useContext } from "react";
import { PageHeaderContext, type PageHeaderContextValue } from "./page-header-context";

export function usePageHeader(): PageHeaderContextValue {
  const ctx = useContext(PageHeaderContext);
  if (!ctx) {
    throw new Error("usePageHeader must be used within a PageHeaderProvider");
  }
  return ctx;
}
