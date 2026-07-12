import { createContext } from "react";
import type { ReactNode } from "react";

export interface PageHeaderContextValue {
  setAfterTitle: (node: ReactNode) => void;
  setEnd: (node: ReactNode) => void;
  setTitle: (title: string | null) => void;
}

export const PageHeaderContext = createContext<PageHeaderContextValue | null>(
  null,
);
