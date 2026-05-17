import type { HTMLAttributes, ReactNode } from "react";

export function FilterGroup({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}

export function Segmented({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}
