import type { HTMLAttributes, ReactNode } from "react";

export function Stats({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}

export function Stat({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}
