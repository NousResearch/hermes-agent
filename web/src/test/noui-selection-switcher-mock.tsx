import type { HTMLAttributes, ReactNode } from "react";

export function SelectionSwitcher({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}
