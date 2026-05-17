import type { HTMLAttributes, ReactNode } from "react";

export function CommandBlock({ children, ...props }: HTMLAttributes<HTMLPreElement> & { children?: ReactNode }) {
  return <pre {...props}>{children}</pre>;
}

export function CopyButton({ children, ...props }: HTMLAttributes<HTMLButtonElement> & { children?: ReactNode }) {
  return <button {...props}>{children ?? "Copy"}</button>;
}
