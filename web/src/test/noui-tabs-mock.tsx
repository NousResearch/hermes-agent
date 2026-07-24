import type { HTMLAttributes, ReactNode } from "react";

export function Tabs({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}
export function TabsList({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}
export function TabsTrigger({ children, ...props }: HTMLAttributes<HTMLButtonElement> & { children?: ReactNode }) {
  return <button {...props}>{children}</button>;
}
export function TabsContent({ children, ...props }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return <div {...props}>{children}</div>;
}
