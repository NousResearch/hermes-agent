import { useState } from "react";
import { cn } from "@/lib/utils";

export function Tabs({
  defaultValue,
  children,
  className,
}: {
  defaultValue: string;
  children: (active: string, setActive: (v: string) => void) => React.ReactNode;
  className?: string;
}) {
  const [active, setActive] = useState(defaultValue);
  return <div className={cn("flex flex-col gap-4", className)}>{children(active, setActive)}</div>;
}

export function TabsList({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "inline-flex h-9 items-center justify-start border-b border-border text-muted-foreground",
        className,
      )}
      {...props}
    />
  );
}

export function TabsTrigger({
  active,
  value,
  onClick,
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { active: boolean; value: string }) {
  return (
    <button
      type="button"
      className={cn(
        "relative inline-flex items-center justify-center whitespace-nowrap px-3 py-1.5 font-display text-xs tracking-[0.1em] uppercase transition-all cursor-pointer",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        active
          ? "text-foreground after:absolute after:bottom-0 after:left-0 after:right-0 after:h-px after:bg-foreground"
          : "hover:text-foreground",
        className,
      )}
      onClick={onClick}
      {...props}
    />
  );
}
