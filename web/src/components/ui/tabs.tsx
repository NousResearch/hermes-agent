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
        "inline-flex items-center justify-start gap-1 rounded-full bg-black/[0.035] p-1 text-muted-foreground",
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
        "relative inline-flex items-center justify-center whitespace-nowrap rounded-full px-3 py-1.5 text-sm font-medium tracking-[-0.01em] transition-all cursor-pointer",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/25",
        active
          ? "bg-white text-foreground shadow-[0_1px_8px_rgba(0,0,0,0.08)]"
          : "hover:bg-white/60 hover:text-foreground",
        className,
      )}
      onClick={onClick}
      {...props}
    />
  );
}
