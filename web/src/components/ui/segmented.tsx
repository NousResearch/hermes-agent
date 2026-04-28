import { cn } from "@/lib/utils";

export function Segmented<T extends string>({
  className,
  onChange,
  options,
  size = "sm",
  value,
}: SegmentedProps<T>) {
  return (
    <div
      role="radiogroup"
      className={cn(
        "inline-flex border border-border bg-background/30",
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;

        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "font-mondwest tracking-[0.1em] uppercase",
              "transition-colors cursor-pointer whitespace-nowrap",
              "border-r border-border last:border-r-0",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30",
              size === "sm" && "h-7 px-2.5 text-[0.65rem]",
              size === "md" && "h-8 px-3 text-xs",
              active
                ? "bg-foreground/90 text-background"
                : "text-muted-foreground hover:bg-foreground/10 hover:text-foreground",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export function FilterGroup({
  children,
  className,
  label,
}: FilterGroupProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className="font-mondwest text-[0.65rem] tracking-[0.12em] uppercase text-muted-foreground/70">
        {label}
      </span>
      {children}
    </div>
  );
}

interface FilterGroupProps {
  children: React.ReactNode;
  className?: string;
  label: string;
}

interface SegmentedOption<T extends string> {
  label: string;
  value: T;
}

interface SegmentedProps<T extends string> {
  className?: string;
  onChange: (value: T) => void;
  options: SegmentedOption<T>[];
  size?: "sm" | "md";
  value: T;
}
