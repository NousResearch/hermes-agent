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
        "inline-flex items-center rounded-lg border border-border bg-muted p-1 text-muted-foreground",
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
              "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors cursor-pointer",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              size === "sm" && "h-7 px-2.5 text-xs",
              size === "md" && "h-8 px-3 text-sm",
              active
                ? "bg-background text-foreground shadow-sm"
                : "hover:bg-background/60 hover:text-foreground",
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
      <span className="text-xs font-medium text-muted-foreground">
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
