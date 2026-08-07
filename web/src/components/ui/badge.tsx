import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center border px-2 py-0.5 font-compressed text-[0.65rem] tracking-[0.15em] uppercase transition-colors",
  {
    variants: {
      variant: {
        default: "border-foreground/20 bg-foreground/10 text-foreground",
        secondary: "border-border bg-secondary text-secondary-foreground",
        destructive: "border-destructive/30 bg-destructive/15 text-destructive",
        outline: "border-border text-muted-foreground",
        success: "grain border-emerald-600/30 bg-emerald-950/70 text-emerald-400",
        warning: "border-warning/30 bg-warning/15 text-warning",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export function Badge({
  className,
  variant,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeVariants>) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
