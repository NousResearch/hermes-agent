import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-1 text-[0.72rem] font-medium tracking-[-0.01em] transition-colors",
  {
    variants: {
      variant: {
        default: "bg-foreground text-background",
        secondary: "bg-secondary text-secondary-foreground",
        destructive: "bg-red-50 text-destructive ring-1 ring-red-100",
        outline: "bg-muted text-muted-foreground ring-1 ring-border",
        success: "bg-green-50 text-success ring-1 ring-green-100",
        warning: "bg-amber-50 text-warning ring-1 ring-amber-100",
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
