import { cn } from "@/lib/utils";

/**
 * Shadcn-style card primitive. Themes can still restyle cards through the
 * optional component-style CSS variables, but the default typography and
 * contrast stay plain Geist/system UI instead of the old Hermes display font.
 */
const CARD_STYLE: React.CSSProperties = {
  clipPath: "var(--component-card-clip-path)",
  borderImage: "var(--component-card-border-image)",
  background: "var(--component-card-background)",
  boxShadow: "var(--component-card-box-shadow)",
};

export function Card({ className, style, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "w-full rounded-xl border border-border bg-card text-card-foreground shadow-sm",
        className,
      )}
      style={{ ...CARD_STYLE, ...style }}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col gap-1.5 p-4", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-base font-semibold leading-none tracking-tight text-card-foreground", className)} {...props} />;
}

export function CardDescription({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-sm text-muted-foreground", className)} {...props} />;
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-4 pt-0", className)} {...props} />;
}
