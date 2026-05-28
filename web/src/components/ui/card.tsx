import { cn, themedBody } from "@/lib/utils";

/**
 * Themed card primitive. Themes can restyle every card without touching
 * call sites by setting CSS vars under the `card` component-style bucket:
 *
 *   componentStyles:
 *     card:
 *       clipPath: "polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px)"
 *       border: "1px solid var(--color-ring)"
 *       background: "linear-gradient(180deg, var(--color-card) 0%, transparent 100%)"
 *       boxShadow: "0 0 0 1px var(--color-ring) inset, 0 0 24px -8px var(--warm-glow)"
 *
 * All properties are optional — vars that aren't set compute to their
 * CSS initial value, so the default shadcn-y card keeps looking normal
 * for themes that don't override anything.
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
        "w-full rounded-lg border border-border bg-card text-card-foreground shadow-[0_1px_2px_rgba(31,35,41,0.04)]",
        themedBody,
        className,
      )}
      style={{ ...CARD_STYLE, ...style }}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col gap-1.5 border-b border-border px-5 py-4", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        "font-sans text-[15px] font-semibold tracking-normal text-text-primary",
        className,
      )}
      {...props}
    />
  );
}

export function CardDescription({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn("font-sans normal-case text-xs text-muted-foreground", className)} {...props} />
  );
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}
