import { forwardRef, type ElementType, type HTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";

type TypographyProps = HTMLAttributes<HTMLElement> & {
  as?: ElementType;
  children?: ReactNode;
  compressed?: boolean;
  courier?: boolean;
  expanded?: boolean;
  mondwest?: boolean;
  mono?: boolean;
  sans?: boolean;
  variant?: "sm" | "md" | "lg" | "xl";
};

const variantClasses: Record<NonNullable<TypographyProps["variant"]>, string> = {
  sm: "text-sm leading-[1.5] tracking-normal",
  md: "text-xl leading-[1.25] tracking-normal",
  lg: "text-2xl leading-[1.2] tracking-normal",
  xl: "text-3xl leading-[1.15] tracking-normal",
};

export const Typography = forwardRef<HTMLElement, TypographyProps>(function Typography(
  {
    as: Component = "span",
    className,
    compressed,
    courier,
    expanded,
    mondwest,
    mono,
    sans,
    variant,
    ...props
  },
  ref,
) {
  const hasFontVariant = compressed || courier || expanded || mondwest || mono || sans;

  return (
    <Component
      className={cn(
        compressed && "font-compressed",
        courier && "font-courier",
        expanded && "font-expanded",
        mondwest && "font-mondwest tracking-normal",
        mono && "font-mono",
        (!hasFontVariant || sans) && "font-sans",
        variant && variantClasses[variant],
        className,
      )}
      ref={ref}
      {...props}
    />
  );
});

export const H2 = forwardRef<HTMLHeadingElement, Omit<TypographyProps, "as">>(function H2(
  { className, variant = "lg", ...props },
  ref,
) {
  return <Typography as="h2" className={cn("font-bold", className)} variant={variant} ref={ref} {...props} />;
});
