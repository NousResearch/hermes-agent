import { cva } from "class-variance-authority";
import { cloneElement } from "react";
import { cn } from "@/lib/utils";

const SHADOW_DEFAULT =
  "shadow-[inset_-1px_-1px_0_0_#00000080,inset_1px_1px_0_0_#ffffff80]";
const SHADOW_INVERT =
  "shadow-[inset_-1px_-1px_0_0_#00000080,inset_1px_1px_0_0_#ffffff29]";
const SHADOW_INVERT_OUTLINED =
  "shadow-[inset_-1px_-1px_0_0_#ffffff12,inset_1px_1px_0_0_#ffffff29]";
const ACTIVE_FILTER =
  "active:[filter:invert(1)_brightness(calc(100-99*var(--foreground-alpha,0)))]";

const buttonVariants = cva(
  [
    "group relative grid min-w-0 cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center justify-center",
    "font-bold uppercase leading-none",
    "disabled:pointer-events-none disabled:bg-midground/15 disabled:text-midground disabled:shadow-none",
  ],
  {
    compoundVariants: [
      {
        class: `bg-midground text-background-base active:invert ${SHADOW_DEFAULT}`,
        destructive: false,
        ghost: false,
        invert: false,
        outlined: false,
      },
      {
        class: `bg-midground/15 text-midground ${SHADOW_INVERT} ${ACTIVE_FILTER}`,
        destructive: false,
        ghost: false,
        invert: true,
        outlined: false,
      },
      {
        class: `shadow-midground ${SHADOW_DEFAULT} ${ACTIVE_FILTER}`,
        destructive: false,
        ghost: false,
        invert: false,
        outlined: true,
      },
      {
        class: `${SHADOW_INVERT_OUTLINED} ${ACTIVE_FILTER}`,
        destructive: false,
        ghost: false,
        invert: true,
        outlined: true,
      },
      {
        class: "bg-transparent text-current hover:bg-midground/10 shadow-none",
        destructive: false,
        ghost: true,
      },
      {
        class: "bg-transparent text-destructive hover:bg-destructive/10 shadow-none",
        destructive: true,
        ghost: true,
      },
      {
        class: `bg-destructive text-destructive-foreground hover:bg-destructive/90 ${SHADOW_INVERT}`,
        destructive: true,
        ghost: false,
        outlined: false,
      },
      {
        class:
          "border border-destructive/40 bg-transparent text-destructive hover:bg-destructive/10 shadow-none",
        destructive: true,
        ghost: false,
        outlined: true,
      },
    ],
    defaultVariants: {
      destructive: false,
      ghost: false,
      invert: false,
      outlined: false,
      size: "default",
    },
    variants: {
      destructive: { true: "" },
      ghost: { true: "" },
      invert: { true: "" },
      outlined: { true: "text-midground bg-transparent" },
      size: {
        default: "min-h-10 px-3 py-2",
        icon: "size-9 grid-cols-1 place-items-center p-2 [&>svg]:size-3.5",
        sm: "min-h-8 px-2.5 py-1.5 text-[0.7rem] [&>svg]:size-3",
        xs: "size-7 grid-cols-1 place-items-center p-1 [&>svg]:size-3",
      },
    },
  },
);

function IconSlot({ icon, side }) {
  return (
    <>
      <span className="w-5" />
      <span
        className={cn(
          "absolute top-1/2 -translate-y-1/2",
          side === "left" ? "left-3" : "right-3",
        )}
      >
        {typeof icon === "object"
          ? cloneElement(icon, { className: cn("size-3.5", icon.props?.className) })
          : icon}
      </span>
    </>
  );
}

export function Button({
  children = null,
  className = undefined,
  destructive = false,
  ghost = false,
  invert = false,
  outlined = false,
  prefix = null,
  size = undefined,
  suffix = null,
  ...props
}) {
  return (
    <button
      className={cn(
        "hermes-button font-mono",
        buttonVariants({ destructive, ghost, invert, outlined, size }),
        className,
      )}
      {...props}
    >
      {!ghost && (
        <span
          aria-hidden
          className="arc-border opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100 group-active:opacity-100"
        />
      )}
      {prefix && <IconSlot icon={prefix} side="left" />}
      <span className="hermes-button__label">{children}</span>
      {suffix && <IconSlot icon={suffix} side="right" />}
    </button>
  );
}
