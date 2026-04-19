import { motion } from "framer-motion";
import type { ReactNode } from "react";
import type { Variants } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { CTA } from "@/components/landing/CTA";

interface HeroProps {
  eyebrow: string;
  title: string;
  description: string;
  badgeLabel: string;
  statusLabel: string;
  statusVariant: "success" | "warning" | "destructive" | "outline";
  prefersReducedMotion: boolean;
  fadeUp: Variants;
  cta: {
    primaryLabel: string;
    primaryIcon?: ReactNode;
    primaryAriaLabel: string;
    onPrimaryClick: () => void;
    secondaryLabel: string;
    secondaryAriaLabel: string;
    onSecondaryClick: () => void;
  };
}

export function Hero({
  eyebrow,
  title,
  description,
  badgeLabel,
  statusLabel,
  statusVariant,
  prefersReducedMotion,
  fadeUp,
  cta,
}: HeroProps) {
  return (
    <motion.div
      className="relative z-10 flex flex-col gap-7 px-5 py-7 sm:px-8 sm:py-9 lg:px-10 lg:py-12"
      initial={prefersReducedMotion ? undefined : "hidden"}
      animate={prefersReducedMotion ? undefined : "visible"}
      variants={fadeUp}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="warning" className="gap-1 border-warning/40 bg-warning/15 text-warning">
          {badgeLabel}
        </Badge>
        <Badge variant={statusVariant} className="gap-1.5">
          {statusVariant === "success" && <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />}
          {statusLabel}
        </Badge>
      </div>

      <div className="max-w-3xl space-y-5">
        <p className="font-compressed text-[0.72rem] uppercase tracking-[0.34em] text-[#fff2df]/78 sm:text-[0.8rem]">
          {eyebrow}
        </p>
        <h1
          id="status-hero"
          className="max-w-3xl text-balance font-expanded text-[2.9rem] leading-[0.88] tracking-[0.025em] text-[#fff7ec] drop-shadow-[0_16px_42px_rgba(0,0,0,0.42)] sm:text-[4rem] lg:text-[5.35rem]"
        >
          {title}
        </h1>
        <p className="max-w-2xl text-base leading-8 text-[#ffe9cf]/90 sm:text-[1.04rem] sm:leading-9">
          {description}
        </p>
      </div>

      <CTA {...cta} />
    </motion.div>
  );
}
