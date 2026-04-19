import { motion } from "framer-motion";
import type { Variants } from "framer-motion";
import type { LucideIcon } from "lucide-react";

interface SupportingItem {
  icon: LucideIcon;
  title: string;
  copy: string;
}

interface SupportingContentProps {
  items: SupportingItem[];
  prefersReducedMotion: boolean;
  fadeUp: Variants;
}

export function SupportingContent({ items, prefersReducedMotion, fadeUp }: SupportingContentProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {items.map(({ icon: Icon, title, copy }, index) => (
        <motion.article
          key={title}
          className="group rounded-[1.65rem] border border-border/80 bg-[linear-gradient(180deg,rgba(255,230,203,0.08),rgba(6,36,36,0.72))] p-5 shadow-[0_18px_40px_rgba(0,0,0,0.22)] backdrop-blur-sm transition-transform duration-300 hover:-translate-y-0.5"
          initial={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
          animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
          variants={prefersReducedMotion ? undefined : fadeUp}
          transition={{ delay: 0.1 + index * 0.07, duration: 0.35 }}
        >
          <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-full border border-warning/35 bg-warning/10 text-warning shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]">
            <Icon className="h-4 w-4" aria-hidden="true" />
          </div>
          <h2 className="font-expanded text-sm uppercase tracking-[0.16em] text-[#fff5e7]">{title}</h2>
          <p className="mt-3 text-sm leading-7 text-[#ffe2c0]/82">{copy}</p>
        </motion.article>
      ))}
    </div>
  );
}
