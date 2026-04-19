import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";

interface CTAProps {
  primaryLabel: string;
  primaryIcon?: ReactNode;
  primaryAriaLabel: string;
  onPrimaryClick: () => void;
  secondaryLabel: string;
  secondaryAriaLabel: string;
  onSecondaryClick: () => void;
}

export function CTA({
  primaryLabel,
  primaryIcon,
  primaryAriaLabel,
  onPrimaryClick,
  secondaryLabel,
  secondaryAriaLabel,
  onSecondaryClick,
}: CTAProps) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap" aria-label="Primary page actions">
      <Button
        size="lg"
        className="justify-center sm:justify-start"
        aria-label={primaryAriaLabel}
        onClick={onPrimaryClick}
      >
        <span>{primaryLabel}</span>
        {primaryIcon}
      </Button>
      <Button
        variant="outline"
        size="lg"
        className="justify-center sm:justify-start"
        aria-label={secondaryAriaLabel}
        onClick={onSecondaryClick}
      >
        <span>{secondaryLabel}</span>
      </Button>
    </div>
  );
}
