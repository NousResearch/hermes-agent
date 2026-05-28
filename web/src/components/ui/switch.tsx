import { forwardRef, type ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export const Switch = forwardRef<HTMLButtonElement, SwitchProps>(function Switch(
  { checked, className, disabled, id, onCheckedChange, ...props },
  ref,
) {
  return (
    <button
      aria-checked={checked}
      className={cn(
        "relative inline-flex h-[18px] w-[34px] shrink-0 cursor-pointer items-center rounded-full border-0 p-0 transition-colors",
        checked ? "bg-primary hover:bg-[#245bdb]" : "bg-[#c9cdd4] hover:bg-[#bfc3cb]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      data-hermes-switch
      disabled={disabled}
      id={id}
      onClick={() => onCheckedChange(!checked)}
      ref={ref}
      role="switch"
      type="button"
      {...props}
    >
      <span
        aria-hidden
        className={cn(
          "absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white shadow-[0_1px_3px_rgba(31,35,41,0.2)] transition-[left,box-shadow] duration-150 ease-out",
          checked ? "left-[18px]" : "left-0.5",
        )}
      />
    </button>
  );
});

interface SwitchProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onChange"> {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}
