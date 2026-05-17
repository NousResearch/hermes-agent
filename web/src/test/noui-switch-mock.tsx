import type { ButtonHTMLAttributes } from "react";

type SwitchProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onChange"> & {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
};

export function Switch({ checked = false, onCheckedChange, ...props }: SwitchProps) {
  return <button role="switch" aria-checked={checked} onClick={() => onCheckedChange?.(!checked)} {...props} />;
}
