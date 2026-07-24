import type { SelectHTMLAttributes, ReactNode } from "react";

type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & { children?: ReactNode };
type SelectOptionProps = { value: string; children?: ReactNode; disabled?: boolean };

export function Select({ children, ...props }: SelectProps) {
  return <select {...props}>{children}</select>;
}

export function SelectOption({ value, children, disabled }: SelectOptionProps) {
  return <option value={value} disabled={disabled}>{children}</option>;
}
