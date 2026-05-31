import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  ghost?: boolean;
  destructive?: boolean;
  size?: string;
  children?: ReactNode;
};

export function Button({ children, ...props }: ButtonProps) {
  const { ghost, destructive, size, ...buttonProps } = props;
  void ghost;
  void destructive;
  void size;
  return <button {...buttonProps}>{children}</button>;
}

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  tone?: string;
  children?: ReactNode;
};

export function Badge({ children, ...props }: BadgeProps) {
  const { tone, ...badgeProps } = props;
  void tone;
  return <span {...badgeProps}>{children}</span>;
}
