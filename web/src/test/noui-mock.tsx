import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  ghost?: boolean;
  destructive?: boolean;
  size?: string;
  children?: ReactNode;
};

export function Button({ ghost: _ghost, destructive: _destructive, size: _size, children, ...props }: ButtonProps) {
  return <button {...props}>{children}</button>;
}

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  tone?: string;
  children?: ReactNode;
};

export function Badge({ tone: _tone, children, ...props }: BadgeProps) {
  return <span {...props}>{children}</span>;
}
