import type { ButtonHTMLAttributes, ReactNode } from "react";

export interface ButtonProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "prefix" | "suffix"> {
  destructive?: boolean | null;
  ghost?: boolean | null;
  invert?: boolean | null;
  outlined?: boolean | null;
  size?: "default" | "icon" | "sm" | "xs" | null;
  prefix?: ReactNode;
  suffix?: ReactNode;
}

export declare function Button(props: ButtonProps): ReactNode;
