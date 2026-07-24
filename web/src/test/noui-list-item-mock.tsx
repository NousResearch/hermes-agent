import type { HTMLAttributes, ReactNode } from "react";

type ListItemProps = HTMLAttributes<HTMLDivElement> & {
  children?: ReactNode;
  title?: ReactNode;
  subtitle?: ReactNode;
  start?: ReactNode;
  end?: ReactNode;
};

export function ListItem({ children, title, subtitle, start, end, ...props }: ListItemProps) {
  return (
    <div {...props}>
      {start}
      {title ? <div>{title}</div> : null}
      {subtitle ? <div>{subtitle}</div> : null}
      {children}
      {end}
    </div>
  );
}
