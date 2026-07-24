import type { HTMLAttributes } from "react";

export function Spinner(props: HTMLAttributes<HTMLDivElement>) {
  return <div role="status" {...props} />;
}
