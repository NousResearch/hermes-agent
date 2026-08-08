import type { HTMLAttributes } from "react";

import buzzLogoUrl from "@/assets/BuzzLogo24px.svg";

/** Square Buzz line mark, colored like the surrounding navigation text. */
export function BuzzIcon({
  style,
  ...props
}: HTMLAttributes<HTMLSpanElement>) {
  const maskImage = `url(${buzzLogoUrl})`;

  return (
    <span
      aria-hidden="true"
      style={{
        WebkitMaskImage: maskImage,
        maskImage,
        WebkitMaskPosition: "center",
        maskPosition: "center",
        WebkitMaskRepeat: "no-repeat",
        maskRepeat: "no-repeat",
        WebkitMaskSize: "contain",
        maskSize: "contain",
        backgroundColor: "currentColor",
        display: "inline-block",
        ...style,
      }}
      {...props}
    />
  );
}
