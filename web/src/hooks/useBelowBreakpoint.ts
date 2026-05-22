import { useEffect, useState } from "react";

/** True when viewport width is strictly below `px` (matches Tailwind `min-width: px`). */
export function useBelowBreakpoint(px: number) {
  const query = `(max-width: ${px - 1}px)`;
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(query).matches : false,
  );

  useEffect(() => {
    const mql = window.matchMedia(query);
    const sync = () => setMatches(mql.matches);
    sync();
    mql.addEventListener("change", sync);
    return () => mql.removeEventListener("change", sync);
  }, [query]);

  return matches;
}
