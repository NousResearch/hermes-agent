import React, { useEffect } from "react";
import { installEmbeddedHubLinks } from "../../lib/embedded-hub-links";

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps) {
  useEffect(() => installEmbeddedHubLinks(window), []);
  return <>{children}</>;
}
