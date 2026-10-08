import React from "react";

// The catalog is exercised without Docusaurus's unrelated navigation shell.
export default function Layout({ children }: React.PropsWithChildren) {
  return <>{children}</>;
}
