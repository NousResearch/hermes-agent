import React from "react";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import routes from "@generated/routes";

/**
 * Whether this page's bundle has a client route for `to` (a docs path without the baseUrl).
 *
 * Catalog listings render from live JSON, but the route table is baked into the bundle, and an edge
 * can serve that bundle from an older deploy. An entry newer than the bundle has no client route,
 * so router navigation would render "Page Not Found" even though the server has the page.
 */
export function useKnownRoute(to: string): boolean {
  return knownPaths().has(useBaseUrl(to));
}

let paths: Set<string> | undefined;

function knownPaths(): Set<string> {
  if (!paths) {
    paths = new Set();
    const walk = (list: typeof routes) =>
      list.forEach((r) => {
        paths!.add(r.path);
        if (r.routes) walk(r.routes as typeof routes);
      });
    walk(routes);
  }
  return paths;
}

/** A router `Link` when the bundle knows the route; otherwise a plain link the server answers. */
export default function RouteLink({ to, ...rest }: React.ComponentProps<typeof Link> & { to: string }) {
  const href = useBaseUrl(to);
  return useKnownRoute(to) ? <Link to={to} {...rest} /> : <a href={href} {...rest} />;
}
