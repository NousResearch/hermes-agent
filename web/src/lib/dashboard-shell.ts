export type DashboardSidebarMode = "expanded" | "collapsed" | "chat-rail";

/**
 * Resolve the shell's desktop sidebar presentation without coupling the route
 * rule to React. Chat is the primary workspace, so it starts with an icon rail
 * on desktop; normal routes continue to honor the user's stored preference.
 */
export function getDashboardSidebarMode(
  pathname: string,
  isMobile: boolean,
  collapsed: boolean,
  chatSidebarExpanded = false,
): DashboardSidebarMode {
  const normalizedPath = pathname.replace(/\/$/, "") || "/";
  if (isMobile) return "expanded";
  if (normalizedPath === "/chat") {
    return chatSidebarExpanded ? "expanded" : "chat-rail";
  }
  return collapsed ? "collapsed" : "expanded";
}
