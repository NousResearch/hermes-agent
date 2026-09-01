export type DashboardSidebarMode = "expanded" | "collapsed" | "chat-rail";

function normalizeDashboardPath(pathname: string): string {
  return pathname.replace(/\/$/, "") || "/";
}

/** Resolve the first workspace shown by the dashboard root route. */
export function getDashboardHomePath(embeddedChatEnabled: boolean): "/chat" | "/sessions" {
  return embeddedChatEnabled ? "/chat" : "/sessions";
}

/** The native chat owns its own toolbar; other routes use the page header host. */
export function shouldRenderDashboardPageHeader(pathname: string): boolean {
  return normalizeDashboardPath(pathname) !== "/chat";
}

/** The global mobile nav header is replaced by the native chat header on /chat. */
export function shouldRenderMobileNavigationHeader(pathname: string, isMobile: boolean): boolean {
  return isMobile && shouldRenderDashboardPageHeader(pathname);
}

/** Keep the session navigator visible on desktop, but opt-in on compact mobile. */
export function shouldShowMobileSessionNavigator(isMobile: boolean, mobileOpen: boolean): boolean {
  return !isMobile || mobileOpen;
}

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
  const normalizedPath = normalizeDashboardPath(pathname);
  if (isMobile) return "expanded";
  if (normalizedPath === "/chat") {
    return chatSidebarExpanded ? "expanded" : "chat-rail";
  }
  return collapsed ? "collapsed" : "expanded";
}
