import { PluginSlot } from "@/plugins";

/** The `sidebar` slot renders only when `layoutVariant === "cockpit"`
 *  (documented in extending-the-dashboard.md). Extracted so the rule is
 *  testable without mounting the full App shell. */
export function CockpitSidebarSlot({ layoutVariant }: { layoutVariant: string }) {
  return layoutVariant === "cockpit" ? <PluginSlot name="sidebar" /> : null;
}
