/** Types for the dashboard plugin system. */

import type { ComponentType } from "react";

export interface PluginManifest {
  name: string;
  label: string;
  description: string;
  icon: string;
  version: string;
  tab: {
    path: string;
    /** "end", "after:<pathSegment>", "before:<pathSegment>" (e.g. "after:skills" → after `/skills`) */
    position?: string;
    /** When set to a built-in route path, this plugin replaces that page instead of adding a new tab. */
    override?: string;
    /** When true, the plugin may register without a sidebar tab (slot-only, etc.). */
    hidden?: boolean;
  };
  /** Declared for discovery; actual slots use registerSlot in the plugin bundle. */
  slots?: string[];
  entry: string;
  css?: string | null;
  has_api: boolean;
  /**
   * Optional Subresource Integrity hash (e.g. "sha384-..."). When set,
   * the browser will refuse to execute the plugin bundle if its hash
   * does not match. This protects against tampered plugin delivery.
   */
  integrity?: string;
  source: string;
}

export interface RegisteredPlugin {
  manifest: PluginManifest;
  component: ComponentType;
}
