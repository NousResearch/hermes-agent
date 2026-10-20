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
    /**
     * Optional live badge on the sidebar row. The shell polls `endpoint` and
     * paints a chip while the JSON value at `path` is truthy (an object's
     * `reason` rides the tooltip). The first consumer is Kanban's pause chip.
     */
    badge?: PluginTabBadge;
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

/** `tab.badge` payload: poll `endpoint`, chip while `path` is truthy. */
export interface PluginTabBadge {
  /** Absolute API path, e.g. "/api/plugins/kanban/paused". */
  endpoint: string;
  /** Dotted JSON path whose truthiness means "badge on" (e.g. "paused"). */
  path: string;
  /** Optional dotted path to a number; >1 switches to `labelCountKey`. */
  countPath?: string;
  /** i18n path for the chip text (e.g. "kanban.pausedBadge"); defaults to "Paused". */
  labelKey?: string;
  /** i18n path for the plural chip text; `{count}` is substituted. */
  labelCountKey?: string;
}
