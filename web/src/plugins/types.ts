/** Types for the dashboard plugin system. */

export interface PluginManifest {
  name: string;
  label: string;
  description: string;
  icon: string;
  version: string;
  tab: {
    path: string;
    position: string;  // "end", "after:<tab>", "before:<tab>"
    /** When set to a built-in route path (e.g. `"/"`, `"/sessions"`), this
     *  plugin's component replaces the built-in page at that route rather
     *  than adding a new tab. Useful for themes that want a custom home
     *  page without losing the rest of the dashboard. */
    override?: string;
    /** When true, the plugin registers its component and slot contributors
     *  without adding a tab to the nav. Used by slot-only plugins (e.g. a
     *  plugin that just injects a header crest). */
    hidden?: boolean;
  };
  /** Named shell slots this plugin populates. Mirrored by the backend's
   *  manifest discovery; used purely as a documentation/discovery aid —
   *  actual slot registration happens when the plugin's JS bundle calls
   *  `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)`. */
  slots?: string[];
  entry: string;
  css?: string | null;
  has_api: boolean;
  source: string;
}

export interface RegisteredPlugin {
  manifest: PluginManifest;
  component: React.ComponentType;
}
