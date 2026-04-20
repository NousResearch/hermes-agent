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
  };
  entry: string;
  css?: string | null;
  has_api: boolean;
  source: string;
}

export interface RegisteredPlugin {
  manifest: PluginManifest;
  component: React.ComponentType;
}
