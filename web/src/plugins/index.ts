export { exposePluginSDK, getPluginComponent, onPluginRegistered, getRegisteredCount } from "./registry";
export { PluginPage } from "./PluginPage";
export { usePlugins } from "./usePlugins";
export { PluginSlot, KNOWN_SLOT_NAMES, registerSlot, getSlotEntries, onSlotRegistered, unregisterPluginSlots } from "./slots";
export type { KnownSlotName } from "./slots";
export type { PluginManifest, RegisteredPlugin } from "./types";
