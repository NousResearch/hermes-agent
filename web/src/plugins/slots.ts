/**
 * Plugin slot registry.
 *
 * Plugins can inject components into named locations in the app shell
 * (header-left, sidebar, backdrop, etc.) by calling
 * `window.__HERMES_PLUGINS__.registerSlot(pluginName, slotName, Component)`
 * from their JS bundle. Multiple plugins can populate the same slot — they
 * render stacked in registration order.
 *
 * The canonical slot names are documented in `KNOWN_SLOT_NAMES` below. The
 * registry accepts any string so plugin ecosystems can define their own
 * slots; the shell only renders `<PluginSlot name="..." />` for the slots
 * it knows about.
 */

import React, { Fragment, useEffect, useState } from "react";

type SlotListener = () => void;

interface SlotEntry {
  plugin: string;
  component: React.ComponentType;
}

/** Map<slotName, SlotEntry[]>. Entries are appended in registration order. */
const _slotRegistry: Map<string, SlotEntry[]> = new Map();
const _slotListeners: Set<SlotListener> = new Set();

function _notifySlots() {
  for (const fn of _slotListeners) {
    try {
      fn();
    } catch {
      /* ignore */
    }
  }
}

/** Register a component for a slot. Called by plugin bundles via
 *  `window.__HERMES_PLUGINS__.registerSlot(...)`.
 *
 *  If the same (plugin, slot) pair is registered twice, the later call
 *  replaces the earlier one — this matches how React HMR expects plugin
 *  re-mounts to behave. */
export function registerSlot(
  plugin: string,
  slot: string,
  component: React.ComponentType,
): void {
  const existing = _slotRegistry.get(slot) ?? [];
  const filtered = existing.filter((e) => e.plugin !== plugin);
  filtered.push({ plugin, component });
  _slotRegistry.set(slot, filtered);
  _notifySlots();
}

/** Read current entries for a slot. Returns a copy so callers can't mutate
 *  registry state. */
function getSlotEntries(slot: string): SlotEntry[] {
  return (_slotRegistry.get(slot) ?? []).slice();
}

/** Subscribe to registry changes. Returns an unsubscribe function. */
function onSlotRegistered(fn: SlotListener): () => void {
  _slotListeners.add(fn);
  return () => {
    _slotListeners.delete(fn);
  };
}

interface PluginSlotProps {
  /** Slot identifier (e.g. `"sidebar"`, `"header-left"`). */
  name: string;
  /** Optional content rendered when no plugins have claimed the slot.
   *  Useful for built-in defaults the plugin would replace. */
  fallback?: React.ReactNode;
}

/** Render all components registered for a given slot, stacked in order.
 *
 *  Component re-renders when the slot registry changes so plugins that
 *  arrive after initial mount show up without a manual refresh. */
export function PluginSlot({ name, fallback }: PluginSlotProps) {
  const [entries, setEntries] = useState<SlotEntry[]>(() => getSlotEntries(name));

  useEffect(() => {
    // Pick up anything registered between the initial `useState` call
    // and the first effect tick, then subscribe for future changes.
    setEntries(getSlotEntries(name));
    const unsub = onSlotRegistered(() => setEntries(getSlotEntries(name)));
    return unsub;
  }, [name]);

  if (entries.length === 0) {
    return fallback ? React.createElement(Fragment, null, fallback) : null;
  }

  return React.createElement(
    Fragment,
    null,
    ...entries.map((entry) =>
      React.createElement(entry.component, { key: entry.plugin }),
    ),
  );
}
