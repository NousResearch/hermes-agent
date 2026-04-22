/**
 * Dashboard Plugin SDK + Registry
 *
 * Exposes React, UI components, hooks, and utilities on the window so
 * that plugin bundles can use them without bundling their own copies.
 *
 * Plugins call window.__HERMES_PLUGINS__.register(name, Component)
 * to register their tab component.
 */

import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  useContext,
  createContext,
} from "react";
import { api, fetchJSON } from "@/lib/api";
import { cn, timeAgo, isoTimeAgo } from "@/lib/utils";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectOption } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/i18n";

// ---------------------------------------------------------------------------
// Plugin registry — plugins call register() to add their component.
// ---------------------------------------------------------------------------

type RegistryListener = () => void;

const _registered: Map<string, React.ComponentType> = new Map();
const _listeners: Set<RegistryListener> = new Set();

function _notify() {
  for (const fn of _listeners) {
    try { fn(); } catch { /* ignore */ }
  }
}

/** Register a plugin component. Called by plugin JS bundles. */
function registerPlugin(name: string, component: React.ComponentType) {
  _registered.set(name, component);
  _notify();
}

/** Get a registered component by plugin name. */
export function getPluginComponent(name: string): React.ComponentType | undefined {
  return _registered.get(name);
}

/** Subscribe to registry changes (returns unsubscribe fn). */
export function onPluginRegistered(fn: RegistryListener): () => void {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

/** Get current count of registered plugins. */
export function getRegisteredCount(): number {
  return _registered.size;
}

// ---------------------------------------------------------------------------
// Expose SDK + registry on window
// ---------------------------------------------------------------------------

declare global {
  interface Window {
    __HERMES_PLUGIN_SDK__: unknown;
    __HERMES_PLUGINS__: {
      register: typeof registerPlugin;
    };
  }
}

export function exposePluginSDK() {
  window.__HERMES_PLUGINS__ = {
    register: registerPlugin,
  };

  window.__HERMES_PLUGIN_SDK__ = {
    // React core — plugins use these instead of importing react
    React,
    hooks: {
      useState,
      useEffect,
      useCallback,
      useMemo,
      useRef,
      useContext,
      createContext,
    },

    // Hermes API client
    api,
    // Raw fetchJSON for plugin-specific endpoints
    fetchJSON,

    // UI components (shadcn/ui primitives)
    components: {
      Card,
      CardHeader,
      CardTitle,
      CardContent,
      Badge,
      Button,
      Input,
      Label,
      Select,
      SelectOption,
      Separator,
      Tabs,
      TabsList,
      TabsTrigger,
    },

    // Utilities
    utils: { cn, timeAgo, isoTimeAgo },

    // Hooks
    useI18n,
  };
}
