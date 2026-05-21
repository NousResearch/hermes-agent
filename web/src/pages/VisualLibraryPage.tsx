import { useLayoutEffect, type ComponentType } from "react";
import {
  Boxes,
  Check,
  Component,
  FileCode2,
  Image,
  Layers,
  Palette,
  SwatchBook,
  Type,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Typography } from "@/components/NouiTypography";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";
import { BUILTIN_THEMES, useTheme } from "@/themes";
import type { DashboardTheme, ThemeLayer } from "@/themes";

const ASSET_SLOTS = [
  {
    name: "bg",
    varName: "--theme-asset-bg",
    purpose: "Full-viewport dashboard backdrop consumed by the shell.",
  },
  {
    name: "hero",
    varName: "--theme-asset-hero",
    purpose: "Primary character, machine, or product render for plugin panels.",
  },
  {
    name: "logo",
    varName: "--theme-asset-logo",
    purpose: "Header or sidebar brand mark for themed installs.",
  },
  {
    name: "crest",
    varName: "--theme-asset-crest",
    purpose: "Faction, profile, or workspace emblem.",
  },
  {
    name: "sidebar",
    varName: "--theme-asset-sidebar",
    purpose: "Secondary art slot for dashboard rails and inspectors.",
  },
  {
    name: "header",
    varName: "--theme-asset-header",
    purpose: "Alternate top-chrome artwork.",
  },
] as const;

const COMPONENT_BUCKETS = [
  "card",
  "header",
  "footer",
  "sidebar",
  "tab",
  "progress",
  "badge",
  "backdrop",
  "page",
] as const;

const SOURCE_POINTERS = [
  {
    label: "Dashboard themes",
    path: "web/src/themes/presets.ts",
    detail: "Built-in palette, typography, density, and layout definitions.",
  },
  {
    label: "Theme runtime",
    path: "web/src/themes/context.tsx",
    detail: "Applies CSS variables, assets, component styles, and user themes.",
  },
  {
    label: "CLI skins",
    path: "hermes_cli/skin_engine.py",
    detail: "Terminal banner, spinner, prompt, and response-box skin system.",
  },
  {
    label: "Dashboard shell",
    path: "web/src/App.tsx",
    detail: "Route, sidebar, plugin slots, and page host wiring.",
  },
] as const;

const QUALITY_GATES = [
  "Theme names stay in sync with the backend built-in list.",
  "User assets route through theme asset variables instead of page-specific globals.",
  "Plugin visual surfaces complement the dashboard shell instead of replacing chat.",
  "New dependencies keep upper bounds and lockfile changes when required.",
] as const;

function layerToCss(layer: ThemeLayer): string {
  return `color-mix(in srgb, ${layer.hex} ${Math.round(layer.alpha * 100)}%, transparent)`;
}

function ThemeSwatch({ theme }: { theme: DashboardTheme }) {
  return (
    <div className="grid h-16 grid-cols-[1.2fr_1fr_1fr] overflow-hidden border border-current/15">
      <div
        className="relative"
        style={{ background: theme.palette.background.hex }}
      >
        <span className="absolute bottom-2 left-2 text-[0.6rem] font-medium uppercase text-white/55">
          bg
        </span>
      </div>
      <div
        className="relative"
        style={{ background: theme.palette.midground.hex }}
      >
        <span className="absolute bottom-2 left-2 text-[0.6rem] font-medium uppercase text-black/55">
          mid
        </span>
      </div>
      <div className="relative" style={{ background: theme.palette.warmGlow }}>
        <span className="absolute bottom-2 left-2 text-[0.6rem] font-medium uppercase text-white/70">
          glow
        </span>
      </div>
    </div>
  );
}

function ThemeCard({
  active,
  onSelect,
  theme,
}: {
  active: boolean;
  onSelect: () => void;
  theme: DashboardTheme;
}) {
  return (
    <Card
      className={cn(
        "overflow-hidden transition-colors",
        active && "border-primary bg-primary/5",
      )}
    >
      <ThemeSwatch theme={theme} />
      <CardContent className="flex flex-col gap-4">
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <Typography
              as="h3"
              mondwest
              className="truncate text-[0.75rem] uppercase tracking-[0.16em]"
            >
              {theme.label}
            </Typography>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              {theme.description}
            </p>
          </div>
          {active ? (
            <Badge className="shrink-0 gap-1">
              <Check className="h-3 w-3" />
              Active
            </Badge>
          ) : null}
        </div>

        <div className="grid gap-2 text-[0.68rem] text-muted-foreground">
          <TokenLine label="Background" value={theme.palette.background.hex} />
          <TokenLine label="Midground" value={theme.palette.midground.hex} />
          <TokenLine label="Radius" value={theme.layout.radius} />
          <TokenLine label="Density" value={theme.layout.density} />
        </div>

        <Button
          className="w-fit"
          disabled={active}
          onClick={onSelect}
          outlined={!active}
          size="sm"
        >
          {active ? "Applied" : "Apply theme"}
        </Button>
      </CardContent>
    </Card>
  );
}

function TokenLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3">
      <span className="uppercase tracking-[0.12em] text-muted-foreground/75">
        {label}
      </span>
      <code className="min-w-0 truncate font-mono text-[0.68rem] text-foreground">
        {value}
      </code>
    </div>
  );
}

function SectionEyebrow({
  children,
  icon: Icon,
}: {
  children: string;
  icon: ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex items-center gap-2 text-muted-foreground">
      <Icon className="h-3.5 w-3.5" />
      <Typography
        mondwest
        className="text-[0.68rem] uppercase tracking-[0.18em]"
      >
        {children}
      </Typography>
    </div>
  );
}

export default function VisualLibraryPage() {
  const { availableThemes, setTheme, theme, themeName } = useTheme();
  const { setAfterTitle, setEnd } = usePageHeader();
  const builtInThemes = Object.values(BUILTIN_THEMES);
  const customThemeCount = Math.max(availableThemes.length - builtInThemes.length, 0);

  useLayoutEffect(() => {
    setAfterTitle(
      <span className="whitespace-nowrap text-xs text-muted-foreground">
        {builtInThemes.length} built-in themes
      </span>,
    );
    setEnd(
      <div className="flex w-full min-w-0 justify-start">
        <Badge className="max-w-full truncate">
          Active: {theme.label}
        </Badge>
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [builtInThemes.length, setAfterTitle, setEnd, theme.label]);

  return (
    <div className="flex flex-col gap-6">
      <section className="grid gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
        <Card className="overflow-hidden">
          <CardContent className="grid gap-6 p-5 sm:p-6">
            <SectionEyebrow icon={SwatchBook}>Visual source map</SectionEyebrow>
            <div className="max-w-3xl">
              <Typography
                as="h2"
                expanded
                className="text-2xl font-bold leading-tight sm:text-3xl"
              >
                Theme tokens, assets, and chrome rules in one place.
              </Typography>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
                This page is the dashboard-side library for Hermes visual
                decisions. It keeps design references close to the runtime
                surfaces that consume them: dashboard themes, shell components,
                plugin slots, and CLI skins.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <StatTile
                icon={Palette}
                label="Themes"
                value={String(availableThemes.length)}
                detail={`${customThemeCount} custom`}
              />
              <StatTile
                icon={Image}
                label="Asset slots"
                value={String(ASSET_SLOTS.length)}
                detail="Theme-scoped"
              />
              <StatTile
                icon={Component}
                label="Buckets"
                value={String(COMPONENT_BUCKETS.length)}
                detail="CSS-variable driven"
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Theme</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <ThemeSwatch theme={theme} />
            <div className="grid gap-2 text-xs">
              <TokenLine label="Name" value={themeName} />
              <TokenLine label="Background" value={layerToCss(theme.palette.background)} />
              <TokenLine label="Midground" value={layerToCss(theme.palette.midground)} />
              <TokenLine label="Font" value={theme.typography.fontSans} />
              <TokenLine label="Layout" value={theme.layout.density} />
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {builtInThemes.map((entry) => (
          <ThemeCard
            active={entry.name === themeName}
            key={entry.name}
            onSelect={() => setTheme(entry.name)}
            theme={entry}
          />
        ))}
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Asset Slots</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3">
            {ASSET_SLOTS.map((slot) => (
              <div
                className="grid gap-2 border border-border bg-muted/20 p-3 sm:grid-cols-[7rem_minmax(0,1fr)]"
                key={slot.name}
              >
                <div className="flex items-center gap-2">
                  <Image className="h-3.5 w-3.5 text-muted-foreground" />
                  <code className="font-mono text-xs text-foreground">
                    {slot.name}
                  </code>
                </div>
                <div className="min-w-0">
                  <code className="block truncate font-mono text-[0.7rem] text-primary">
                    {slot.varName}
                  </code>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">
                    {slot.purpose}
                  </p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Component Buckets</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex flex-wrap gap-2">
              {COMPONENT_BUCKETS.map((bucket) => (
                <Badge className="gap-1.5" key={bucket}>
                  <Layers className="h-3 w-3" />
                  {bucket}
                </Badge>
              ))}
            </div>
            <div className="border border-border bg-background/35 p-4">
              <SectionEyebrow icon={FileCode2}>Variable pattern</SectionEyebrow>
              <code className="mt-3 block overflow-x-auto whitespace-nowrap font-mono text-xs text-foreground">
                --component-&lt;bucket&gt;-&lt;css-property&gt;
              </code>
              <p className="mt-3 text-xs leading-5 text-muted-foreground">
                Themes can tune card borders, sidebar chrome, page surfaces,
                progress treatment, and other shell pieces without rewriting
                dashboard page components.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Source Pointers</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3">
            {SOURCE_POINTERS.map((item) => (
              <div className="border-l border-border pl-3" key={item.path}>
                <div className="flex items-center gap-2">
                  <Boxes className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-sm font-medium">{item.label}</span>
                </div>
                <code className="mt-1 block truncate font-mono text-[0.7rem] text-primary">
                  {item.path}
                </code>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  {item.detail}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Contribution Checks</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3">
            {QUALITY_GATES.map((gate) => (
              <div className="flex items-start gap-3" key={gate}>
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center border border-primary/40 bg-primary/10 text-primary">
                  <Check className="h-3 w-3" />
                </span>
                <p className="text-sm leading-6 text-muted-foreground">
                  {gate}
                </p>
              </div>
            ))}
            <div className="mt-2 border border-border bg-muted/20 p-4">
              <SectionEyebrow icon={Type}>Rule of thumb</SectionEyebrow>
              <p className="mt-2 text-xs leading-5 text-muted-foreground">
                Use a theme or plugin slot when a visual decision should be
                reusable. Put styling inside one page only when the decision is
                truly local to that workflow.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function StatTile({
  detail,
  icon: Icon,
  label,
  value,
}: {
  detail: string;
  icon: ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="min-w-0 border border-border bg-muted/20 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
          {label}
        </span>
        <Icon className="h-4 w-4 shrink-0 text-primary" />
      </div>
      <div className="mt-4 flex items-end justify-between gap-3">
        <Typography expanded className="text-3xl font-bold leading-none">
          {value}
        </Typography>
        <span className="pb-1 text-xs text-muted-foreground">{detail}</span>
      </div>
    </div>
  );
}
