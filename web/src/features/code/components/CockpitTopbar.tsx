import { useState, useRef, useEffect } from "react";
import { ChevronDown, Server } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { BackendStatusIndicator } from "@/components/BackendStatusIndicator";
import { ProviderModelSwitcher } from "./ProviderModelSwitcher";

interface CockpitTopbarProps {
  title: string;
  subtitle?: string;
  activeArea: string;
}

export function CockpitTopbar({ title, subtitle }: CockpitTopbarProps) {
  const [providerExpanded, setProviderExpanded] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setProviderExpanded(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <header className="flex items-center justify-between px-4 py-2 border-b border-border bg-card/50">
      {/* Left: Title */}
      <div className="flex items-center gap-3 min-w-0">
        <div>
          <h1 className="text-sm font-display font-bold tracking-wide uppercase">
            {title}
          </h1>
          {subtitle && (
            <p className="text-[10px] text-muted-foreground truncate">{subtitle}</p>
          )}
        </div>
      </div>

      {/* Right: Status indicators */}
      <div className="flex items-center gap-4">
        {/* Backend status */}
        <div className="hidden sm:flex items-center">
          <BackendStatusIndicator />
        </div>

        {/* Agent / session status */}
        <SessionStatusBadge />

        {/* Provider / Model Switcher */}
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setProviderExpanded(!providerExpanded)}
            className="flex items-center gap-1.5 px-2 py-1 border border-border rounded hover:bg-foreground/5 transition-colors"
          >
            <Server className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-[10px] font-compressed tracking-widest uppercase text-muted-foreground">
              Provider
            </span>
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </button>

          {providerExpanded && (
            <div className="absolute right-0 top-full mt-2 z-50 w-80">
              <ProviderModelSwitcher onSelectionChange={() => setProviderExpanded(false)} />
            </div>
          )}
        </div>

        {/* Compact backend status for mobile */}
        <div className="sm:hidden">
          <BackendStatusIndicator compact />
        </div>
      </div>
    </header>
  );
}

function SessionStatusBadge() {
  const [count, setCount] = useState(0);
  const [status, setStatus] = useState<"running" | "idle">("idle");

  // Poll session count — lightweight
  useState(() => {
    const load = async () => {
      try {
        const { codeApi } = await import("@/lib/codeApi");
        const data = await codeApi.getCodeSessions({ status: "running", limit: 1 });
        setCount(data.total);
        setStatus(data.total > 0 ? "running" : "idle");
      } catch {
        setStatus("idle");
      }
    };
    load();
    const interval = setInterval(load, 10000);
    return () => clearInterval(interval);
  });

  if (status === "running") {
    return (
      <Badge variant="success" className="text-[10px] gap-1">
        <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
        {count} agent{count !== 1 ? "s" : ""} active
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className="text-[10px] text-muted-foreground">
      No active agents
    </Badge>
  );
}
