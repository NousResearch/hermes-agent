import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Bot, ChevronDown } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import { cn } from "@/lib/utils";

interface SessionInfo {
  model?: string;
  provider?: string;
  credential_warning?: string;
}

const STATE_LABEL: Record<ConnectionState, string> = {
  idle: "idle",
  connecting: "connecting",
  open: "live",
  closed: "closed",
  error: "error",
};

const STATE_TONE: Record<
  ConnectionState,
  "secondary" | "warning" | "success" | "destructive"
> = {
  idle: "secondary",
  connecting: "warning",
  open: "success",
  closed: "secondary",
  error: "destructive",
};

interface ChatModelControlProps {
  className?: string;
  buttonClassName?: string;
  onSlashCommand?: (slashCommand: string) => void;
}

export function ChatModelControl({
  className,
  buttonClassName,
  onSlashCommand,
}: ChatModelControlProps) {
  const [version, setVersion] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const gw = useMemo(() => new GatewayClient(), [version]);

  const [state, setState] = useState<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [info, setInfo] = useState<SessionInfo>({});
  const [modelOpen, setModelOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const offState = gw.onState(setState);

    const offSessionInfo = gw.on<SessionInfo>("session.info", (ev) => {
      if (ev.session_id) {
        setSessionId(ev.session_id);
      }

      if (ev.payload) {
        setInfo((prev) => ({ ...prev, ...ev.payload }));
      }
    });

    const offError = gw.on<{ message?: string }>("error", (ev) => {
      const message = ev.payload?.message;

      if (message) {
        setError(message);
      }
    });

    gw.connect()
      .then(() => {
        if (cancelled) {
          return;
        }
        return gw.request<{ session_id: string }>("session.create", {});
      })
      .then((created) => {
        if (cancelled || !created?.session_id) {
          return;
        }
        setSessionId(created.session_id);
        setError(null);
      })
      .catch((e: Error) => {
        if (!cancelled) {
          setError(e.message);
        }
      });

    return () => {
      cancelled = true;
      offState();
      offSessionInfo();
      offError();
      gw.close();
    };
  }, [gw]);

  const onModelSubmit = useCallback(
    (slashCommand: string) => {
      if (onSlashCommand) {
        onSlashCommand(slashCommand);
        setModelOpen(false);
        return;
      }

      if (!sessionId) {
        return;
      }

      void gw.request("slash.exec", {
        session_id: sessionId,
        command: slashCommand,
      });
      setModelOpen(false);
    },
    [gw, onSlashCommand, sessionId],
  );

  const reconnect = useCallback(() => {
    setError(null);
    setSessionId(null);
    setInfo({});
    setVersion((v) => v + 1);
  }, []);

  const canPickModel = state === "open" && !!sessionId;
  const modelLabel = (info.model ?? "model").split("/").slice(-1)[0] ?? "model";
  const title =
    error ??
    info.credential_warning ??
    (canPickModel ? "Switch Hermes model" : "Model picker is connecting");

  return (
    <div className={cn("flex min-w-0 items-center gap-1.5", className)}>
      <Button
        ghost
        disabled={!canPickModel}
        onClick={() => setModelOpen(true)}
        title={title}
        aria-label="Switch Hermes model"
        className={cn(
          "h-6 min-w-0 rounded border border-current/35 px-2 py-1",
          "bg-black/20 backdrop-blur-sm",
          "opacity-80 hover:opacity-100 hover:border-current/65",
          "transition-opacity duration-150 normal-case font-normal tracking-normal",
          "text-[0.65rem] sm:h-7 sm:px-2.5 sm:text-xs",
          !canPickModel && "cursor-not-allowed opacity-55",
          buttonClassName,
        )}
      >
        <span className="inline-flex min-w-0 items-center gap-1.5">
          <Bot className="h-3 w-3 shrink-0" />
          <span className="hidden max-w-[8rem] truncate tracking-wide min-[560px]:inline">
            {modelLabel}
          </span>
          <ChevronDown className="h-3 w-3 shrink-0 opacity-65" />
        </span>
      </Button>

      <Badge
        tone={STATE_TONE[state]}
        className="hidden shrink-0 px-1.5 py-0 text-[9px] min-[760px]:inline-flex"
      >
        {STATE_LABEL[state]}
      </Badge>

      {error && (
        <Button
          ghost
          onClick={reconnect}
          title={error}
          aria-label="Retry model picker connection"
          className="hidden h-6 px-1.5 py-0 text-[0.65rem] normal-case opacity-70 hover:opacity-100 min-[920px]:inline-flex"
        >
          retry
        </Button>
      )}

      {modelOpen && canPickModel && sessionId && (
        <ModelPickerDialog
          gw={gw}
          sessionId={sessionId}
          title="Switch Model"
          onClose={() => setModelOpen(false)}
          onSubmit={onModelSubmit}
        />
      )}
    </div>
  );
}
