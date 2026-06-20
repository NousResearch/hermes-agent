import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { fetchJSON } from "@/lib/api";

// 网页端工作流:浏览器访问本地 Hermes,接同一个本地 Langflow。
// EasyHermes 账号只负责 Kari 节点/计费/云端发 key;无账号仍然启动本地画布。
interface AuthStatus {
  loggedIn: boolean;
  cloudBaseUrl: string;
  cloudReachable?: boolean;
  error?: string | null;
}
interface BackendStatus {
  state: string;
  url: string;
  error?: string | null;
}
type Phase = "checking" | "starting" | "ready" | "error";

const POLL_MS = 1500;
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function post<T>(path: string, body?: unknown): Promise<T> {
  return fetchJSON<T>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export default function WorkflowPage() {
  const [phase, setPhase] = useState<Phase>("checking");
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState("http://127.0.0.1:7860");
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const checkAuth = useCallback(async () => {
    setPhase("checking");
    setError(null);
    try {
      const a = await fetchJSON<AuthStatus>("/api/workflow/auth-status");
      if (!mounted.current) return;
      setError(a.error ?? null);
      setPhase("starting");
    } catch (e) {
      if (mounted.current) {
        setError(msg(e));
        setPhase("error");
      }
    }
  }, []);

  useEffect(() => {
    void checkAuth();
  }, [checkAuth]);

  // Start + poll the local Langflow whenever we enter the "starting" phase.
  useEffect(() => {
    if (phase !== "starting") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, ms);
      });

    (async () => {
      try {
        let st = await post<BackendStatus>("/api/workflow/start");
        while (!cancelled) {
          if (st.url) setUrl(st.url);
          if (st.state === "ready") {
            setPhase("ready");
            return;
          }
          if (st.state === "error" || st.state === "exited") {
            setError(st.error || st.state);
            setPhase("error");
            return;
          }
          await sleep(POLL_MS);
          if (cancelled) return;
          st = await fetchJSON<BackendStatus>("/api/workflow/status");
        }
      } catch (e) {
        if (!cancelled) {
          setError(msg(e));
          setPhase("error");
        }
      }
    })();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [phase]);

  if (phase === "ready") {
    // 左:本地画布(Langflow,爱马仕经 MCP 驱动)。右:爱马仕 /copilot 气泡页,
    // 复用本地 EasyHermes agent,不再嵌 xterm /chat。
    const basePath =
      (window as unknown as { __HERMES_BASE_PATH__?: string }).__HERMES_BASE_PATH__ ?? "";
    return (
      <div className="flex h-full w-full">
        <iframe className="h-full flex-1 border-0" src={url} title="工作流画布" />
        <iframe
          className="h-full w-[420px] shrink-0 border-0 border-l border-border"
          src={`${basePath}/copilot`}
          title="爱马仕 Copilot"
        />
      </div>
    );
  }

  if (phase === "error") {
    return (
      <div className="grid h-full place-items-center p-6">
        <div className="flex flex-col items-center gap-3 text-center">
          <p className="text-sm font-medium">本地工作流不可用</p>
          {error && <p className="max-w-md text-xs text-muted-foreground">{error}</p>}
          <Button onClick={() => void checkAuth()}>重试</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="grid h-full place-items-center p-6">
      <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
        <Spinner />
        <span>{phase === "starting" ? "启动本地工作流…" : "检查中…"}</span>
      </div>
    </div>
  );
}
