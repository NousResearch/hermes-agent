import { useCallback, useEffect, useRef, useState } from "react";
import { Globe, Keyboard, MousePointer2, RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type { RemoteBrowserStatus } from "@/lib/api";

const DEFAULT_URL = "https://make.powerautomate.com/";

function displayHost(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url || "not connected";
  }
}

export default function RemoteBrowserPage() {
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [url, setUrl] = useState(DEFAULT_URL);
  const [status, setStatus] = useState<RemoteBrowserStatus | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [typing, setTyping] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const shot = await api.getRemoteBrowserScreenshot();
      setImageDataUrl(shot.data.image_data_url);
      setStatus(shot.data.status);
    } catch (err) {
      try {
        setStatus(await api.getRemoteBrowserStatus());
      } catch {
        // Keep the original error below.
      }
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setEnd(
      <Button outlined size="sm" onClick={() => void refresh()} disabled={loading}>
        {loading ? <Spinner className="h-4 w-4" /> : <RefreshCw className="h-4 w-4" />}
        Refresh
      </Button>,
    );
    return () => setEnd(null);
  }, [loading, refresh, setEnd]);

  useEffect(() => {
    let cancelled = false;
    api.getRemoteBrowserStatus()
      .then((next) => {
        if (!cancelled) setStatus(next);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function openUrl() {
    setLoading(true);
    setError(null);
    try {
      const result = await api.openRemoteBrowser(url);
      if (result.status) setStatus(result.status);
      await refresh();
      showToast("Remote browser opened", "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      showToast(message, "error");
    } finally {
      setLoading(false);
    }
  }

  async function sendText() {
    if (!typing) return;
    setLoading(true);
    try {
      await api.typeRemoteBrowser(typing);
      setTyping("");
      await refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err), "error");
    } finally {
      setLoading(false);
    }
  }

  async function press(key: string) {
    setLoading(true);
    try {
      await api.pressRemoteBrowserKey(key);
      await refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err), "error");
    } finally {
      setLoading(false);
    }
  }

  async function clickImage(event: React.MouseEvent<HTMLImageElement>) {
    const image = imageRef.current;
    if (!image) return;
    const rect = image.getBoundingClientRect();
    const scaleX = image.naturalWidth / rect.width;
    const scaleY = image.naturalHeight / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    setLoading(true);
    try {
      await api.clickRemoteBrowser(x, y);
      await refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err), "error");
    } finally {
      setLoading(false);
    }
  }

  async function scrollViewport(event: React.WheelEvent<HTMLDivElement>) {
    if (!event.shiftKey) return;
    event.preventDefault();
    setLoading(true);
    try {
      await api.scrollRemoteBrowser(event.deltaY);
      await refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err), "error");
    } finally {
      setLoading(false);
    }
  }

  const connected = status?.connected ?? false;

  return (
    <div className="space-y-4 p-4 lg:p-6">
      <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Remote Browser</h1>
          <p className="text-sm text-muted-foreground">
            Open a server-side browser, log in yourself, then let Hermes continue from the same session.
          </p>
        </div>
        <Badge tone={connected ? "success" : "warning"}>
          {connected ? displayHost(status?.url ?? "") : "disconnected"}
        </Badge>
      </div>

      <Card>
        <CardContent className="space-y-3 py-4">
          <div className="flex flex-col gap-2 md:flex-row">
            <Input value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://…" />
            <Button onClick={() => void openUrl()} disabled={loading}>
              <Globe className="h-4 w-4" />
              Open
            </Button>
          </div>
          <div className="grid gap-2 text-xs text-muted-foreground md:grid-cols-3">
            <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4" />Dashboard auth protects this surface.</div>
            <div className="flex items-center gap-2"><MousePointer2 className="h-4 w-4" />Click the screenshot to click the remote page.</div>
            <div className="flex items-center gap-2"><Keyboard className="h-4 w-4" />Type below; passwords are sent directly to the browser action.</div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-3 py-4">
          <div className="flex flex-col gap-2 md:flex-row">
            <Input
              value={typing}
              onChange={(event) => setTyping(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void sendText();
              }}
              placeholder="Type into focused field"
            />
            <Button outlined onClick={() => void sendText()} disabled={loading || !typing}>Type</Button>
            <Button outlined onClick={() => void press("Enter")} disabled={loading}>Enter</Button>
            <Button outlined onClick={() => void press("Tab")} disabled={loading}>Tab</Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Hold Shift while scrolling over the viewport to scroll the remote page. Refresh after MFA/device prompts.
          </p>
        </CardContent>
      </Card>

      {error ? <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</div> : null}

      <Card>
        <CardContent className="py-4">
          <div
            className="relative flex min-h-[420px] items-center justify-center overflow-auto rounded-md border border-border bg-black/80"
            onWheel={(event) => void scrollViewport(event)}
          >
            {loading && (
              <div className="absolute right-3 top-3 rounded-full bg-background/80 px-3 py-1 text-xs text-muted-foreground">
                <Spinner className="mr-1 inline h-3 w-3" /> working
              </div>
            )}
            {imageDataUrl ? (
              <img
                ref={imageRef}
                src={imageDataUrl}
                alt="Remote browser viewport"
                className="max-h-[72vh] max-w-full cursor-crosshair select-none"
                draggable={false}
                onClick={(event) => void clickImage(event)}
              />
            ) : (
              <div className="text-center text-sm text-muted-foreground">
                No viewport yet. Open a URL to start the remote browser.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      <Toast toast={toast} />
    </div>
  );
}
