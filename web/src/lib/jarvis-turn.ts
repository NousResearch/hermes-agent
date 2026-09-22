import type { GatewayClient } from './gatewayClient';

interface JarvisTurnOptions {
  sessionId: string;
  storedSessionId: string;
  text: string;
  voiceContext: string;
  signal: AbortSignal;
  onDelta?: (text: string) => void;
  onActivity?: (activity: string) => void;
}

/** A voice turn uses the same event/terminal-outcome contract as Hermes chat. */
export function submitJarvisTurn(gateway: GatewayClient, options: JarvisTurnOptions): Promise<string> {
  return new Promise((resolve, reject) => {
    let text = '';
    let settled = false;
    let timeout: ReturnType<typeof setTimeout>;
    const unsubscribe: Array<() => void> = [];
    const matches = (sid?: string) => sid === options.sessionId || sid === options.storedSessionId;
    const cleanup = () => {
      settled = true;
      clearTimeout(timeout);
      unsubscribe.forEach(off => off());
      options.signal.removeEventListener('abort', abort);
    };
    const fail = (error: unknown) => {
      if (settled) return;
      cleanup();
      reject(error);
    };
    const abort = () => fail(new Error('Call request cancelled.'));
    const activity = () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => fail(new Error(
        'No response from Hermes for two minutes. The task may still be running; check its session before retrying.',
      )), 120_000);
    };
    if (options.signal.aborted) { abort(); return; }
    options.signal.addEventListener('abort', abort, { once: true });
    unsubscribe.push(gateway.onState(state => {
      if (state === 'closed' || state === 'error') {
        fail(new Error('Connection to Hermes lost. Reconnect and check the session before retrying.'));
      }
    }));
    unsubscribe.push(gateway.onAny(event => {
      // Global notifications and other sessions must not keep this turn alive.
      if (matches(event.session_id)) activity();
    }));
    unsubscribe.push(gateway.on('message.delta', event => {
      if (!matches(event.session_id) || !event.payload?.text) return;
      text += event.payload.text;
      options.onActivity?.('Receiving response');
      options.onDelta?.(text);
    }));
    unsubscribe.push(gateway.on('tool.start', event => {
      if (matches(event.session_id)) options.onActivity?.('Hermes is running a tool');
    }));
    unsubscribe.push(gateway.on('message.complete', event => {
      if (!matches(event.session_id)) return;
      const result = event.payload;
      if (result?.status === 'error' || result?.status === 'interrupted') {
        fail(new Error(result.error || (typeof result.text === 'string' && result.text) ||
          (result.status === 'interrupted' ? 'Task interrupted.' : 'Hermes could not complete the task.')));
        return;
      }
      // Streamed commentary may precede tool calls; the terminal text is authoritative.
      const final = typeof result?.text === 'string' ? result.text.trim() : text.trim();
      if (!final) { fail(new Error('Hermes returned no spoken response. Check the session for task results.')); return; }
      cleanup();
      resolve(final);
    }));
    unsubscribe.push(gateway.on('error', event => {
      if (matches(event.session_id)) fail(new Error(event.payload?.message || 'Hermes request failed.'));
    }));
    activity();
    options.onActivity?.('Waiting for Hermes');
    void gateway.request<{ status: string }>('prompt.submit', {
      session_id: options.sessionId,
      text: options.text,
      surface: 'voice-live',
      voice_context: options.voiceContext,
    }).then(result => {
      if (settled) return;
      if (result.status !== 'streaming') options.onActivity?.('Waiting for the current Hermes task');
      activity();
    }, fail);
  });
}
