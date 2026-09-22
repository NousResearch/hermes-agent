// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { GatewayClient } from './gatewayClient';
import { submitJarvisTurn } from './jarvis-turn';

class Socket extends EventTarget {
  static OPEN = 1;
  static instances: Socket[] = [];
  readyState = 0;
  sent: Array<{ id: string; method: string }> = [];
  constructor() { super(); Socket.instances.push(this); }
  send(raw: string) {
    const request = JSON.parse(raw);
    this.sent.push(request);
    queueMicrotask(() => this.frame({ jsonrpc: '2.0', id: request.id, result: { status: 'streaming' } }));
  }
  close() { this.readyState = 3; this.dispatchEvent(new Event('close')); }
  frame(value: unknown) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(value) })); }
  event(type: string, payload: unknown, session_id = 'runtime') {
    this.frame({ jsonrpc: '2.0', method: 'event', params: { type, payload, session_id } });
  }
}

let gateway: GatewayClient;
async function connect() {
  vi.stubGlobal('WebSocket', Socket);
  gateway = new GatewayClient();
  const connecting = gateway.connect('test-token');
  await Promise.resolve();
  const socket = Socket.instances.at(-1)!;
  socket.readyState = 1;
  socket.dispatchEvent(new Event('open'));
  await connecting;
  return socket;
}
function submit(signal = new AbortController().signal, onDelta = vi.fn()) {
  return submitJarvisTurn(gateway, {
    sessionId: 'runtime', storedSessionId: 'stored', text: 'Check the task',
    voiceContext: 'Short spoken reply', signal, onDelta,
  });
}
afterEach(() => { gateway?.close(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it('streams through the real RPC client and respects authoritative final outcomes and session isolation', async () => {
  const socket = await connect();
  const delta = vi.fn();
  const result = submit(undefined, delta);
  socket.event('message.complete', { text: 'Unrelated' }, 'other');
  socket.event('message.delta', { text: 'Checking now. ' });
  socket.event('message.delta', { text: 'Done.' });
  expect(delta).toHaveBeenLastCalledWith('Checking now. Done.');
  socket.event('message.complete', { text: 'Verified result.', status: 'complete' });
  await expect(result).resolves.toBe('Verified result.');
  const failure = submit();
  const rejected = expect(failure).rejects.toThrow('Provider unavailable');
  socket.event('message.delta', { text: 'I will do it.' });
  socket.event('message.complete', { text: '', status: 'error', error: 'Provider unavailable' });
  await rejected;
});

it('keeps tool work alive but never turns timeout, disconnect, or cancellation into success', async () => {
  vi.useFakeTimers();
  const socket = await connect();
  const controller = new AbortController();
  const running = submit(controller.signal);
  const rejected = expect(running).rejects.toThrow('cancelled');
  await vi.advanceTimersByTimeAsync(110_000);
  socket.event('tool.start', { tool: 'terminal' });
  await vi.advanceTimersByTimeAsync(110_000);
  controller.abort();
  await rejected;
  const silent = submit();
  const timedOut = expect(silent).rejects.toThrow('may still be running');
  socket.event('message.delta', { text: 'Partial response' });
  await vi.advanceTimersByTimeAsync(110_000);
  socket.event('status.update', { status: 'busy' }, 'other');
  await vi.advanceTimersByTimeAsync(10_000);
  await timedOut;
  const disconnected = submit();
  const lost = expect(disconnected).rejects.toThrow('Connection to Hermes lost');
  socket.close();
  await lost;
});
