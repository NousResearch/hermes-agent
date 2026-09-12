interface PtySocket {
  readonly readyState: number;
  send(data: string): void;
}

/**
 * Submit one finalized browser transcript to Ink's composer.
 *
 * Return is deliberately a separate frame after xterm commits the PTY echo: a
 * combined mobile WebSocket frame can be classified as a paste and remain in
 * the composer. The captured socket must still be current before Return is
 * sent, so a reconnect cannot submit into a replacement PTY session.
 */
export function submitVoiceTranscriptToPty(
  currentSocket: () => PtySocket | null,
  transcript: string,
  afterNextTerminalWrite: (ready: () => void) => void,
  onReturnFailed: () => void,
): boolean {
  const socket = currentSocket();
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;

  afterNextTerminalWrite(() => {
    if (currentSocket() === socket && socket.readyState === WebSocket.OPEN) {
      socket.send("\r");
    } else {
      onReturnFailed();
    }
  });
  socket.send(transcript);
  return true;
}
