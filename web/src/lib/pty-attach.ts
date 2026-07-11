/**
 * Keep-alive PTY attach helpers for the dashboard chat.
 *
 * The attach token is the identity the server uses to reattach a browser tab
 * to a living PTY. Profile switches and forced-fresh starts must rotate it so
 * the previous profile's process is not reused.
 */

/** Whether this connect should mint a new keep-alive attach token. */
export function shouldRotatePtyAttachToken(
  forceFresh: boolean,
  profileChanged: boolean,
): boolean {
  return forceFresh || profileChanged;
}

/**
 * After an async URL build (e.g. gated-mode ticket fetch), decide whether it
 * is still safe to construct the WebSocket.
 *
 * Effect cleanup sets ``unmounting = true`` when the connect effect is torn
 * down (profile switch, unmount, reconnect nonce change). Opening a socket
 * after that would assign ``wsRef`` for a stale scope.
 */
export function shouldOpenPtySocketAfterUrlBuild(unmounting: boolean): boolean {
  return !unmounting;
}
