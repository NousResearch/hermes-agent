/**
 * Release the WebGL contexts an xterm WebglAddon left behind.
 *
 * `@xterm/addon-webgl` (0.19) removes its canvas on dispose but never calls
 * `WEBGL_lose_context.loseContext()`, so the GL context stays alive until the
 * browser garbage-collects the canvas. Browsers cap live contexts (~16) and
 * force-lose the oldest when the cap is hit; ChatPage rebuilds the terminal on
 * every PTY reconnect, so a reconnect storm walks straight into that cap and
 * eventually loses the context of the terminal the user is looking at.
 *
 * Call it on the terminal host BEFORE `term.dispose()` removes the canvases.
 * Returns the number of contexts released.
 */
export function loseWebglContexts(host: ParentNode): number {
  let released = 0
  for (const canvas of Array.from(host.querySelectorAll('canvas'))) {
    // getContext() with the type a canvas already holds returns that context;
    // for a 2D canvas both WebGL lookups return null without creating one.
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl')
    const lose = gl?.getExtension('WEBGL_lose_context')
    if (lose) {
      lose.loseContext()
      released += 1
    }
  }
  return released
}
