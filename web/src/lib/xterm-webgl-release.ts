/** Release WebGL contexts left alive when xterm's WebglAddon is disposed. */
export function loseWebglContexts(host: ParentNode): number {
  let released = 0;
  for (const canvas of Array.from(host.querySelectorAll("canvas"))) {
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    const lose = gl?.getExtension("WEBGL_lose_context");
    if (lose) {
      lose.loseContext();
      released += 1;
    }
  }
  return released;
}
