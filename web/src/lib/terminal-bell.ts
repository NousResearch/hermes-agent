/** Play the dashboard equivalent of a terminal BEL. */
export function playTerminalBell(): void {
  try {
    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const stopAt = context.currentTime + 0.1;

    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0.05, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, stopAt);
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(stopAt);
    oscillator.addEventListener("ended", () => void context.close());
  } catch {
    // Some browsers block audio until a user gesture; a BEL must not disrupt chat.
  }
}
