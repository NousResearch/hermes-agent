// app/ layer (excluding slash): notices, turn-controller activity lines, gateway
// event copy, server-request prompts, session lifecycle, queue, pet.
// Owned namespaces: `notice`, `notify`, `turn`, `session`, `queue`, `event`, `request`, `pet`.

export const appEn = {
  notice: {},
  // Desktop notification bodies for display.bell_on_complete / bell_on_prompt.
  notify: {
    complete: 'Response complete',
    prompt: 'Waiting for your answer'
  }
}
