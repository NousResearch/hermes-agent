// Overlays, pickers and dialogs: prompts, model/pet pickers, hubs, agents panel,
// billing/subscription, connection setup. Owned namespaces: `overlay`, `prompt`,
// `picker`, `hub`, `agents`, `billing`, `connection`, `journey`.

export const overlaysEn = {
  prompt: {
    approval: {
      title: '⚠ approval required',
      pageMore: (from: number, to: number, total: number) =>
        `lines ${from}–${to} of ${total} · press v to see the rest before choosing`,
      pageEnd: (from: number, to: number, total: number) =>
        `lines ${from}–${to} of ${total} (end) · press v to go back to the top`,
      once: 'Allow once',
      session: 'Allow this session',
      always: 'Always allow',
      deny: 'Deny',
      tell: 'No, and tell Hermes what to do instead',
      tellHint: 'type what Hermes should do instead · Enter deny and send · Esc back'
    },
    rewind: {
      title: '↺ rewind — pick an earlier prompt to edit and resend',
      detail: 'Sending the edit removes that prompt and everything after it from this session.',
      editing: (turn: string) => `editing prompt ${turn}`,
      preview: (turn: string) => `prompt ${turn}`,
      marker: '↺ selected in the rewind picker',
      listHint: '↑/↓ select · Enter edit & resend · f fork from here · Esc close',
      editHint: 'Enter rewind here and send · Esc back to the list'
    },
    clarify: {
      other: 'Other (type your answer)',
      skipped: '(skipped)',
      toggle: 'Space toggle',
      confirmAndContinue: 'confirm and continue',
      lockAnswer: 'lock answer',
      typingHint: (enterAction: string) => `Enter ${enterAction} · Esc back`,
      hint: (enterAction: string) =>
        `↑/↓ select · Enter ${enterAction} · Tab/Shift+Tab switch question · Esc/Ctrl+C cancel`
    },
    confirm: {
      confirm: 'Yes',
      cancel: 'No',
      hint: '↑/↓ select · Enter confirm · Y/N quick · Esc cancel'
    }
  }
}
