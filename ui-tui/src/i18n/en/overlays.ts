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
      prime: 'esc again to edit previous message',
      browse: 'Browsing transcript · ↑↓/jk scroll · ←→/hl prompts · ↵ edit · f fork · esc back',
      pending: (turn: string) => `↺ Editing prompt ${turn} — Enter rewinds here and sends · Esc Esc cancels`
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
