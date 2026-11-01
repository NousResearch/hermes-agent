// Overlays, pickers and dialogs: prompts, model/pet pickers, hubs, agents panel,
// billing/subscription, connection setup. Owned namespaces: `overlay`, `prompt`,
// `picker`, `hub`, `agents`, `billing`, `connection`, `journey`.

export const overlaysEn = {
  prompt: {
    approval: {
      title: '⚠ approval required',
      moreLines: (count: number) => `… +${count} more line${count === 1 ? '' : 's'} (full text above)`,
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
      listHint: '↑/↓ select · Enter edit · Esc close',
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
