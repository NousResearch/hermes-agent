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
      deny: 'Deny'
    },
    clarify: {
      other: 'Other (type your answer)',
      skipped: '(skipped)',
      confirmAndContinue: 'confirm and continue',
      lockAnswer: 'lock answer',
      batchTypingHint: (enterAction: string) => `Enter ${enterAction} · Esc back`,
      batchHint: (enterAction: string) =>
        `↑/↓ select · Enter ${enterAction} · Tab/Shift+Tab switch question · Esc/Ctrl+C cancel`,
      typingHint: (escAction: string) => `Enter send · Esc ${escAction} ·`,
      back: 'back',
      cancel: 'cancel',
      macClipboardHint: 'Cmd+C copy · Cmd+V paste · Ctrl+C cancel',
      ctrlCCancel: 'Ctrl+C cancel'
    },
    confirm: {
      confirm: 'Yes',
      cancel: 'No',
      hint: '↑/↓ select · Enter confirm · Y/N quick · Esc cancel'
    }
  }
}
