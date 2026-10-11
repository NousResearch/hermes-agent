// The /connectors overlay: the user's hosted connector accounts.
// Owned namespace: `connectors`. Leaves are strings or `(...args) => string`;
// packs use positional `{0}`, `{1}` placeholders, so keep argument order stable.
//
// Hotkey chords inside hints are part of the hint string. Account status values
// come from the backend and are mapped to these labels at render time.

export const connectorsEn = {
  connectors: {
    title: 'Connector accounts',
    loading: 'loading accounts…',
    empty: 'No connector accounts yet. Press a to add one.',
    unnamedTag: '(unnamed)',
    retiredFolded: (count: number) => `${count} retired (Tab to show)`,
    retiredShown: (count: number) => `${count} retired shown (Tab to hide)`,
    status: {
      active: 'active',
      expired: 'expired',
      failed: 'failed',
      inactive: 'inactive',
      pending: 'pending',
      retired: 'retired',
      revoked: 'revoked'
    },
    rename: {
      // {0} app, {1} current name
      title: (app: string, name: string) => `Rename ${app} account ${name}`,
      hint: 'Enter save · Esc cancel'
    },
    add: {
      pickApp: 'Add an account to which app?',
      pickHint: '↑/↓ select · Enter choose · Esc cancel',
      noApps: 'No connectors are available to add.',
      name: (app: string) => `Name the new ${app} account`,
      nameHint: 'Enter connect · Esc cancel'
    },
    link: {
      // {0} app, {1} name
      title: (app: string, name: string) => `Connect ${app} account ${name}`,
      reconnectTitle: (app: string, name: string) => `Reconnect ${app} account ${name}`,
      waiting: 'Waiting for you to finish in the browser…',
      hint: 'Enter open in browser · Esc stop waiting',
      starting: 'Starting…'
    },
    remove: {
      // {0} app, {1} name
      confirm: (app: string, name: string) => `Remove ${app} account ${name}? This signs Hermes out of it.`
    },
    notice: {
      // {0} app, {1} name
      connected: (app: string, name: string) => `${app} account ${name} is connected.`,
      renamed: (app: string, name: string) => `Renamed to ${name} (${app}).`,
      removed: (app: string, name: string) => `Removed ${app} account ${name}.`,
      notConnected: (app: string) => `${app} did not connect. Try again.`,
      stopped: 'Stopped waiting. The link still works until it expires.',
      waitEnded: 'Stopped waiting: the link expired. The list shows where the account stands.',
      nameInvalid: 'Use 1-32 lowercase letters, digits or -, starting with a letter or digit.',
      nameTaken: 'That name is already used.',
      retiredNoReconnect: 'A retired account cannot be reconnected. Add another account instead.',
      browserDidNotOpen: 'The browser did not open. Copy the link above.'
    },
    hint: 'r rename · a add · c reconnect · x remove · Tab retired · Esc/q close',
    errorLine: (message: string) => `error: ${message}`
  }
}
