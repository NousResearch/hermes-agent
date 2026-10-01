// Connection setup overlay and the learning-journey (starmap) overlay.
// Owned namespaces: `connection`, `journey`. Leaves are strings or
// `(...args) => string`; packs supply `{0}`, `{1}` positional placeholders, so keep
// argument order stable and comment it when >1 arg.
//
// Not keyed on purpose: hotkey chords inside hints are part of the hint string,
// glyphs (`▸`, `✦`, `└─`, `›`) stay in code, and everything the learning graph
// renderer (learning_graph_render.py) ships already rendered — legend labels,
// category names, bucket labels, summary lines, node meta — is shown verbatim.

export const connectionEn = {
  connection: {
    // Card title verb, indexed by the target's `action` (backend value, not shown raw).
    verb: {
      authorize: 'Authorize',
      connect: 'Connect',
      enable: 'Enable',
      install: 'Install',
      reconnect: 'Reconnect'
    },
    header: {
      moreToAnswer: (count: number) => `${count} more to answer after this one.`
    },
    field: {
      set: 'Set',
      required: (label: string) => `${label} is required.`
    },
    selector: {
      skip: 'Skip',
      tryAgain: 'Try again'
    },
    failure: {
      expired: 'The link expired.',
      failed: 'That did not work.'
    },
    status: {
      finishing: 'Finishing…',
      working: 'Working…',
      pending: 'Pending…'
    },
    authorized: {
      title: 'Authorized. Tools unavailable.',
      continue: 'Continue',
      hint: 'Enter or Esc continue'
    },
    hint: {
      finishing: 'Esc close · Ctrl+C stop the turn',
      browser: 'Enter open in browser · Esc skip · Ctrl+C stop the turn',
      working: 'Esc skip · Ctrl+C stop the turn',
      retry: '←/→ select · Enter confirm · Esc skip · Ctrl+C stop the turn',
      form: '↑/↓ or Tab move · ←/→ select · Enter confirm · Esc skip · Ctrl+C stop the turn'
    },
    notice: {
      answerNotDelivered: 'That answer did not reach Hermes. Try again.',
      restartFailed: 'Hermes could not start that again. Try again.',
      browserDidNotOpen: 'The browser did not open. Copy the link above.'
    }
  },
  journey: {
    title: 'Journey',
    subtitle: 'learned skills & memories over time',
    closeHint: 'Esc/q close',
    error: (message: string) => `error: ${message}`,
    loading: 'assembling your learning map…',
    empty: 'No learning yet — your learned skills and memories will start mapping out here as you use Hermes.',
    noDetail: 'No additional detail recorded yet.',
    notice: {
      cannotEdit: 'cannot edit',
      noChanges: 'no changes'
    },
    confirmDelete: (label: string) => `delete ${label}? y/N`,
    slice: {
      skillsOne: (count: number) => `${count} skill`,
      skillsOther: (count: number) => `${count} skills`,
      memoriesOne: (count: number) => `${count} memory`,
      memoriesOther: (count: number) => `${count} memories`
    },
    hint: {
      item: '↑↓/jk scroll · PgUp/PgDn page · e edit · d delete · Esc/← back · q close',
      // Timeline hint segments; the component joins the applicable ones with ' · '.
      move: '↑↓/jk move',
      open: 'Enter/→ open',
      edit: 'e edit',
      delete: 'd delete',
      topBottom: 'g/G top/bottom',
      close: 'q close'
    }
  }
}
