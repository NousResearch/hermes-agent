// Content tables: tool verbs, fortunes, kaomoji faces, the setup-required panel.
// Owned namespace: `content`.
//
// Lists live as numbered leaves (`fortune01`…) because packs are flat
// dotted-key → string maps and cannot carry array leaves; the accessors in
// src/content/*.ts rebuild the arrays from `Object.values(...)` at call time.

export const contentEn = {
  content: {
    // Keyed by tool name (untranslated identifier) → progress verb.
    verbs: {
      browser: 'browsing',
      clarify: 'asking',
      create_file: 'creating',
      delegate_task: 'delegating',
      delete_file: 'deleting',
      execute_code: 'executing',
      image_generate: 'generating',
      list_files: 'listing',
      memory: 'remembering',
      patch: 'patching',
      read_file: 'reading',
      run_command: 'running',
      search_code: 'searching',
      search_files: 'searching',
      terminal: 'terminal',
      web_extract: 'extracting',
      web_search: 'searching',
      write_file: 'writing'
    },
    fortunes: {
      fortune01: 'you are one clean refactor away from clarity',
      fortune02: 'a tiny rename today prevents a huge bug tomorrow',
      fortune03: 'your next commit message will be immaculate',
      fortune04: 'the edge case you are ignoring is already solved in your head',
      fortune05: 'minimal diff, maximal calm',
      fortune06: 'today favors bold deletions over new abstractions',
      fortune07: 'the right helper is already in your codebase',
      fortune08: 'you will ship before overthinking catches up',
      fortune09: 'tests are about to save your future self',
      fortune10: 'your instincts are correctly suspicious of that one branch'
    },
    legendaryFortunes: {
      legendary01: 'legendary drop: one-line fix, first try',
      legendary02: 'legendary drop: every flaky test passes cleanly',
      legendary03: 'legendary drop: your diff teaches by itself'
    },
    // Kaomoji are language-neutral but keyed so a pack may swap them.
    faces: {
      face01: '(｡•́︿•̀｡)',
      face02: '(◔_◔)',
      face03: '(¬‿¬)',
      face04: '( •_•)>⌐■-■',
      face05: '(⌐■_■)',
      face06: '(´･_･`)',
      face07: '◉_◉',
      face08: '(°ロ°)',
      face09: '( ˘⌣˘)♡',
      face10: 'ヽ(>∀<☆)☆',
      face11: '٩(๑❛ᴗ❛๑)۶',
      face12: '(⊙_⊙)',
      face13: '(¬_¬)',
      face14: '( ͡° ͜ʖ ͡°)',
      face15: 'ಠ_ಠ'
    },
    setup: {
      title: 'Setup Required',
      intro: 'Hermes needs a model provider before the TUI can start a session.',
      actions: 'Actions',
      setupRow: 'run the first-time setup wizard in-place (adds a provider)',
      modelRow: 'pick a model (needs a session — add a provider first)',
      exitRow: 'exit and run `hermes setup` manually',
      footer: 'In the dashboard the Models page sets the profile default; on Desktop it is Settings -> Models.'
    }
  }
}
