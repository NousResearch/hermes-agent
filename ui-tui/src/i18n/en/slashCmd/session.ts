// slashCmd.session — replies and usage hints of app/slash/commands/session.ts.
// Command NAMES/aliases/arg syntax
// stay literal; config VALUES echoed back (auto/light/dark, fast/normal, hide/show, …)
// are identifiers and are interpolated, not translated.

export const slashCmdSessionEn = {
  session: {
    bg: {
      started: (taskId: string) => `bg ${taskId} started`,
      usage: '/bg <prompt>'
    },
    branch: {
      branched: (title: string) => `branched → ${title}`
    },
    btw: {
      answering: (taskId: string) => `btw ${taskId} — answering from a conversation snapshot`,
      usage: '/btw <question>'
    },
    busy: {
      mode: (mode: string) => `busy input mode: ${mode}`,
      usage: 'usage: /busy [queue|steer|interrupt|status]'
    },
    compress: {
      // {0}=message count (pre-formatted)
      compressedOne: (count: string) => `compressed ${count} message`,
      compressedOther: (count: string) => `compressed ${count} messages`,
      nothing: 'nothing to compress',
      // {0}=compact token count, appended to compressedOne/Other
      tokSuffix: (tokens: string) => ` · ${tokens} tok`
    },
    fast: {
      mode: (mode: string) => `fast mode: ${mode}`,
      usage: 'usage: /fast [normal|fast|ultrafast|status|on|off|toggle]'
    },
    indicator: {
      current: (style: string) => `indicator: ${style}`,
      switched: (style: string) => `indicator → ${style}`,
      usage: (styles: string) => `usage: /indicator [${styles}]`
    },
    model: {
      cancel: 'Cancel',
      expensiveDetail: 'This model has unusually high known pricing.',
      expensiveTitle: 'Expensive model selection',
      invalidResponse: 'error: invalid response: model switch',
      switchAnyway: 'Switch anyway',
      switched: (model: string) => `model → ${model}`,
      switchedDeferred: (model: string) => `model → ${model} (applies next turn)`
    },
    personality: {
      changed: (value: string) => `personality: ${value}`,
      changedCleared: (value: string) => `personality: ${value} · transcript cleared`,
      defaultValue: 'default'
    },
    pet: {
      noOutput: '/pet: no output',
      // {0}=warning text, {1}=command output
      warning: (warning: string, body: string) => `warning: ${warning}\n${body}`
    },
    reasoning: {
      current: (value: string) => `reasoning: ${value}`,
      // {0}=effort value, {1}=display mode
      currentWithDisplay: (value: string, display: string) => `reasoning: ${value} · display ${display}`
    },
    sessions: {
      busyGuardAction: 'switch sessions'
    },
    skin: {
      current: (skin: string) => `skin: ${skin}`,
      defaultValue: 'default',
      switched: (skin: string) => `skin → ${skin}`
    },
    theme: {
      current: (theme: string) => `theme: ${theme}`,
      switched: (theme: string) => `theme → ${theme}`,
      usage: 'usage: /theme [auto|light|dark]'
    },
    usage: {
      compressions: (count: string) => `Compressions: ${count}`,
      // {0}=used tokens (with ~ mark when estimated), {1}=max tokens, {2}=percent (with ~ mark)
      context: (used: string, max: string, percent: string) => `Context: ${used} / ${max} (${percent}%)`,
      noCalls: 'no API calls yet',
      rowApiCalls: 'API calls',
      rowInputTokens: 'Input tokens',
      rowModel: 'Model',
      rowOutputTokens: 'Output tokens',
      rowTotalTokens: 'Total tokens',
      usageTitle: 'Usage'
    },
    verbose: {
      current: (value: string) => `verbose: ${value}`
    },
    voice: {
      disabled: 'Voice mode disabled.',
      enabled: 'Voice mode enabled',
      enabledWithTts: 'Voice mode enabled (TTS enabled)',
      off: 'OFF',
      offHint: '  /voice off  to disable voice mode',
      on: 'ON',
      recordHint: (recordKey: string) => `  ${recordKey} to start/stop recording`,
      requirements: '  Requirements:',
      statusMode: (mode: string) => `  Mode:       ${mode}`,
      statusRecordKey: (recordKey: string) => `  Record key: ${recordKey}`,
      statusTitle: 'Voice Mode Status',
      statusTts: (tts: string) => `  TTS:        ${tts}`,
      ttsDisabled: 'Voice TTS disabled.',
      ttsEnabled: 'Voice TTS enabled.',
      ttsHint: '  /voice tts  to toggle speech output'
    },
    yolo: {
      off: 'yolo off',
      on: 'yolo on'
    }
  }
}
