import type { Translations } from './en.js'

export const sv: Translations = {
  hotkeys: {
    'copy selection': 'kopiera markeringen',
    'clear draft / interrupt / exit': 'rensa utkast / avbryt / avsluta',
    'copy selection when forwarded by the terminal':
      'kopiera markeringen när terminalen vidarebefordrar tangenttryckningen',
    'copy selection / clear draft / interrupt / exit': 'kopiera markeringen / rensa utkast / avbryt / avsluta',
    exit: 'avsluta',
    'open $EDITOR (Alt+G fallback for VSCode/Cursor)': 'öppna $EDITOR (Alt+G som alternativ i VSCode/Cursor)',
    'redraw / repaint': 'rita om',
    'paste text; /paste attaches clipboard image': 'klistra in text; /paste bifogar bilden från urklipp',
    'discard draft (recall with ↑)': 'kasta utkastet (hämta tillbaka med ↑)',
    'apply completion': 'använd kompletteringen',
    'completions / queue edit / history': 'kompletteringar / redigera kön / historik',
    'open live session switcher (deletes queued message while editing)':
      'öppna sessionsväxlaren (tar bort köat meddelande vid redigering)',
    'expand live agents (keeps your draft)': 'visa aktiva agenter (behåller utkastet)',
    'collapse / restore live agent preview': 'fäll ihop / återställ förhandsvisningen av aktiva agenter',
    'open model picker (keeps your draft; applies to next turn mid-stream)':
      'öppna modellväljaren (behåller utkastet; gäller nästa tur under strömning)',
    'home / end of line': 'radens början / slut',
    'undo / redo input edits': 'ångra / gör om textändringar',
    'delete word': 'ta bort ord',
    'kill to line start / end (repeat across lines)': 'ta bort till radens början / slut (upprepa över flera rader)',
    'jump word': 'hoppa ett ord',
    'start / end of line': 'radens början / slut',
    'insert newline': 'infoga radbrytning',
    'multi-line continuation (fallback)': 'fortsätt på nästa rad (alternativ)',
    'run a shell command (e.g. !ls, !git status)': 'kör ett skalkommando (t.ex. !ls, !git status)',
    'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")':
      'infoga skalutdata i texten (t.ex. "grenen är {!git branch --show-current}")'
  },
  help: {
    title: '? snabbhjälp',
    hint: '  ·  skriv /help för hela panelen  ·  backsteg för att stänga',
    commands: 'Vanliga kommandon',
    hotkeys: 'Kortkommandon',
    full: 'hela listan med kommandon och kortkommandon',
    clear: 'starta en ny session',
    resume: 'växla mellan aktiva eller återuppta tidigare sessioner',
    details: 'ange samtalsutskriftens detaljnivå',
    copy: 'kopiera markeringen eller agentens senaste meddelande',
    quit: 'avsluta Hermes'
  },
  secrets: {
    sudo: 'sudo-lösenord krävs',
    forVariable: (name: string) => `för ${name}`,
    unlock: (name: string) => `Lås upp ${name} för den här sessionen`,
    hint: 'huvudlösenord · dolt · skickas endast till hanterarens CLI · Esc behåller låsningen'
  },
  setup: {
    title: 'Konfiguration krävs',
    description: 'Hermes behöver en modellleverantör innan TUI:n kan starta en session.',
    model: 'konfigurera leverantör och modell här',
    wizard: 'kör hela installationsguiden här',
    exit: 'avsluta och kör `hermes setup` manuellt',
    actions: 'Åtgärder'
  },
  approval: {
    always: 'Tillåt alltid',
    deny: 'Neka',
    once: 'Tillåt en gång',
    session: 'Tillåt under sessionen',
    required: (description: string) => `⚠ godkännande krävs · ${description}`,
    overflow: (count: number) => `… +${count} ${count === 1 ? 'rad' : 'rader'} till (hela texten finns ovan)`,
    hint: (count: number) => `↑/↓ välj · Enter bekräfta · 1–${count} snabbval · Esc/Ctrl+C neka`
  },
  clarify: {
    ask: 'fråga',
    questions: (count: number) => `${count} ${count === 1 ? 'fråga' : 'frågor'}`,
    skipped: '(överhoppad)',
    other: 'Annat (skriv ditt svar)',
    confirmContinue: 'bekräfta och fortsätt',
    lock: 'lås svaret',
    typingHint: (action: string) => `Enter ${action} · Esc tillbaka`,
    batchHint: (action: string) => `↑/↓ välj · Enter ${action} · Tab/Shift+Tab byt fråga · Esc/Ctrl+C avbryt`,
    answered: (count: number, total: number) => `${count}/${total} besvarade`,
    inputHint: (back: boolean) => `Enter skicka · Esc ${back ? 'tillbaka' : 'avbryt'}`,
    clipboardHint: 'Cmd+C kopiera · Cmd+V klistra in · Ctrl+C avbryt',
    cancelHint: 'Ctrl+C avbryt',
    hint: (count: number) => `↑/↓ välj · Enter bekräfta · 1–${count} snabbval · Esc/Ctrl+C avbryt`
  },
  confirm: { no: 'Nej', yes: 'Ja', hint: '↑/↓ välj · Enter bekräfta · Y/N snabbval · Esc avbryt' },
  agents: {
    queued: 'Köat för underagenten – tillämpas vid nästa verktygsgräns.',
    notQueued: 'Inte köat: underagenten är klar eller tar inte längre emot vägledning.',
    error: (error: string) => `Inte köat: ${error}`,
    steer: (id: string) => `Vägled ${id}`,
    guidance: 'Vägledningen köas till nästa verktygsgräns. Det pågående arbetet avbryts inte.',
    queueing: 'Köar …',
    hint: 'Enter köa · Esc tillbaka · utkastet i huvudfältet behålls',
    loading: 'Läser in samtalet i realtid …',
    lastLines: '[senaste 16 KiB]',
    unavailable:
      'Samtalet är inte tillgängligt i realtid. Underagenten kan vara klar. Förlopp och utdata finns kvar nedan.',
    refreshFailed: 'Kunde inte uppdatera samtalet i realtid.',
    live: 'Samtal i realtid'
  },
  queue: {
    title: (count: number) => `köade (${count})`,
    editing: (index: number) => ` · redigerar ${index} · Ctrl+X ta bort · Esc avbryt`,
    more: (count: number) => `… och ${count} till`
  }
}
