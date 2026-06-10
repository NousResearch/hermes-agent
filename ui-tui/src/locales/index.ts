export interface LocaleText {
  common: {
    loading: string;
    error: string;
    cancel: string;
    confirm: string;
    submit: string;
    save: string;
    delete: string;
    edit: string;
    close: string;
  };
  menu: {
    help: string;
    exit: string;
    settings: string;
    newSession: string;
    resumeSession: string;
    availableTools: string;
    availableSkills: string;
    systemPrompt: string;
    mcpServers: string;
  };
  statusBar: {
    online: string;
    offline: string;
    connecting: string;
    disconnected: string;
  };
  messages: {
    welcome: string;
    sessionStarted: string;
    sessionEnded: string;
    copiedToClipboard: string;
    helpForCommands: string;
    scanningSkills: string;
    noSystemPrompt: string;
    response: string;
    thinking: string;
    session: string;
    longMessage: string;
    emptyToolResult: string;
    andMoreCategories: string;
    andMoreToolsets: string;
    tool: string;
    tools: string;
    skill: string;
    skills: string;
    commitsBehind: string;
    commitBehind: string;
    runToUpdate: string;
    noActiveSession: string;
    noTitleSet: string;
    noConversationYet: string;
    nothingToUndo: string;
    nothingToRetry: string;
    noStatus: string;
    failedToSave: string;
    copied: string;
    characters: string;
    clipboardCopyFailed: string;
    nothingToCopy: string;
    usage: string;
    toolCalls: string;
    progress: string;
    spawned: string;
    spawnTree: string;
    activity: string;
  };
  errors: {
    connectionFailed: string;
    timeout: string;
    unknownError: string;
  };
  commands?: Record<string, string>;
  categories?: Record<string, string>;
}

export type Locale = 'en' | 'zh';

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { en } from './en.js'
import { zh } from './zh.js'

const locales: Record<Locale, LocaleText> = {
  en,
  zh,
};

export const $localeAtom = atom<Locale>('en')

export function registerLocale(locale: Locale, texts: LocaleText): void {
  locales[locale] = texts;
}

export function setLocale(locale: Locale): void {
  if (locales[locale]) {
    $localeAtom.set(locale)
  } else {
    $localeAtom.set('en')
  }
}

export function getLocale(): Locale {
  return $localeAtom.get()
}

export function t(): LocaleText {
  return locales[$localeAtom.get()] || locales['en']
}

export function useLocale(): { locale: Locale; t: LocaleText } {
  const locale = useStore($localeAtom)
  return {
    locale,
    t: locales[locale] || locales['en'],
  }
}
