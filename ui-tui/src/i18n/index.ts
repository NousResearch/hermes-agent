import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { en, type Translations } from './en.js'
import { sv } from './sv.js'

export type TuiLanguage = 'en' | 'sv'
export const $tuiLanguage = atom<TuiLanguage>('en')

export function setTuiLanguage(value: unknown): void {
  const language = typeof value === 'string' ? value.trim().toLowerCase().replaceAll('_', '-') : ''
  $tuiLanguage.set(language === 'svenska' || language === 'swedish' || language.split('-')[0] === 'sv' ? 'sv' : 'en')
}

export function getTranslations(): Translations {
  return $tuiLanguage.get() === 'sv' ? sv : en
}

export function useTranslations(): Translations {
  return useStore($tuiLanguage) === 'sv' ? sv : en
}
