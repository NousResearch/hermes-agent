import React, {type ReactNode, useEffect} from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

const DEFAULT_LOCALE = 'en';
const SUPPORTED_LOCALES = ['en', 'zh-Hans'] as const;
const NON_DEFAULT_LOCALES = SUPPORTED_LOCALES.filter(
  (locale) => locale !== DEFAULT_LOCALE,
);
const LOCALE_STORAGE_KEY = 'hermes-agent-docs-locale';
const PENDING_LOCALE_STORAGE_KEY = 'hermes-agent-docs-pending-locale';

function normalizeBaseUrl(baseUrl: string): string {
  if (!baseUrl || baseUrl === '/') {
    return '';
  }

  return `/${baseUrl.replace(/^\/+|\/+$/g, '')}`;
}

function stripBaseUrl(pathname: string, baseUrl: string): string {
  const base = normalizeBaseUrl(baseUrl);

  if (!base) {
    return pathname || '/';
  }

  if (pathname === base) {
    return '/';
  }

  if (pathname.startsWith(`${base}/`)) {
    return pathname.slice(base.length) || '/';
  }

  return pathname || '/';
}

function getLocaleFromPath(pathname: string, baseUrl: string): string {
  const pathWithoutBase = stripBaseUrl(pathname, baseUrl);
  const firstSegment = pathWithoutBase.split('/').filter(Boolean)[0];

  return NON_DEFAULT_LOCALES.includes(firstSegment as (typeof NON_DEFAULT_LOCALES)[number])
    ? firstSegment
    : DEFAULT_LOCALE;
}

function getPathForLocale(
  pathname: string,
  search: string,
  hash: string,
  baseUrl: string,
  targetLocale: string,
): string {
  const base = normalizeBaseUrl(baseUrl);
  const pathWithoutBase = stripBaseUrl(pathname, baseUrl);
  const segments = pathWithoutBase.split('/').filter(Boolean);

  if (NON_DEFAULT_LOCALES.includes(segments[0] as (typeof NON_DEFAULT_LOCALES)[number])) {
    segments.shift();
  }

  const localizedSegments =
    targetLocale === DEFAULT_LOCALE ? segments : [targetLocale, ...segments];
  const localizedPath = `${base}/${localizedSegments.join('/')}`.replace(/\/+/g, '/');

  return `${localizedPath === '' ? '/' : localizedPath}${search}${hash}`;
}

function getPreferredBrowserLocale(): string {
  const browserLanguages = [
    ...(navigator.languages ?? []),
    navigator.language,
  ].filter(Boolean);

  for (const browserLanguage of browserLanguages) {
    const normalizedLanguage = browserLanguage.toLowerCase();

    if (normalizedLanguage === 'zh' || normalizedLanguage.startsWith('zh-')) {
      return 'zh-Hans';
    }

    if (normalizedLanguage === 'en' || normalizedLanguage.startsWith('en-')) {
      return 'en';
    }
  }

  return DEFAULT_LOCALE;
}

function persistLocaleChoiceFromLocaleLinks(baseUrl: string): () => void {
  const handleClick = (event: MouseEvent) => {
    const target = event.target;

    if (!(target instanceof Element)) {
      return;
    }

    const anchor = target.closest<HTMLAnchorElement>('a[href]');

    if (!anchor) {
      return;
    }

    const label = anchor.textContent?.trim();
    const isLocaleSwitcherLink = label === 'English' || label === '简体中文';

    if (!isLocaleSwitcherLink) {
      return;
    }

    const url = new URL(anchor.href, window.location.href);
    const targetLocale = getLocaleFromPath(url.pathname, baseUrl);

    window.localStorage.setItem(PENDING_LOCALE_STORAGE_KEY, targetLocale);
  };

  document.addEventListener('click', handleClick, true);

  return () => document.removeEventListener('click', handleClick, true);
}

export default function Root({children}: {children: ReactNode}): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const baseUrl = siteConfig.baseUrl;

  useEffect(() => {
    const cleanup = persistLocaleChoiceFromLocaleLinks(baseUrl);
    const currentLocale = getLocaleFromPath(window.location.pathname, baseUrl);
    const pendingLocale = window.localStorage.getItem(PENDING_LOCALE_STORAGE_KEY);

    if (pendingLocale === currentLocale) {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
      window.localStorage.removeItem(PENDING_LOCALE_STORAGE_KEY);
      return cleanup;
    }

    const storedLocale = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    const preferredLocale = storedLocale || getPreferredBrowserLocale();

    if (currentLocale !== preferredLocale) {
      const preferredPath = getPathForLocale(
        window.location.pathname,
        window.location.search,
        window.location.hash,
        baseUrl,
        preferredLocale,
      );

      if (preferredPath !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
        window.location.replace(preferredPath);
        return cleanup;
      }
    }

    window.localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
    return cleanup;
  }, [baseUrl]);

  return <>{children}</>;
}
