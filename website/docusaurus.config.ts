import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Ares',
  tagline: 'An evidence-native, Hermes-compatible AI workbench',
  favicon: 'img/favicon.ico',

  // The deployment workflow publishes this fork under the repository Pages
  // path.  Do not point the Ares build at the upstream Hermes site.
  url: 'https://recursiveintell.github.io',
  baseUrl: '/hermes-agent/docs/',

  organizationName: 'RecursiveIntell',
  projectName: 'hermes-agent',

  onBrokenLinks: 'warn',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      en: {
        label: 'English',
      },
      'zh-Hans': {
        label: '简体中文',
        htmlLang: 'zh-Hans',
      },
    },
  },

  themes: [
    '@docusaurus/theme-mermaid',
  ],

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        // Static-host redirects for renamed doc pages (GitHub Pages can't
        // do server-side redirects). Paths are relative to baseUrl (/docs/).
        redirects: [
          {
            // Renamed in #44470 (Automation Blueprints terminology rebrand)
            from: '/guides/automation-templates',
            to: '/guides/automation-blueprints',
          },
          {
            // Moved when the Plugins subcategory was created under
            // Developer Guide > Extending (docs restructure, July 2026)
            from: '/guides/build-a-hermes-plugin',
            to: '/developer-guide/plugins',
          },
          {
            // Users guess these short paths from abbreviated links and hit
            // raw 404s (consumer-onboarding audit finding #1, Aug 2026).
            from: '/quickstart',
            to: '/getting-started/quickstart',
          },
          {
            from: '/installation',
            to: '/getting-started/installation',
          },
        ],
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',  // Docs at the root of /docs/
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/RecursiveIntell/hermes-agent/edit/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/hermes-agent-banner.png',
    // Algolia DocSearch (replaces @easyops-cn/docusaurus-search-local).
    // The local plugin shipped a ~16 MB client-side lunr index that every
    // visitor downloaded and hydrated before their first result; DocSearch
    // answers from Algolia's servers with no client index at all. These are
    // public search-only credentials — safe to commit (the admin key is not
    // in the repo). Index is populated by the Algolia Crawler configured at
    // crawler.algolia.com; contextualSearch scopes results to the active
    // locale via the docusaurus_tag/lang facets the crawler records carry.
    algolia: {
      appId: '2JLBVEYZN5',
      apiKey: '8fda2a49223ce185ac30c2dbf6898a07',
      indexName: 'hermes docs',
      contextualSearch: true,
    },
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    navbar: {
      title: 'Ares',
      logo: {
        alt: 'Ares',
        src: 'img/ares-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/skills',
          label: 'Skills',
          position: 'left',
        },

        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/RecursiveIntell/hermes-agent#readme',
          label: 'Ares',
          position: 'right',
        },
        {
          href: 'https://github.com/RecursiveIntell/hermes-agent',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://github.com/NousResearch/hermes-agent',
          label: 'Upstream Hermes',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            { label: 'Ares Distribution', to: '/ares' },
            { label: 'Getting Started', to: '/getting-started/quickstart' },
            { label: 'User Guide', to: '/user-guide/cli' },
            { label: 'Developer Guide', to: '/developer-guide/architecture' },
            { label: 'Reference', to: '/reference/cli-commands' },
          ],
        },
        {
          title: 'Source',
          items: [
            { label: 'Ares Issues', href: 'https://github.com/RecursiveIntell/hermes-agent/issues' },
            { label: 'Ares Repository', href: 'https://github.com/RecursiveIntell/hermes-agent' },
            { label: 'Upstream Hermes', href: 'https://github.com/NousResearch/hermes-agent' },
          ],
        },
        {
          title: 'Provenance',
          items: [
            { label: 'Ares README', href: 'https://github.com/RecursiveIntell/hermes-agent#readme' },
            { label: 'RecursiveIntell', href: 'https://github.com/RecursiveIntell' },
            { label: 'Hermes Agent upstream', href: 'https://github.com/NousResearch/hermes-agent' },
          ],
        },
      ],
      copyright: `Ares is a downstream distribution by <a href="https://github.com/RecursiveIntell">RecursiveIntell</a>, derived from <a href="https://github.com/NousResearch/hermes-agent">Hermes Agent</a> · MIT License · ${new Date().getFullYear()}`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'yaml', 'json', 'python', 'toml'],
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
