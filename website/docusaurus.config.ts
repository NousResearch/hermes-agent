import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import relativeDocLinks from './src/remark/relativeDocLinks';

const config: Config = {
  title: 'Hermes Agent',
  tagline: 'The self-improving AI agent',
  favicon: 'img/favicon.ico',

  url: 'https://hermes-agent.nousresearch.com',
  baseUrl: '/docs/',

  organizationName: 'NousResearch',
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
    // Static /plugins/<name> and /plugins/by/<author> pages generated from the catalog JSON.
    './plugins/plugin-catalog-pages',
    // "Ask AI" in the navbar, answered from these docs by each reader's own AI
    // (Codex, Claude Code, LM Studio, Ollama…). Runs next to Algolia DocSearch.
    [
      '@leuria/docusaurus',
      {
        suggestions: [
          'How do I install Hermes?',
          'How do I connect Hermes to Telegram?',
          'How do I use MCP servers with Hermes?',
        ],
      },
    ],
    [
      '@docusaurus/plugin-client-redirects',
      {
        // Static-host redirects for renamed doc pages (GitHub Pages can't
        // do server-side redirects). Paths are relative to baseUrl (/docs/).
        redirects: [
          {
            // Skill pages that moved from bundled/ to optional/ when the skill
            // graduated (08a7c25eec). The bundled URLs still rank and inbound
            // links still point here, so keep them working.
            from: '/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode',
            to: '/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-opencode',
          },
          {
            from: '/user-guide/skills/bundled/creative/creative-design-md',
            to: '/user-guide/skills/optional/creative/creative-design-md',
          },
          {
            from: '/user-guide/skills/bundled/media/media-gif-search',
            to: '/user-guide/skills/optional/media/media-gif-search',
          },
          {
            from: '/user-guide/skills/bundled/media/media-songsee',
            to: '/user-guide/skills/optional/media/media-songsee',
          },
          {
            from: '/user-guide/skills/bundled/productivity/productivity-box',
            to: '/user-guide/skills/optional/productivity/productivity-box',
          },
          {
            from: '/user-guide/skills/bundled/research/research-llm-wiki',
            to: '/user-guide/skills/optional/research/research-llm-wiki',
          },
          {
            from: '/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring',
            to: '/user-guide/skills/optional/software-development/software-development-hermes-agent-skill-authoring',
          },
          {
            from: '/user-guide/skills/bundled/software-development/software-development-inspecting-hermes-desktop-dom',
            to: '/user-guide/skills/optional/software-development/software-development-inspecting-hermes-desktop-dom',
          },
          {
            from: '/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger',
            to: '/user-guide/skills/optional/software-development/software-development-node-inspect-debugger',
          },
          {
            from: '/user-guide/skills/bundled/software-development/software-development-python-debugpy',
            to: '/user-guide/skills/optional/software-development/software-development-python-debugpy',
          },
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
          editUrl: 'https://github.com/NousResearch/hermes-agent/edit/main/website/',
          // Relative `.md` links (readable on GitHub, #114428) must also resolve
          // across the zh-Hans fallback boundary; see src/remark/relativeDocLinks.js.
          beforeDefaultRemarkPlugins: [[relativeDocLinks, {siteDir: __dirname}]],
        },
        blog: false,
        sitemap: {
          // The public host 308s /docs/plugins/* to /marketplace/* and
          // robots.txt disallows /docs/search, so neither belongs in the
          // sitemap; /docs/skills is a redirect to /docs/user-guide/skills.
          ignorePatterns: ['/docs/plugins/**', '/docs/search', '/docs/skills*'],
        },
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
      apiKey: '9629ec26628d1a126535fd5ef408990d',
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
      title: 'Hermes Agent',
      logo: {
        alt: 'Hermes Agent',
        src: 'img/logo.png',
        srcDark: 'img/logo-dark.png',
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
          to: '/plugins',
          label: 'Plugins',
          position: 'left',
        },
        {
          href: 'https://hermes-agent.nousresearch.com/',
          label: 'Download',
          position: 'left',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://hermes-agent.nousresearch.com',
          label: 'Home',
          position: 'right',
        },
        {
          href: 'https://github.com/NousResearch/hermes-agent',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://discord.gg/NousResearch',
          label: 'Discord',
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
            { label: 'Getting Started', to: '/getting-started/quickstart' },
            { label: 'User Guide', to: '/user-guide/cli' },
            { label: 'Developer Guide', to: '/developer-guide/architecture' },
            { label: 'Reference', to: '/reference/cli-commands' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Discord', href: 'https://discord.gg/NousResearch' },
            { label: 'GitHub Issues', href: 'https://github.com/NousResearch/hermes-agent/issues' },
            { label: 'Skills Hub', href: 'https://agentskills.io' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Desktop Download', href: 'https://hermes-agent.nousresearch.com/' },
            { label: 'Compare Hermes Agent', href: 'https://hermes-agent.nousresearch.com/compare' },
            { label: 'GitHub', href: 'https://github.com/NousResearch/hermes-agent' },
            { label: 'Nous Research', href: 'https://nousresearch.com' },
          ],
        },
      ],
      copyright: `Built by <a href="https://nousresearch.com">Nous Research</a> · MIT License · ${new Date().getFullYear()}`,
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
