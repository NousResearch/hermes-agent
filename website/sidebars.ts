import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/quickstart',
        'getting-started/installation',
        'getting-started/updating',
        'getting-started/learning-path',
      ],
    },
    {
      type: 'category',
      label: 'Guides & Tutorials',
      collapsed: false,
      items: [
        'guides/tips',
        'guides/daily-briefing-bot',
        'guides/team-telegram-assistant',
        'guides/python-library',
      ],
    },
    {
      type: 'category',
      label: 'User Guide',
      collapsed: false,
      items: [
        'user-guide/cli',
        'user-guide/configuration',
        'user-guide/sessions',
        'user-guide/security',
        {
          type: 'category',
          label: 'Messaging Gateway',
          items: [
            'user-guide/messaging/index',
            'user-guide/messaging/telegram',
            'user-guide/messaging/discord',
            'user-guide/messaging/slack',
            'user-guide/messaging/whatsapp',
            'user-guide/messaging/homeassistant',
          ],
        },
        {
          type: 'category',
          label: 'Core Features',
          items: [
            'user-guide/features/tools',
            'user-guide/features/skills',
            'user-guide/features/memory',
            'user-guide/features/context-files',
            'user-guide/features/personality',
          ],
        },
        {
          type: 'category',
          label: 'Automation',
          items: [
            'user-guide/features/cron',
            'user-guide/features/delegation',
            'user-guide/features/code-execution',
            'user-guide/features/hooks',
          ],
        },
        {
          type: 'category',
          label: 'Web & Media',
          items: [
            'user-guide/features/browser',
            'user-guide/features/vision',
            'user-guide/features/image-generation',
            'user-guide/features/tts',
          ],
        },
        {
          type: 'category',
          label: 'Integrations',
          items: [
            'user-guide/features/mcp',
            'user-guide/features/honcho',
            'user-guide/features/provider-routing',
          ],
        },
        {
          type: 'category',
          label: 'Advanced',
          items: [
            'user-guide/features/batch-processing',
            'user-guide/features/rl-training',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Developer Guide',
      items: [
        'developer-guide/architecture',
        'developer-guide/environments',
        'developer-guide/adding-tools',
        'developer-guide/creating-skills',
        'developer-guide/contributing',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli-commands',
        'reference/environment-variables',
        'reference/faq',
      ],
    },
  ],
};

export default sidebars;
