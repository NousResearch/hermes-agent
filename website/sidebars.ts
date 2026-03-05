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
      ],
    },
    {
      type: 'category',
      label: 'User Guide',
      collapsed: false,
      items: [
        'user-guide/cli',
        'user-guide/configuration',
        {
          type: 'category',
          label: 'Messaging Gateway',
          items: [
            'user-guide/messaging/index',
            'user-guide/messaging/telegram',
            'user-guide/messaging/discord',
            'user-guide/messaging/slack',
            'user-guide/messaging/whatsapp',
          ],
        },
        {
          type: 'category',
          label: 'Features',
          items: [
            'user-guide/features/tools',
            'user-guide/features/skills',
            'user-guide/features/memory',
            'user-guide/features/mcp',
            'user-guide/features/cron',
            'user-guide/features/hooks',
            'user-guide/features/delegation',
            'user-guide/features/code-execution',
            'user-guide/features/tts',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Developer Guide',
      items: [
        'developer-guide/architecture',
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
      ],
    },
  ],
};

export default sidebars;
