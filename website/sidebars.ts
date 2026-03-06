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
          label: 'Features',
          items: [
            'user-guide/features/tools',
            'user-guide/features/skills',
            'user-guide/features/memory',
            'user-guide/features/context-files',
            'user-guide/features/personality',
            'user-guide/features/mcp',
            'user-guide/features/cron',
            'user-guide/features/hooks',
            'user-guide/features/delegation',
            'user-guide/features/code-execution',
            'user-guide/features/browser',
            'user-guide/features/image-generation',
            'user-guide/features/vision',
            'user-guide/features/tts',
            'user-guide/features/provider-routing',
            'user-guide/features/honcho',
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
