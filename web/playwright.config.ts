import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './browser-tests',
  use: { baseURL: 'http://127.0.0.1:4179' },
  projects: [{ name: 'chromium', use: { browserName: 'chromium', launchOptions: { executablePath: process.env.CHROMIUM_EXECUTABLE } } }, { name: 'webkit', use: { browserName: 'webkit' } }],
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4179 --strictPort',
    url: 'http://127.0.0.1:4179/browser-tests/terminal.html',
    reuseExistingServer: false,
  },
});
