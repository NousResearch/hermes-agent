import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(new URL('../package.json', import.meta.url)));

const copies = [
  {
    from: 'node_modules/@nous-research/ui/dist/fonts',
    to: 'public/fonts',
  },
  {
    from: 'node_modules/@nous-research/ui/dist/assets',
    to: 'public/ds-assets',
  },
];

for (const { from, to } of copies) {
  const source = join(root, from);
  const destination = join(root, to);

  if (!existsSync(source)) {
    throw new Error(`Missing asset source: ${source}`);
  }

  rmSync(destination, { recursive: true, force: true });
  mkdirSync(dirname(destination), { recursive: true });
  cpSync(source, destination, { recursive: true });
}
