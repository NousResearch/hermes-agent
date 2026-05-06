import { buildSync } from '../node_modules/esbuild/lib/main.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const result = buildSync({
  entryPoints: [path.join(__dirname, '../src/entry-exports.ts')],
  bundle: true,
  platform: 'node',
  format: 'esm',
  packages: 'external',
  outdir: path.join(__dirname, '../dist'),
});

if (result.errors.length > 0) {
  console.error('Build failed:', result.errors);
  process.exit(1);
}

console.log('Build complete.');
