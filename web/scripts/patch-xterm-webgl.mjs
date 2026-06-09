#!/usr/bin/env node
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const packagePath = join(root, 'node_modules', '@xterm', 'addon-webgl', 'package.json');

if (!existsSync(packagePath)) {
  console.warn('[patch-xterm-webgl] @xterm/addon-webgl is not installed; skipping');
  process.exit(0);
}

const pkg = JSON.parse(readFileSync(packagePath, 'utf8'));
if (pkg.version !== '0.19.0') {
  console.warn(`[patch-xterm-webgl] Expected @xterm/addon-webgl 0.19.0, found ${pkg.version}; skipping`);
  process.exit(0);
}

const bundledFiles = [
  join(root, 'node_modules', '@xterm', 'addon-webgl', 'lib', 'addon-webgl.mjs'),
  join(root, 'node_modules', '@xterm', 'addon-webgl', 'lib', 'addon-webgl.js'),
];

const bundlePattern = /([A-Za-z_$][\w$]*)\.texImage2D\(\1\.TEXTURE_2D,0,\1\.RGBA,\1\.RGBA,\1\.UNSIGNED_BYTE,([^)]*?\.canvas)\),\1\.generateMipmap\(\1\.TEXTURE_2D\),/g;

for (const file of bundledFiles) {
  if (!existsSync(file)) {
    throw new Error(`[patch-xterm-webgl] Expected bundle file missing: ${file}`);
  }

  let source = readFileSync(file, 'utf8');
  if (!source.includes('generateMipmap')) {
    console.log(`[patch-xterm-webgl] ${file} already patched`);
    continue;
  }

  const patched = source.replace(
    bundlePattern,
    (_match, gl, canvasExpr) =>
      `${gl}.texImage2D(${gl}.TEXTURE_2D,0,${gl}.RGBA,${gl}.RGBA,${gl}.UNSIGNED_BYTE,${canvasExpr}),` +
      `${gl}.texParameteri(${gl}.TEXTURE_2D,${gl}.TEXTURE_MIN_FILTER,${gl}.LINEAR),` +
      `${gl}.texParameteri(${gl}.TEXTURE_2D,${gl}.TEXTURE_MAG_FILTER,${gl}.LINEAR),`,
  );

  if (patched === source || patched.includes('generateMipmap')) {
    throw new Error(`[patch-xterm-webgl] Failed to remove generateMipmap from ${file}`);
  }

  writeFileSync(file, patched);
  console.log(`[patch-xterm-webgl] patched ${file}`);
}

const sourceFile = join(root, 'node_modules', '@xterm', 'addon-webgl', 'src', 'GlyphRenderer.ts');
if (existsSync(sourceFile)) {
  const source = readFileSync(sourceFile, 'utf8');
  const patched = source.replace(
    '    gl.generateMipmap(gl.TEXTURE_2D);',
    '    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);\n    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);',
  );
  if (patched !== source) {
    writeFileSync(sourceFile, patched);
    console.log(`[patch-xterm-webgl] patched ${sourceFile}`);
  }
}

console.log('[patch-xterm-webgl] xterm WebGL glyph atlas mipmaps disabled');
