const path = require('path')
const { dependency, home, nodePaths } = require('./runtime.cjs')
const esbuild = dependency('esbuild')
if (process.argv.includes('--install') && !home) throw new Error('Set HERMES_HOME to the existing Hermes data directory.')
esbuild.buildSync({
  entryPoints: [path.join(__dirname, 'plugin.js')],
  absWorkingDir: __dirname,
  outfile: process.argv.includes('--install') ? path.join(home, 'desktop-plugins/word-peek/plugin.js') : path.join(__dirname, 'dist/plugin.js'),
  bundle: true, platform: 'browser', format: 'esm', target: 'chrome120',
  external: ['@hermes/plugin-sdk'],
  nodePaths,
  minify: false
})

