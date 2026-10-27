const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const roots = [process.env.HERMES_HOME, path.join(os.homedir(), '.hermes'), process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'hermes')].filter(Boolean)
const home = process.env.HERMES_HOME || roots.find(p => fs.existsSync(p))
const checkout = process.env.HERMES_CHECKOUT || (home && path.join(home, 'hermes-agent'))
const nodePaths = [path.join(__dirname, 'node_modules'), checkout && path.join(checkout, 'node_modules')].filter(Boolean)
const dependency = name => require(require.resolve(name, { paths: [__dirname, ...(checkout ? [checkout] : [])] }))
module.exports = { home, checkout, nodePaths, dependency }

