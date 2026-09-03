'use strict'

const fs = require('node:fs')
const path = require('node:path')

function fail(code) {
  process.stdout.write(JSON.stringify({
    ok: false,
    error: {code, message: 'Orca RPC request rejected'}
  }) + '\n')
  process.exitCode = 1
}

function plainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function validInput(input) {
  if (!plainObject(input) || typeof input.resourcesPath !== 'string' || !plainObject(input.params)) {
    return 'invalid_request'
  }
  if (input.method === 'accounts.list') {
    const keys = Object.keys(input.params)
    return keys.length === 1 && input.params.refreshUsage === false ? null : 'invalid_params'
  }
  if (input.method === 'accounts.selectCodexForTarget') {
    const accountId = input.params.accountId
    const target = input.params.target
    const validAccount = accountId === null || (typeof accountId === 'string' && accountId.length > 0)
    const validTarget = plainObject(target) && target.runtime === 'host' && target.wslDistro === null
    const targetKeys = validTarget ? Object.keys(target).sort().join(',') : ''
    const paramKeys = Object.keys(input.params).sort().join(',')
    return validAccount && validTarget && targetKeys === 'runtime,wslDistro' && paramKeys === 'accountId,target'
      ? null
      : 'invalid_params'
  }
  return 'invalid_method'
}

async function main() {
  let input
  try {
    input = JSON.parse(fs.readFileSync(0, 'utf8'))
  } catch {
    fail('invalid_request')
    return
  }
  const validationError = validInput(input)
  if (validationError) {
    fail(validationError)
    return
  }
  try {
    const runtimeClientPath = path.join(
      input.resourcesPath,
      'app.asar.unpacked',
      'out',
      'cli',
      'runtime-client.js'
    )
    const {RuntimeClient} = require(runtimeClientPath)
    const client = new RuntimeClient(undefined, 10000, null, null)
    const response = await client.call(input.method, input.params)
    process.stdout.write(JSON.stringify({ok: true, response}) + '\n')
  } catch {
    fail('runtime_error')
  }
}

void main()
