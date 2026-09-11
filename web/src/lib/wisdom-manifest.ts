export interface WisdomToolRequirement {
  name: string
  minimum_version: string | null
  auto_install: false
  requires_admin: boolean
}

export interface WisdomPluginRequirement {
  id: string
  minimum_version: string | null
  required: boolean
}

export interface WisdomSystemSpecification extends Record<string, unknown> {
  hermes: { minimum_version: string }
  platforms: string[]
  architectures: string[]
  model: {
    capabilities: string[]
    minimum_context_window: number | null
  }
  tools: WisdomToolRequirement[]
  plugins: WisdomPluginRequirement[]
  credentials: string[]
  connections: string[]
  filesystem: { read: string[]; write: string[] }
  network: { destinations: string[] }
  runtime: { shell: boolean; browser: boolean; code: boolean; sandbox: boolean }
  hardware: string[]
  known_limitations: string[]
}

export interface WisdomManifestV1 {
  schema_version: 1
  name: string
  requirements: WisdomSystemSpecification
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return value
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label} must be text`)
  return value
}

function requireNullableString(value: unknown, label: string): string | null {
  if (value === null) return null
  return requireString(value, label)
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be true or false`)
  return value
}

function requireStringList(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new Error(`${label} must be a list of text values`)
  }
  return value
}

function requireExactKeys(value: Record<string, unknown>, keys: string[], label: string): void {
  const expected = new Set(keys)
  const unexpected = Object.keys(value).filter(key => !expected.has(key))
  const missing = keys.filter(key => !(key in value))
  if (unexpected.length > 0) throw new Error(`${label} contains unsupported fields: ${unexpected.join(', ')}`)
  if (missing.length > 0) throw new Error(`${label} is missing fields: ${missing.join(', ')}`)
}

export function parseWisdomSystemSpecification(value: unknown): WisdomSystemSpecification {
  const specification = requireRecord(value, 'System Specification')
  requireExactKeys(
    specification,
    [
      'hermes',
      'platforms',
      'architectures',
      'model',
      'tools',
      'plugins',
      'credentials',
      'connections',
      'filesystem',
      'network',
      'runtime',
      'hardware',
      'known_limitations'
    ],
    'System Specification'
  )

  const hermes = requireRecord(specification.hermes, 'Hermes requirement')
  requireExactKeys(hermes, ['minimum_version'], 'Hermes requirement')
  const model = requireRecord(specification.model, 'Model requirement')
  requireExactKeys(model, ['capabilities', 'minimum_context_window'], 'Model requirement')
  if (model.minimum_context_window !== null && typeof model.minimum_context_window !== 'number') {
    throw new Error('Minimum context window must be a number or blank')
  }
  const filesystem = requireRecord(specification.filesystem, 'Filesystem requirement')
  requireExactKeys(filesystem, ['read', 'write'], 'Filesystem requirement')
  const network = requireRecord(specification.network, 'Network requirement')
  requireExactKeys(network, ['destinations'], 'Network requirement')
  const runtime = requireRecord(specification.runtime, 'Runtime requirement')
  requireExactKeys(runtime, ['shell', 'browser', 'code', 'sandbox'], 'Runtime requirement')

  if (!Array.isArray(specification.tools)) throw new Error('Tools must be a list')
  const tools = specification.tools.map((raw, index) => {
    const tool = requireRecord(raw, `Tool ${index + 1}`)
    requireExactKeys(tool, ['name', 'minimum_version', 'auto_install', 'requires_admin'], `Tool ${index + 1}`)
    if (tool.auto_install !== false) throw new Error(`Tool ${index + 1} cannot request automatic installation`)
    return {
      name: requireString(tool.name, `Tool ${index + 1} name`),
      minimum_version: requireNullableString(tool.minimum_version, `Tool ${index + 1} minimum version`),
      auto_install: false as const,
      requires_admin: requireBoolean(tool.requires_admin, `Tool ${index + 1} administrator requirement`)
    }
  })

  if (!Array.isArray(specification.plugins)) throw new Error('Plugins must be a list')
  const plugins = specification.plugins.map((raw, index) => {
    const plugin = requireRecord(raw, `Plugin ${index + 1}`)
    requireExactKeys(plugin, ['id', 'minimum_version', 'required'], `Plugin ${index + 1}`)
    return {
      id: requireString(plugin.id, `Plugin ${index + 1} ID`),
      minimum_version: requireNullableString(plugin.minimum_version, `Plugin ${index + 1} minimum version`),
      required: requireBoolean(plugin.required, `Plugin ${index + 1} requirement`)
    }
  })

  return {
    hermes: { minimum_version: requireString(hermes.minimum_version, 'Minimum Hermes version') },
    platforms: requireStringList(specification.platforms, 'Platforms'),
    architectures: requireStringList(specification.architectures, 'Architectures'),
    model: {
      capabilities: requireStringList(model.capabilities, 'Model capabilities'),
      minimum_context_window: model.minimum_context_window === null ? null : Number(model.minimum_context_window)
    },
    tools,
    plugins,
    credentials: requireStringList(specification.credentials, 'Credentials'),
    connections: requireStringList(specification.connections, 'Connections'),
    filesystem: {
      read: requireStringList(filesystem.read, 'Readable paths'),
      write: requireStringList(filesystem.write, 'Writable paths')
    },
    network: { destinations: requireStringList(network.destinations, 'Network destinations') },
    runtime: {
      shell: requireBoolean(runtime.shell, 'Shell requirement'),
      browser: requireBoolean(runtime.browser, 'Browser requirement'),
      code: requireBoolean(runtime.code, 'Code execution requirement'),
      sandbox: requireBoolean(runtime.sandbox, 'Sandbox requirement')
    },
    hardware: requireStringList(specification.hardware, 'Hardware'),
    known_limitations: requireStringList(specification.known_limitations, 'Known limitations')
  }
}

export function parseWisdomManifest(value: string): WisdomManifestV1 {
  const manifest = requireRecord(JSON.parse(value) as unknown, 'Manifest')
  requireExactKeys(manifest, ['schema_version', 'name', 'requirements'], 'Manifest')
  if (manifest.schema_version !== 1) throw new Error('Only manifest schema version 1 is supported')
  return {
    schema_version: 1,
    name: requireString(manifest.name, 'Skill name'),
    requirements: parseWisdomSystemSpecification(manifest.requirements)
  }
}

function boundedTextError(value: string, label: string): string | null {
  if (!value.trim()) return `${label} is required.`
  if (new TextEncoder().encode(value).length > 512) return `${label} must be 512 UTF-8 bytes or fewer.`
  return null
}

function listError(values: string[], label: string): string | null {
  if (values.length > 64) return `${label} can contain at most 64 entries.`
  for (const value of values) {
    const error = boundedTextError(value, `${label} entry`)
    if (error) return error
  }
  return null
}

export function wisdomSystemSpecificationValidationError(specification: WisdomSystemSpecification): string | null {
  const errors = [
    boundedTextError(specification.hermes.minimum_version, 'Minimum Hermes version'),
    specification.model.minimum_context_window !== null &&
    (!Number.isSafeInteger(specification.model.minimum_context_window) ||
      specification.model.minimum_context_window < 1)
      ? 'Minimum context window must be blank or a positive whole number.'
      : null,
    listError(specification.platforms, 'Platforms'),
    listError(specification.architectures, 'Architectures'),
    listError(specification.model.capabilities, 'Model capabilities'),
    listError(specification.credentials, 'Credentials'),
    listError(specification.connections, 'Connections'),
    listError(specification.filesystem.read, 'Readable paths'),
    listError(specification.filesystem.write, 'Writable paths'),
    listError(specification.network.destinations, 'Network destinations'),
    listError(specification.hardware, 'Hardware requirements'),
    listError(specification.known_limitations, 'Known limitations'),
    specification.tools.length > 64 ? 'Tools can contain at most 64 entries.' : null,
    specification.plugins.length > 64 ? 'Plugins can contain at most 64 entries.' : null
  ]
  for (const [index, tool] of specification.tools.entries()) {
    errors.push(boundedTextError(tool.name, `Tool ${index + 1} name`))
    if (tool.minimum_version !== null) errors.push(boundedTextError(tool.minimum_version, `Tool ${index + 1} version`))
  }
  for (const [index, plugin] of specification.plugins.entries()) {
    errors.push(boundedTextError(plugin.id, `Plugin ${index + 1} ID`))
    if (plugin.minimum_version !== null) {
      errors.push(boundedTextError(plugin.minimum_version, `Plugin ${index + 1} version`))
    }
  }
  return errors.find((error): error is string => !!error) ?? null
}

export function wisdomManifestValidationError(value: string): string | null {
  let manifest: WisdomManifestV1
  try {
    manifest = parseWisdomManifest(value)
  } catch (reason) {
    return reason instanceof Error ? reason.message : String(reason)
  }
  return (
    boundedTextError(manifest.name, 'Skill name') ?? wisdomSystemSpecificationValidationError(manifest.requirements)
  )
}
