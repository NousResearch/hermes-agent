import { findSlashCommand } from './registry.js'
import type { SlashCommand } from './types.js'

/**
 * Stable catalog identity consumed by the Ink client projection.
 *
 * `command_id` and `execution_owner` are present on commands.catalog.v2.
 * Older gateway peers expose only `name`, so the canonical slash name and
 * current static-handler ownership remain the explicit mixed-version fallback.
 */
export interface InkCatalogCommandIdentity {
  command_id?: null | string
  execution_owner?: null | string
  name: string
}

/** Execution-only binding contributed by the Ink TUI client. */
export interface InkClientCommandBinding {
  readonly canonicalName: string
  readonly commandId: string
  readonly run: SlashCommand['run']
}

function normalizedCommandName(name: string): string {
  return name.trim().replace(/^\/+/, '').toLowerCase()
}

function stableCommandId(identity: InkCatalogCommandIdentity, canonicalName: string): string {
  if (identity.command_id == null) {
    return canonicalName
  }

  const commandId = identity.command_id.trim()

  if (!commandId) {
    throw new Error(`Ink command binding for "${canonicalName}" has an empty command_id`)
  }

  return commandId
}

function isClientOwned(identity: InkCatalogCommandIdentity): boolean {
  if (identity.execution_owner == null) {
    return true
  }

  const owner = identity.execution_owner.trim().toLowerCase()

  if (!owner) {
    throw new Error(`Ink command binding for "${identity.name}" has an empty execution_owner`)
  }

  return owner === 'client'
}

/**
 * Resolve one catalog row into an Ink-owned client binding.
 *
 * Explicit v2 server/plugin/skill/agent-turn ownership refuses a local binding;
 * those commands must settle through commands.invoke. A missing owner is the
 * compatibility contract for the current v1 catalog and preserves the existing
 * local Ink handler while mixed-version peers remain supported.
 */
export function resolveInkClientCommandBinding(
  identity: InkCatalogCommandIdentity
): InkClientCommandBinding | null {
  if (!isClientOwned(identity)) {
    return null
  }

  const enteredName = normalizedCommandName(identity.name)

  if (!enteredName) {
    return null
  }

  const command = findSlashCommand(enteredName)

  if (!command) {
    return null
  }

  const canonicalName = `/${command.name.toLowerCase()}`

  return Object.freeze({
    canonicalName,
    commandId: stableCommandId(identity, canonicalName),
    run: command.run
  })
}

/**
 * Project a catalog into immutable Ink client bindings.
 *
 * Duplicate stable ids or duplicate canonical commands are authority
 * collisions and fail closed rather than allowing catalog order or aliases to
 * choose an execution owner.
 */
export function projectInkClientCommandBindings(
  identities: readonly InkCatalogCommandIdentity[]
): readonly InkClientCommandBinding[] {
  const bindings: InkClientCommandBinding[] = []
  const byCommandId = new Map<string, string>()
  const byCanonicalName = new Map<string, string>()

  for (const identity of identities) {
    const binding = resolveInkClientCommandBinding(identity)

    if (!binding) {
      continue
    }

    const idKey = binding.commandId.toLowerCase()
    const idOwner = byCommandId.get(idKey)

    if (idOwner) {
      throw new Error(
        `Ink command_id collision: "${binding.commandId}" is bound by both "${idOwner}" and "${binding.canonicalName}"`
      )
    }

    const canonicalOwner = byCanonicalName.get(binding.canonicalName)

    if (canonicalOwner) {
      throw new Error(
        `Ink canonical command collision: "${binding.canonicalName}" is projected by both "${canonicalOwner}" and "${identity.name}"`
      )
    }

    byCommandId.set(idKey, binding.canonicalName)
    byCanonicalName.set(binding.canonicalName, identity.name)
    bindings.push(binding)
  }

  return Object.freeze(bindings)
}
