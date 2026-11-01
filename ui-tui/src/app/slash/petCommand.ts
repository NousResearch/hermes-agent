import { t } from '../../i18n/runtime.js'
import { rpcErrorMessage } from '../../lib/rpc.js'
import { patchOverlayState } from '../overlayStore.js'

import type { SlashRunCtx } from './types.js'

interface Gallery {
  active: string
  enabled: boolean
  pets: { slug: string; installed: boolean }[]
}

interface AdoptedPet {
  ok: boolean
  slug: string
  displayName?: string | null
}

async function selectPet(slug: string, ctx: SlashRunCtx): Promise<void> {
  const result = await ctx.gateway.gw.request<AdoptedPet>('pet.select', { slug })

  ctx.guarded<AdoptedPet>(pet => {
    ctx.ran?.()
    ctx.transcript.sys(t('slashCmd.session.pet.adopted', pet.displayName || pet.slug))
  })(result)
}

async function disablePet(ctx: SlashRunCtx): Promise<void> {
  const result = await ctx.gateway.gw.request<{ ok: boolean }>('pet.disable', {})

  ctx.guarded<{ ok: boolean }>(() => {
    ctx.ran?.()
    ctx.transcript.sys(t('slashCmd.session.pet.disabled'))
  })(result)
}

async function togglePet(ctx: SlashRunCtx): Promise<void> {
  // Metadata must come from the owner: a hidden pane does not imply the pet is disabled.
  const gallery = await ctx.gateway.gw.request<Gallery>('pet.gallery', { localOnly: true })

  if (ctx.stale() || !gallery) {
    return
  }

  if (gallery.enabled) {
    return disablePet(ctx)
  }

  const installed = gallery.pets.filter(pet => pet.installed)
  const slug = installed.find(pet => pet.slug === gallery.active)?.slug || installed[0]?.slug

  if (!slug) {
    return ctx.transcript.sys(t('slashCmd.session.pet.noneInstalled'))
  }

  return selectPet(slug, ctx)
}

export function runPetCommand(arg: string, ctx: SlashRunCtx): void {
  const parts = arg.trim().split(/\s+/)
  const verb = parts[0]!.toLowerCase()

  if (parts.length === 1 && ['list', 'gallery', 'browse', 'all'].includes(verb)) {
    patchOverlayState({ petPicker: true })
    ctx.ran?.()

    return
  }

  let action: Promise<void>

  if (parts.length === 1 && (!verb || verb === 'toggle')) {
    action = togglePet(ctx)
  } else if (parts.length === 1 && verb === 'off') {
    action = disablePet(ctx)
  } else if (verb === 'scale' && parts.length === 2) {
    // The existing owner writer validates numbers and clamps to the engine bounds.
    action = ctx.gateway.gw.request<{ ok: boolean; scale: number }>('pet.scale', { scale: parts[1] }).then(
      ctx.guarded(result => {
        ctx.ran?.()
        ctx.transcript.sys(t('slashCmd.session.pet.scaled', result.scale))
      })
    )
  } else if (parts.length === 1 && verb !== 'scale') {
    action = selectPet(parts[0]!, ctx)
  } else {
    ctx.transcript.sys(t('slashCmd.session.pet.usage'))

    return
  }

  action.catch((error: unknown) => {
    if (!ctx.stale()) {
      ctx.transcript.sys(t('slashCmd.session.pet.failed', rpcErrorMessage(error)))
    }
  })
}
