import type { Key } from '@hermes/ink'

import { t } from '../i18n/runtime.js'

import type { ComposerActions, InputHandlerActions, InputHandlerContext } from './interfaces.js'
import { patchUiState } from './uiStore.js'

// Keep journal-first queue actions separate from the global input routing phases.
export function discardEditedQueueItem(actions: ComposerActions, index: number) {
  try {
    actions.removeQueue(index)
  } catch (error) {
    patchUiState({ status: t('canonical.queue.notSavedQueueKept', error instanceof Error ? error.message : String(error)) })

    return
  }

  return actions.clearIn()
}

export function retryQueuedInput(
  key: Key,
  ch: string,
  composer: InputHandlerContext['composer'],
  sid: null | string,
  dispatchSubmission: InputHandlerActions['dispatchSubmission']
) {
  if (key.meta && !key.ctrl && key.super !== true && ch.toLowerCase() === 'k' && composer.refs.queueRef.current.length && sid) {
    const next = composer.actions.dequeue(true)

    if (next) {
      composer.actions.setQueueEdit(null)
      dispatchSubmission(next)
    }
  }
}
