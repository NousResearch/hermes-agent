import { Text } from '@hermes/ink'

import { Dialog, Overlay, type OverlayZone } from '../../components/overlay.js'
import { useI18n } from '../../i18n/index.js'
import { defineWidgetApp } from '../registry.js'
import { isCtrl } from '../types.js'

const ZONES: readonly OverlayZone[] = [
  'bottom',
  'bottom-left',
  'bottom-right',
  'center',
  'left',
  'right',
  'top',
  'top-left',
  'top-right'
]

const USAGE = `usage: /dialog-test [zone]   zones: ${ZONES.join(', ')}`

export interface DialogTestState {
  body: string
  hint?: string
  title?: string
  zone: OverlayZone
}

export const dialogTestApp = defineWidgetApp<DialogTestState>({
  id: 'dialog-test',
  help: 'open a sample dialog overlay with a faked backdrop',
  helpKey: 'widget.dialog.help',
  usage: USAGE,
  usageKey: 'widget.dialog.usage',

  init(arg) {
    const zone = (arg.trim().toLowerCase() || 'center') as OverlayZone

    if (!ZONES.includes(zone)) {
      return null
    }

    return { body: '', zone }
  },

  reduce(state, { ch, key }) {
    return key.escape || key.return || ch === 'q' || isCtrl(key, ch, 'c') ? null : state
  },

  render({ cols, state }) {
    return <DialogTestView cols={cols} state={state} />
  }
})

function DialogTestView({ cols, state }: { cols: number; state: DialogTestState }) {
  const { t: ti } = useI18n()
  const body = state.body || ti('widget.dialog.body', { zone: state.zone })

  return (
    <Overlay backdrop zone={state.zone}>
      <Dialog
        hint={state.hint ?? ti('widget.dialog.hint')}
        title={state.title ?? ti('widget.dialog.title')}
        width={Math.min(60, cols - 8)}
      >
        {body.split('\n').map((line, i) => (
          <Text key={i}>{line || ' '}</Text>
        ))}
      </Dialog>
    </Overlay>
  )
}
