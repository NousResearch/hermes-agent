import { Text } from '@hermes/ink'

import { Dialog, Overlay, type OverlayZone } from '../../components/overlay.js'
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

const defaultBody = (zone: OverlayZone) =>
  [
    'This is a viewport-level overlay with a backdrop.',
    '',
    `Zone: ${zone}`,
    'Try: /dialog-test top-right · bottom · left · ...'
  ].join('\n')

export const dialogTestApp = defineWidgetApp<DialogTestState>({
  id: 'dialog-test',
  usage: USAGE,

  init(arg) {
    const zone = (arg.trim().toLowerCase() || 'center') as OverlayZone

    if (!ZONES.includes(zone)) {
      return null
    }

    return { body: defaultBody(zone), hint: 'Esc/q/Enter close · Ctrl+C close', title: 'Dialog primitive', zone }
  },

  reduce(state, { ch, key }) {
    return key.escape || key.return || ch === 'q' || isCtrl(key, ch, 'c') ? null : state
  },

  render({ cols, state, t }) {
    void t

    return (
      <Overlay backdrop zone={state.zone}>
        <Dialog hint={state.hint ?? 'Esc/q close'} title={state.title} width={Math.min(60, cols - 8)}>
          {state.body.split('\n').map((line, i) => (
            <Text key={i}>{line || ' '}</Text>
          ))}
        </Dialog>
      </Overlay>
    )
  }
})

