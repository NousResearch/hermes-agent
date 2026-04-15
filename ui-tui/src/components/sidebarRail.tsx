import { Box, NoSelect } from '@hermes/ink'

import type { Theme } from '../theme.js'
import type { WidgetSpec } from '../widgets.js'
import { WidgetHost } from '../widgets.js'

export function SidebarRail({ t, widgets, width }: { t: Theme; widgets: WidgetSpec[]; width: number }) {
  return (
    <NoSelect flexDirection="column" flexShrink={0} width={width}>
      <Box borderColor={t.color.bronze as any} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
        <WidgetHost region="sidebar" widgets={widgets} />
      </Box>
    </NoSelect>
  )
}
