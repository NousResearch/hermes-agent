import { Box, Link, Text } from '@hermes/ink'
import type { MediaAttachment } from '@hermes/shared/gateway-events'

import type { Theme } from '../theme.js'

/** Files a reply delivered (the gateway parsed its `MEDIA:` tags): one
 *  `▸ path` row each, linked with file:// for local absolute paths. */
export function AttachmentRows({ attachments, t }: { attachments: readonly MediaAttachment[]; t: Theme }) {
  return (
    <Box flexDirection="column">
      {attachments.map(({ path }) => (
        <Text color={t.color.muted} key={path} wrap="wrap-trim">
          {'▸ '}

          <Link url={/^(?:\/|[a-z]:[\\/])/i.test(path) ? `file://${path}` : path}>
            <Text color={t.color.accent} underline>
              {path}
            </Text>
          </Link>
        </Text>
      ))}
    </Box>
  )
}
