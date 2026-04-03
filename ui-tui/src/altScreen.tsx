import { useEffect, type PropsWithChildren } from 'react'
import { Box, useStdout } from 'ink'

const ENTER = '\x1b[?1049h\x1b[2J\x1b[H'
const LEAVE = '\x1b[?1049l'

export function AltScreen({ children }: PropsWithChildren) {
  const { stdout } = useStdout()
  const rows = stdout?.rows ?? 24
  const cols = stdout?.columns ?? 80

  useEffect(() => {
    process.stdout.write(ENTER)

    const leave = () => process.stdout.write(LEAVE)
    process.on('exit', leave)

    return () => {
      leave()
      process.off('exit', leave)
    }
  }, [])

  return (
    <Box flexDirection="column" height={rows} width={cols} overflow="hidden">
      {children}
    </Box>
  )
}
