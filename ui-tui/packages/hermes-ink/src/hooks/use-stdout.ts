import { useMemo } from 'react'
export type StdoutHandle = {
  stdout: NodeJS.WriteStream
  write: (data: string) => boolean
}

export default function useStdout(): StdoutHandle {
  return useMemo(
    () => ({
      stdout: process.stdout,
      write: data => process.stdout.write(data)
    }),
    []
  )
}
