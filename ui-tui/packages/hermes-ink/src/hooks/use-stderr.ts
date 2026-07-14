import { useMemo } from 'react'
export type StderrHandle = {
  stderr: NodeJS.WriteStream
  write: (data: string) => boolean
}

export default function useStderr(): StderrHandle {
  return useMemo(
    () => ({
      stderr: process.stderr,
      write: data => process.stderr.write(data)
    }),
    []
  )
}
