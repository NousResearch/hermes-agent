import type { ProcessListItem } from '../gatewayTypes.js'

export const countRunningProcesses = (processes?: ProcessListItem[]) =>
  processes?.filter(process => process.status === 'running').length ?? 0

export const visibleBackgroundTaskCount = (bgTaskCount: number, backgroundProcessCount: number) =>
  Math.max(bgTaskCount, backgroundProcessCount)
