import type { SessionListItem } from '../gatewayTypes.js'

export const RESUME_PAGE_SIZE = 10

export interface ResumeViewport {
  end: number
  offset: number
  page: number
  totalPages: number
}

export const clampResumeSelection = (selected: number, count: number) => {
  if (count <= 0) {
    return 0
  }

  return Math.max(0, Math.min(selected, count - 1))
}

export const pageForSelection = (selected: number, pageSize = RESUME_PAGE_SIZE) =>
  Math.floor(Math.max(0, selected) / Math.max(1, pageSize)) + 1

export const moveResumeSelection = (selected: number, delta: number, count: number) =>
  clampResumeSelection(selected + delta, count)

export const buildResumeViewport = (
  items: SessionListItem[],
  selected: number,
  pageSize = RESUME_PAGE_SIZE
): ResumeViewport => {
  const safePageSize = Math.max(1, pageSize)

  if (!items.length) {
    return {
      end: 0,
      offset: 0,
      page: 1,
      totalPages: 1
    }
  }

  const clamped = clampResumeSelection(selected, items.length)
  const offset = Math.floor(clamped / safePageSize) * safePageSize

  return {
    end: Math.min(offset + safePageSize, items.length),
    offset,
    page: pageForSelection(clamped, safePageSize),
    totalPages: Math.max(1, Math.ceil(items.length / safePageSize))
  }
}
