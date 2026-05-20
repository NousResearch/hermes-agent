import { lastGrapheme } from './intl.js'
let earlyInputBuffer = ''
let isCapturing = false
let readableHandler: (() => void) | null = null

export function startCapturingEarlyInput(): void {
  if (!process.stdin.isTTY || isCapturing || process.argv.includes('-p') || process.argv.includes('--print')) {
    return
  }

  isCapturing = true
  earlyInputBuffer = ''

  try {
    process.stdin.setEncoding('utf8')
    process.stdin.setRawMode(true)
    process.stdin.ref()

    readableHandler = () => {
      let chunk = process.stdin.read()

      while (chunk !== null) {
        if (typeof chunk === 'string') {
          processChunk(chunk)
        }

        chunk = process.stdin.read()
      }
    }

    process.stdin.on('readable', readableHandler)
  } catch {
    isCapturing = false
  }
}

function processChunk(str: string): void {
  let i = 0

  while (i < str.length) {
    const char = str[i]!
    const code = char.charCodeAt(0)

    if (code === 3) {
      stopCapturingEarlyInput()
      process.exit(130)

      return
    }

    if (code === 4) {
      stopCapturingEarlyInput()

      return
    }

    if (code === 127 || code === 8) {
      if (earlyInputBuffer.length > 0) {
        const last = lastGrapheme(earlyInputBuffer)
        earlyInputBuffer = earlyInputBuffer.slice(0, -(last.length || 1))
      }

      i++

      continue
    }

    if (code === 27) {
      i++

      while (i < str.length && !(str.charCodeAt(i) >= 64 && str.charCodeAt(i) <= 126)) {
        i++
      }

      if (i < str.length) {
        i++
      }

      continue
    }

    if (code < 32 && code !== 9 && code !== 10 && code !== 13) {
      i++

      continue
    }

    if (code === 13) {
      earlyInputBuffer += '\n'
      i++

      continue
    }

    earlyInputBuffer += char
    i++
  }
}

export function stopCapturingEarlyInput(): void {
  if (!isCapturing) {
    return
  }

  isCapturing = false

  if (readableHandler) {
    process.stdin.removeListener('readable', readableHandler)
    readableHandler = null
  }
}

export function consumeEarlyInput(): string {
  stopCapturingEarlyInput()
  const input = earlyInputBuffer.trim()
  earlyInputBuffer = ''

  return input
}

export function hasEarlyInput(): boolean {
  return earlyInputBuffer.trim().length > 0
}

export function seedEarlyInput(text: string): void {
  earlyInputBuffer = text
}

export function isCapturingEarlyInput(): boolean {
  return isCapturing
}
