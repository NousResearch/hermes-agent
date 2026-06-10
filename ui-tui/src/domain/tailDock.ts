export type TranscriptTailSlot = 'assistant' | 'queue' | 'todos'

export interface TranscriptTailState {
  assistant: boolean
  queue: boolean
  todos: boolean
}

export const transcriptTailSlots = ({ assistant, queue, todos }: TranscriptTailState): TranscriptTailSlot[] => {
  const slots: TranscriptTailSlot[] = []

  if (queue) {
    slots.push('queue')
  }

  if (todos) {
    slots.push('todos')
  }

  if (assistant) {
    slots.push('assistant')
  }

  return slots
}
