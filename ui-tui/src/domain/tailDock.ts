export type TranscriptTailSlot = 'assistant' | 'queue' | 'todos'

export interface TranscriptTailState {
  queue: boolean
  todos: boolean
}

export const transcriptTailSlots = ({ queue, todos }: TranscriptTailState): TranscriptTailSlot[] => {
  const slots: TranscriptTailSlot[] = []

  if (queue) {
    slots.push('queue')
  }

  if (todos) {
    slots.push('todos')
  }

  slots.push('assistant')

  return slots
}
