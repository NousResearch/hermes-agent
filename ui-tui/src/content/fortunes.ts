const FORTUNES = [
  'you are one clean refactor away from clarity',
  'a tiny rename today prevents a huge bug tomorrow',
  'your next commit message will be immaculate',
  'the edge case you are ignoring is already solved in your head',
  'minimal diff, maximal calm',
  'today favors bold deletions over new abstractions',
  'the right helper is already in your codebase',
  'you will ship before overthinking catches up',
  'tests are about to save your future self',
  'your instincts are correctly suspicious of that one branch'
]

const LEGENDARY = [
  'legendary drop: one-line fix, first try',
  'legendary drop: every flaky test passes cleanly',
  'legendary drop: your diff teaches by itself'
]

const hash = (s: string) => [...s].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261) >>> 0

const fromScore = (n: number) => {
  const rare = n % 20 === 0
  const bag = rare ? LEGENDARY : FORTUNES

  return `${rare ? '🌟' : '🔮'} ${bag[n % bag.length]}`
}

export const randomFortune = () => fromScore(Math.floor(Math.random() * 0x7fffffff))
export const dailyFortune = (seed: null | string) => fromScore(hash(`${seed || 'anon'}|${new Date().toDateString()}`))
