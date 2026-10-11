// 目标与后台进程的展示文案。
export const activityEn = {
  goal: {
    turns: (used: unknown, max: unknown) => `${used}/${max} turns`,
    until: (time: unknown) => `until ${time}`,
    waiting: (type: unknown, target: unknown) => `on ${type} ${target}`,
    parked: 'goal parked',
    paused: 'goal paused',
    active: 'goal'
  },
  process: {
    last: (detail: unknown) => `last: ${detail}`,
    exit: (code: unknown) => `exit ${code}`,
    ago: (verdict: unknown, seconds: unknown) => `${verdict} · ${seconds}s ago`,
    background: 'background process',
    title: 'Processes',
    done: (count: unknown) => `${count} done`,
    lost: 'lost',
    killed: 'killed'
  }
}
