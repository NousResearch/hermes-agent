// 交互结果与错误反馈文案。
export const feedbackEn = {
  errors: {
    rpc: (message: unknown) => `error: ${message}`,
    invalidResponse: (method: unknown) => `error: invalid response: ${method}`
  },
  command: {
    ambiguous: (commands: unknown) => `ambiguous command: ${commands}`,
    noOutputParen: '(no output)',
    skillPayloadMissing: (command: unknown) => `/${command}: skill payload missing message`,
    emptyMessage: (command: unknown) => `/${command}: empty message`,
    noOutput: (command: unknown) => `/${command}: no output`
  },
  submission: {
    sessionNotReady: 'session not ready yet',
    shellExit: (code: unknown) => `exit ${code}`,
    steerRejectedQueued: 'steer rejected — message queued for next turn',
    steerFailedQueued: 'steer failed — message queued for next turn'
  },
  sys: { on: 'on', off: 'off' },
  usage: {
    bar: {
      planFallback: 'plan',
      percentUsed: (percent: unknown) => `${percent}% used`,
      remainingOfTotal: (remaining: unknown, total: unknown, percent: unknown) =>
        `${remaining} left of ${total}${percent}`,
      topUpLabel: 'top-up',
      neverExpires: (remaining: unknown) => `${remaining} · never expires`,
      totalSpendable: (total: unknown) => `Total spendable: ${total}`
    }
  }
}
