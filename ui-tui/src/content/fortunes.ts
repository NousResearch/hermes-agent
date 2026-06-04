const FORTUNES = [
  '一次干净的重构，就能让一切豁然开朗',
  '今天一个小小重命名，明天省去一个大 Bug',
  '你的下一条提交信息将完美无瑕',
  '你忽略的边界情况，已在脑中解决',
  '最少的变更，最大的从容',
  '今日宜大胆删除，而非新增抽象',
  '合适的工具函数就在你的代码库中',
  '你将在过度思考追上之前发布',
  '测试即将拯救未来的你',
  '你的直觉在正确地质疑那个分支'
]

const LEGENDARY = [
  '🌟 传奇掉落：一行修复，一次通过',
  '🌟 传奇掉落：所有不稳定测试顺利通过',
  '🌟 传奇掉落：你的差异不教自明'
]

const hash = (s: string) => [...s].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261) >>> 0

const fromScore = (n: number) => {
  const rare = n % 20 === 0
  const bag = rare ? LEGENDARY : FORTUNES

  return `${rare ? '🌟' : '🔮'} ${bag[n % bag.length]}`
}

export const randomFortune = () => fromScore(Math.floor(Math.random() * 0x7fffffff))
export const dailyFortune = (seed: null | string) => fromScore(hash(`${seed || 'anon'}|${new Date().toDateString()}`))
