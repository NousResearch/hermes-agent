export const CHAT_CARD_BREAKPOINT = 80

export const isCompactChatCardWidth = (cols: number) => cols < CHAT_CARD_BREAKPOINT

export const shouldUseChatCard = (cols: number, compact: boolean | undefined) => !compact && !isCompactChatCardWidth(cols)
