export function shouldRenderPersistentChatHost(
  chatHostMounted: boolean,
  isChatRoute: boolean,
) {
  return chatHostMounted || isChatRoute;
}
