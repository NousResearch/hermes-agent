import { useLayoutEffect, useState, useSyncExternalStore } from "react";
import { ChatDraftUploads, type UploadContext } from "@/lib/chatDraftUploads";

export function useChatFileUploads({ profile, scope, active }: UploadContext) {
  // Stable methods are dependencies of the long-lived PTY effect. Progress
  // must never replace the socket or xterm instance via callback churn.
  const [controller] = useState(() => new ChatDraftUploads());
  useLayoutEffect(() => controller.setContext({ profile, scope, active }), [controller, profile, scope, active]);
  useLayoutEffect(() => () => controller.dispose(), [controller]);
  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot);
  return { ...snapshot, select: controller.select, bind: controller.bind, receive: controller.receive, retry: controller.retry, remove: controller.remove };
}
