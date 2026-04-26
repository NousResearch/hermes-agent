import { useCallback, useState } from "react";

export function useConfirmDelete<TId>({
  onDelete,
}: {
  onDelete: (id: TId) => Promise<void>;
}) {
  const [pendingId, setPendingId] = useState<TId | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const requestDelete = useCallback((id: TId) => {
    setPendingId(id);
  }, []);

  const cancel = useCallback(() => {
    if (!isDeleting) setPendingId(null);
  }, [isDeleting]);

  const confirm = useCallback(async () => {
    if (pendingId === null) return;
    const id = pendingId;
    setIsDeleting(true);
    try {
      await onDelete(id);
      setPendingId(null);
    } catch {
      // Dialog stays open; caller can surface errors in onDelete before rethrowing
    } finally {
      setIsDeleting(false);
    }
  }, [pendingId, onDelete]);

  return {
    cancel,
    confirm,
    isDeleting,
    isOpen: pendingId !== null,
    pendingId,
    requestDelete,
  } as const;
}
