import { useCallback, useState } from "react";

export function useToast(duration = 3000) {
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const showToast = useCallback(
    (message: string, type: "success" | "error") => {
      setToast({ message, type });
      setTimeout(() => setToast(null), duration);
    },
    [duration],
  );

  return { toast, showToast };
}
