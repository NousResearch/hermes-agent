export function Toast({ toast }: { toast: { message: string; type: "success" | "error" } | null }) {
  if (!toast) return null;

  return (
    <div
      className={`fixed top-4 right-4 z-50 rounded-lg px-4 py-2 text-sm font-medium shadow-lg ${
        toast.type === "success"
          ? "bg-success/20 text-success border border-success/30"
          : "bg-destructive/20 text-destructive border border-destructive/30"
      }`}
    >
      {toast.message}
    </div>
  );
}
