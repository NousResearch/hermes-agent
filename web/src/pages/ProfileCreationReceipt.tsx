import { X } from "lucide-react";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Button } from "@nous-research/ui/ui/components/button";
import { useI18n } from "@/i18n";

export interface CreationReceipt {
  title: string;
  messages: string[];
}

/** Exceptional creation results outlive the success toast and the create modal. */
export function ProfileCreationReceipt({ receipt, onDismiss }: {
  receipt: CreationReceipt | null;
  onDismiss: () => void;
}) {
  const { t } = useI18n();
  if (!receipt) return null;
  return (
    <Card role="status" className="text-destructive">
      <CardContent className="flex items-start gap-3 py-4">
        <div className="min-w-0 flex-1 space-y-2 text-sm">
          <p className="font-medium">{receipt.title}</p>
          {receipt.messages.map((message, index) => <p key={index}>{message}</p>)}
        </div>
        <Button ghost size="icon" aria-label={t.app.dismiss ?? "Dismiss"} onClick={onDismiss}><X /></Button>
      </CardContent>
    </Card>
  );
}
