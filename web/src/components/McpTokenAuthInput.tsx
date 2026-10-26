import type { ReactElement } from "react";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import type { McpHttpAuth } from "@/lib/api";
import { mcpTokenAuthFields } from "@/lib/mcp-server-create";

interface McpTokenAuthInputProps {
  auth: McpHttpAuth;
  id: string;
  value: string;
  onChange: (value: string) => void;
  className: string;
  profile?: "this profile" | "the new profile";
}

export function McpTokenAuthInput({
  auth,
  id,
  value,
  onChange,
  className,
  profile = "this profile",
}: McpTokenAuthInputProps): ReactElement | null {
  const fields = mcpTokenAuthFields(auth, profile);
  if (!fields) return null;
  return (
    <div className={className}>
      <Label htmlFor={id}>{fields.label}</Label>
      <Input
        id={id}
        type="password"
        autoComplete="new-password"
        placeholder={fields.placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <p className="text-xs text-muted-foreground">{fields.hint}</p>
    </div>
  );
}
