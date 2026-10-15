import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import type { McpNetwork } from "@/lib/api";

interface McpNetworkFieldsProps {
  id: string;
  network: McpNetwork;
  transport: "http" | "sse";
  onNetworkChange: (value: McpNetwork) => void;
  onTransportChange: (value: "http" | "sse") => void;
}

export function McpNetworkFields({
  id,
  network,
  transport,
  onNetworkChange,
  onTransportChange,
}: McpNetworkFieldsProps) {
  return (
    <>
      <div className="grid gap-2">
        <Label htmlFor={`${id}-network`}>Network target</Label>
        <Select
          id={`${id}-network`}
          value={network}
          onValueChange={value => onNetworkChange(value as McpNetwork)}
        >
          <SelectOption value="auto">
            Automatic (backend first, then Windows from WSL)
          </SelectOption>
          <SelectOption value="local">Backend only</SelectOption>
          <SelectOption value="windows">Windows loopback</SelectOption>
        </Select>
        <p className="text-xs text-muted-foreground">
          Relative to the selected backend, not this browser. New entries save
          this choice explicitly; legacy entries without it stay backend-only.
          Windows loopback requires a WSL or Windows backend and a localhost URL.
        </p>
      </div>
      <div className="grid gap-2">
        <Label htmlFor={`${id}-http-transport`}>HTTP protocol</Label>
        <Select
          id={`${id}-http-transport`}
          value={transport}
          onValueChange={value => onTransportChange(value as "http" | "sse")}
        >
          <SelectOption value="http">Streamable HTTP</SelectOption>
          <SelectOption value="sse">Legacy SSE</SelectOption>
        </Select>
      </div>
    </>
  );
}
