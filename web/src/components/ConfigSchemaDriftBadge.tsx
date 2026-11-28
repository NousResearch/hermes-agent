import { Badge } from "@nous-research/ui/ui/components/badge";

import { configSchemaBehind } from "@/lib/config-schema-drift";

interface ConfigSchemaDriftBadgeProps {
  status: Parameters<typeof configSchemaBehind>[0];
}

/** "config schema N behind" beside the Hermes version; renders nothing when the schema is current. */
export function ConfigSchemaDriftBadge({ status }: ConfigSchemaDriftBadgeProps) {
  const behind = configSchemaBehind(status);
  if (behind <= 0) {
    return null;
  }
  return (
    <span
      title={`config.yaml is at schema version ${status?.config_version}; this build expects ${status?.latest_config_version}. Run \`hermes setup\` to migrate it.`}
    >
      <Badge tone="warning">config schema {behind} behind</Badge>
    </span>
  );
}
