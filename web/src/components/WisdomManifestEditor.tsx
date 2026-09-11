import { useId } from 'react'
import { Plus, Trash2 } from 'lucide-react'

import { Button } from '@nous-research/ui/ui/components/button'
import { Checkbox } from '@nous-research/ui/ui/components/checkbox'
import { Input } from '@nous-research/ui/ui/components/input'
import { Label } from '@nous-research/ui/ui/components/label'
import { parseWisdomManifest, wisdomManifestValidationError } from '@/lib/wisdom-manifest'
import type {
  WisdomManifestV1,
  WisdomPluginRequirement,
  WisdomSystemSpecification,
  WisdomToolRequirement
} from '@/lib/wisdom-manifest'

interface SpecificationEditorProps {
  value: WisdomSystemSpecification
  disabled?: boolean
  onChange: (value: WisdomSystemSpecification) => void
}

interface ManifestEditorProps {
  value: string
  disabled?: boolean
  onChange: (value: string) => void
}

interface StringListFieldProps {
  label: string
  description: string
  itemLabel: string
  value: string[]
  disabled?: boolean
  placeholder?: string
  onChange: (value: string[]) => void
}

const PLATFORM_OPTIONS = [
  { value: 'macOS', label: 'macOS' },
  { value: 'Linux', label: 'Linux' },
  { value: 'Windows', label: 'Windows' }
] as const

const ARCHITECTURE_OPTIONS = [
  { value: 'arm64', label: 'ARM64 / Apple silicon' },
  { value: 'x86_64', label: 'x86-64 / AMD64' }
] as const

function StringListField({
  label,
  description,
  itemLabel,
  value,
  disabled = false,
  placeholder,
  onChange
}: StringListFieldProps) {
  const id = useId()
  return (
    <div className="space-y-2">
      <div>
        <Label className="text-xs font-medium">{label}</Label>
        <p className="mt-0.5 text-[11px] leading-4 text-text-tertiary">{description}</p>
      </div>
      {value.map((item, index) => (
        <div key={`${id}-${index}`} className="flex items-center gap-2">
          <Input
            aria-label={`${itemLabel} ${index + 1}`}
            disabled={disabled}
            maxLength={512}
            placeholder={placeholder}
            value={item}
            onChange={event =>
              onChange(value.map((existing, itemIndex) => (itemIndex === index ? event.target.value : existing)))
            }
          />
          <Button
            aria-label={`Remove ${itemLabel} ${index + 1}`}
            size="icon"
            outlined
            disabled={disabled}
            onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
      <Button
        size="sm"
        outlined
        disabled={disabled || value.length >= 64}
        onClick={() => onChange([...value, ''])}
        prefix={<Plus className="h-3.5 w-3.5" />}
      >
        Add {itemLabel}
      </Button>
    </div>
  )
}

function PresetMultiSelect({
  label,
  description,
  value,
  options,
  disabled,
  onChange
}: {
  label: string
  description: string
  value: string[]
  options: ReadonlyArray<{ value: string; label: string }>
  disabled?: boolean
  onChange: (value: string[]) => void
}) {
  const id = useId()
  const presets = new Set(options.map(option => option.value))
  const custom = value.filter(item => !presets.has(item))
  return (
    <div className="space-y-3">
      <div>
        <Label className="text-xs font-medium">{label}</Label>
        <p className="mt-0.5 text-[11px] leading-4 text-text-tertiary">{description}</p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {options.map(option => {
          const checkboxId = `${id}-${option.value}`
          const checked = value.includes(option.value)
          return (
            <div key={option.value} className="flex items-center gap-2">
              <Checkbox
                id={checkboxId}
                checked={checked}
                disabled={disabled}
                onCheckedChange={next =>
                  onChange(next === true ? [...value, option.value] : value.filter(item => item !== option.value))
                }
              />
              <Label htmlFor={checkboxId} className="cursor-pointer text-xs font-normal">
                {option.label}
              </Label>
            </div>
          )
        })}
      </div>
      <StringListField
        label={`Other ${label.toLowerCase()}`}
        description="Keep this empty unless the skill targets a platform not listed above."
        itemLabel={`other ${label.toLowerCase().replace(/s$/, '')}`}
        value={custom}
        disabled={disabled}
        onChange={next => onChange([...value.filter(item => presets.has(item)), ...next])}
      />
    </div>
  )
}

function BooleanField({
  id,
  label,
  description,
  checked,
  disabled,
  onChange
}: {
  id: string
  label: string
  description: string
  checked: boolean
  disabled?: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <div className="flex items-start gap-2">
      <Checkbox id={id} checked={checked} disabled={disabled} onCheckedChange={next => onChange(next === true)} />
      <div>
        <Label htmlFor={id} className="cursor-pointer text-xs font-medium">
          {label}
        </Label>
        <p className="mt-0.5 text-[11px] leading-4 text-text-tertiary">{description}</p>
      </div>
    </div>
  )
}

function ToolRequirements({
  value,
  disabled,
  onChange
}: {
  value: WisdomToolRequirement[]
  disabled?: boolean
  onChange: (value: WisdomToolRequirement[]) => void
}) {
  const id = useId()
  return (
    <div className="space-y-3">
      {value.map((tool, index) => (
        <div key={index} className="grid gap-3 border border-border/80 p-3 sm:grid-cols-2">
          <div>
            <Label htmlFor={`${id}-tool-${index}`} className="text-xs">
              Tool name
            </Label>
            <Input
              id={`${id}-tool-${index}`}
              className="mt-1"
              disabled={disabled}
              maxLength={512}
              value={tool.name}
              onChange={event =>
                onChange(
                  value.map((item, itemIndex) => (itemIndex === index ? { ...item, name: event.target.value } : item))
                )
              }
            />
          </div>
          <div>
            <Label htmlFor={`${id}-tool-version-${index}`} className="text-xs">
              Minimum version (optional)
            </Label>
            <Input
              id={`${id}-tool-version-${index}`}
              className="mt-1"
              disabled={disabled}
              maxLength={512}
              value={tool.minimum_version ?? ''}
              onChange={event =>
                onChange(
                  value.map((item, itemIndex) =>
                    itemIndex === index ? { ...item, minimum_version: event.target.value || null } : item
                  )
                )
              }
            />
          </div>
          <BooleanField
            id={`${id}-tool-admin-${index}`}
            label="Administrator permission required"
            description="Installing or enabling this tool needs an administrator's action."
            checked={tool.requires_admin}
            disabled={disabled}
            onChange={requires_admin =>
              onChange(value.map((item, itemIndex) => (itemIndex === index ? { ...item, requires_admin } : item)))
            }
          />
          <div className="flex items-center justify-between gap-3 text-[11px] text-text-tertiary">
            <span>Automatic installation is always off.</span>
            <Button
              size="sm"
              outlined
              disabled={disabled}
              onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
              prefix={<Trash2 className="h-3.5 w-3.5" />}
            >
              Remove
            </Button>
          </div>
        </div>
      ))}
      <Button
        size="sm"
        outlined
        disabled={disabled || value.length >= 64}
        onClick={() =>
          onChange([...value, { name: '', minimum_version: null, auto_install: false, requires_admin: false }])
        }
        prefix={<Plus className="h-3.5 w-3.5" />}
      >
        Add tool
      </Button>
    </div>
  )
}

function PluginRequirements({
  value,
  disabled,
  onChange
}: {
  value: WisdomPluginRequirement[]
  disabled?: boolean
  onChange: (value: WisdomPluginRequirement[]) => void
}) {
  const id = useId()
  return (
    <div className="space-y-3">
      {value.map((plugin, index) => (
        <div key={index} className="grid gap-3 border border-border/80 p-3 sm:grid-cols-2">
          <div>
            <Label htmlFor={`${id}-plugin-${index}`} className="text-xs">
              Plugin ID
            </Label>
            <Input
              id={`${id}-plugin-${index}`}
              className="mt-1"
              disabled={disabled}
              maxLength={512}
              value={plugin.id}
              onChange={event =>
                onChange(
                  value.map((item, itemIndex) => (itemIndex === index ? { ...item, id: event.target.value } : item))
                )
              }
            />
          </div>
          <div>
            <Label htmlFor={`${id}-plugin-version-${index}`} className="text-xs">
              Minimum version (optional)
            </Label>
            <Input
              id={`${id}-plugin-version-${index}`}
              className="mt-1"
              disabled={disabled}
              maxLength={512}
              value={plugin.minimum_version ?? ''}
              onChange={event =>
                onChange(
                  value.map((item, itemIndex) =>
                    itemIndex === index ? { ...item, minimum_version: event.target.value || null } : item
                  )
                )
              }
            />
          </div>
          <BooleanField
            id={`${id}-plugin-required-${index}`}
            label="Required"
            description="Without this plugin, the skill cannot provide its complete behavior."
            checked={plugin.required}
            disabled={disabled}
            onChange={required =>
              onChange(value.map((item, itemIndex) => (itemIndex === index ? { ...item, required } : item)))
            }
          />
          <div className="flex items-center justify-end">
            <Button
              size="sm"
              outlined
              disabled={disabled}
              onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
              prefix={<Trash2 className="h-3.5 w-3.5" />}
            >
              Remove
            </Button>
          </div>
        </div>
      ))}
      <Button
        size="sm"
        outlined
        disabled={disabled || value.length >= 64}
        onClick={() => onChange([...value, { id: '', minimum_version: null, required: true }])}
        prefix={<Plus className="h-3.5 w-3.5" />}
      >
        Add plugin
      </Button>
    </div>
  )
}

export function WisdomSystemSpecificationEditor({ value, disabled = false, onChange }: SpecificationEditorProps) {
  const id = useId()
  const update = <K extends keyof WisdomSystemSpecification>(key: K, next: WisdomSystemSpecification[K]) => {
    onChange({ ...value, [key]: next })
  }

  return (
    <div className="space-y-4">
      <p className="border border-border bg-muted/10 p-3 text-[11px] leading-4 text-text-secondary">
        Hermes pre-fills new drafts from this authoring device and requirements explicitly recorded on the skill.
        Review these values before sharing: selected platforms and architectures restrict where teammates can install
        the skill. Clear every selection only when the skill is known to work everywhere.
      </p>
      <fieldset className="grid gap-4 border border-border p-4 sm:grid-cols-2">
        <legend className="px-2 text-xs font-semibold uppercase tracking-wide">Compatibility targets</legend>
        <div>
          <Label htmlFor={`${id}-hermes-version`} className="text-xs font-medium">
            Minimum Hermes version
          </Label>
          <p className="mb-2 mt-0.5 text-[11px] text-text-tertiary">Older Hermes installations will be blocked.</p>
          <Input
            id={`${id}-hermes-version`}
            disabled={disabled}
            maxLength={512}
            value={value.hermes.minimum_version}
            onChange={event => update('hermes', { minimum_version: event.target.value })}
          />
        </div>
        <div>
          <Label htmlFor={`${id}-context-window`} className="text-xs font-medium">
            Minimum model context window
          </Label>
          <p className="mb-2 mt-0.5 text-[11px] text-text-tertiary">Leave blank when no minimum is required.</p>
          <Input
            id={`${id}-context-window`}
            type="number"
            min={1}
            step={1}
            disabled={disabled}
            value={value.model.minimum_context_window ?? ''}
            onChange={event =>
              update('model', {
                ...value.model,
                minimum_context_window: event.target.value === '' ? null : Number(event.target.value)
              })
            }
          />
        </div>
        <PresetMultiSelect
          label="Platforms"
          description="Checked systems are the allowed install targets. New drafts start with this authoring system."
          value={value.platforms}
          options={PLATFORM_OPTIONS}
          disabled={disabled}
          onChange={platforms => update('platforms', platforms)}
        />
        <PresetMultiSelect
          label="Architectures"
          description="Checked processors are the allowed install targets. New drafts start with this device."
          value={value.architectures}
          options={ARCHITECTURE_OPTIONS}
          disabled={disabled}
          onChange={architectures => update('architectures', architectures)}
        />
        <StringListField
          label="Model capabilities"
          description="Capabilities the active model must provide, such as vision or tool use."
          itemLabel="model capability"
          value={value.model.capabilities}
          disabled={disabled}
          placeholder="vision"
          onChange={capabilities => update('model', { ...value.model, capabilities })}
        />
        <StringListField
          label="Hardware"
          description="Physical hardware requirements, such as a GPU."
          itemLabel="hardware requirement"
          value={value.hardware}
          disabled={disabled}
          placeholder="gpu"
          onChange={hardware => update('hardware', hardware)}
        />
      </fieldset>

      <fieldset className="border border-border p-4">
        <legend className="px-2 text-xs font-semibold uppercase tracking-wide">Runtime access</legend>
        <p className="mb-4 text-[11px] leading-4 text-text-tertiary">
          These requirements are checked locally before installation. They do not grant permissions by themselves.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <BooleanField
            id={`${id}-shell`}
            label="Shell commands"
            description="The skill needs to run terminal commands."
            checked={value.runtime.shell}
            disabled={disabled}
            onChange={shell => update('runtime', { ...value.runtime, shell })}
          />
          <BooleanField
            id={`${id}-browser`}
            label="Browser control"
            description="The skill needs an interactive browser."
            checked={value.runtime.browser}
            disabled={disabled}
            onChange={browser => update('runtime', { ...value.runtime, browser })}
          />
          <BooleanField
            id={`${id}-code`}
            label="Code execution"
            description="The skill needs a code-execution environment."
            checked={value.runtime.code}
            disabled={disabled}
            onChange={code => update('runtime', { ...value.runtime, code })}
          />
          <BooleanField
            id={`${id}-sandbox`}
            label="Sandbox required"
            description="The skill must run with sandbox isolation enabled."
            checked={value.runtime.sandbox}
            disabled={disabled}
            onChange={sandbox => update('runtime', { ...value.runtime, sandbox })}
          />
        </div>
      </fieldset>

      <fieldset className="border border-border p-4">
        <legend className="px-2 text-xs font-semibold uppercase tracking-wide">Tools and plugins</legend>
        <div className="space-y-5">
          <div>
            <h4 className="text-xs font-medium">Tools</h4>
            <p className="mb-3 mt-0.5 text-[11px] text-text-tertiary">
              Hermes pre-fills explicit skill requirements and checks whether they are enabled. This manifest can never
              install them automatically.
            </p>
            <ToolRequirements value={value.tools} disabled={disabled} onChange={tools => update('tools', tools)} />
          </div>
          <div className="border-t border-border pt-4">
            <h4 className="text-xs font-medium">Plugins</h4>
            <p className="mb-3 mt-0.5 text-[11px] text-text-tertiary">
              Hermes pre-fills explicitly declared plugin IDs. Confirm whether each is required.
            </p>
            <PluginRequirements
              value={value.plugins}
              disabled={disabled}
              onChange={plugins => update('plugins', plugins)}
            />
          </div>
        </div>
      </fieldset>

      <fieldset className="grid gap-5 border border-border p-4 sm:grid-cols-2">
        <legend className="px-2 text-xs font-semibold uppercase tracking-wide">Credentials and connections</legend>
        <StringListField
          label="Credentials"
          description="Names only. Secret values are never included in the manifest."
          itemLabel="credential"
          value={value.credentials}
          disabled={disabled}
          placeholder="EXAMPLE_TOKEN"
          onChange={credentials => update('credentials', credentials)}
        />
        <StringListField
          label="Connections"
          description="Configured services or linked accounts the skill needs."
          itemLabel="connection"
          value={value.connections}
          disabled={disabled}
          placeholder="team-mcp"
          onChange={connections => update('connections', connections)}
        />
      </fieldset>

      <fieldset className="grid gap-5 border border-border p-4 sm:grid-cols-2">
        <legend className="px-2 text-xs font-semibold uppercase tracking-wide">Data access</legend>
        <StringListField
          label="Filesystem read access"
          description="Paths the skill needs to read."
          itemLabel="read path"
          value={value.filesystem.read}
          disabled={disabled}
          placeholder="~/project"
          onChange={read => update('filesystem', { ...value.filesystem, read })}
        />
        <StringListField
          label="Filesystem write access"
          description="Paths the skill needs to modify."
          itemLabel="write path"
          value={value.filesystem.write}
          disabled={disabled}
          placeholder="~/project/output"
          onChange={write => update('filesystem', { ...value.filesystem, write })}
        />
        <StringListField
          label="Network destinations"
          description="Hosts or services the skill needs to contact."
          itemLabel="network destination"
          value={value.network.destinations}
          disabled={disabled}
          placeholder="api.example.com"
          onChange={destinations => update('network', { destinations })}
        />
        <StringListField
          label="Known limitations"
          description="Important constraints teammates should understand before installation."
          itemLabel="known limitation"
          value={value.known_limitations}
          disabled={disabled}
          placeholder="Manual verification is still required"
          onChange={known_limitations => update('known_limitations', known_limitations)}
        />
      </fieldset>
    </div>
  )
}

export function WisdomManifestEditor({ value, disabled = false, onChange }: ManifestEditorProps) {
  const id = useId()
  let manifest: WisdomManifestV1
  try {
    manifest = parseWisdomManifest(value)
  } catch (reason) {
    return (
      <div role="alert" className="mt-3 border border-red-500/50 p-3 text-xs text-red-300">
        This server-reviewed manifest cannot be safely represented as a form. Reload the draft and try again.{' '}
        {reason instanceof Error ? reason.message : String(reason)}
      </div>
    )
  }

  const validationError = wisdomManifestValidationError(value)
  const update = (next: WisdomManifestV1) => onChange(JSON.stringify(next))

  return (
    <div className="mt-3 space-y-4">
      {validationError && (
        <div role="alert" className="border border-amber-500/50 bg-amber-500/5 p-3 text-xs text-amber-200">
          {validationError}
        </div>
      )}
      <fieldset className="grid gap-4 border border-border p-4 sm:grid-cols-[1fr_auto]">
        <legend className="px-2 text-xs font-semibold uppercase tracking-wide">Skill identity</legend>
        <div>
          <Label htmlFor={`${id}-manifest-name`} className="text-xs font-medium">
            Skill name
          </Label>
          <p className="mb-2 mt-0.5 text-[11px] text-text-tertiary">The package name recorded in the manifest.</p>
          <Input
            id={`${id}-manifest-name`}
            disabled={disabled}
            maxLength={512}
            value={manifest.name}
            onChange={event => update({ ...manifest, name: event.target.value })}
          />
        </div>
        <div className="min-w-32">
          <Label className="text-xs font-medium">Schema version</Label>
          <p className="mt-2 font-mono text-sm">1</p>
          <p className="mt-0.5 text-[11px] text-text-tertiary">Fixed by the V1 contract.</p>
        </div>
      </fieldset>
      <WisdomSystemSpecificationEditor
        value={manifest.requirements}
        disabled={disabled}
        onChange={requirements => update({ ...manifest, requirements })}
      />
    </div>
  )
}
