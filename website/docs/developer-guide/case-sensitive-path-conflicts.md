# Case-sensitive path conflict handling

Hermes checks simple terminal write targets (`mkdir`, `touch`, `cp`, `mv`, and
`install`) before execution. The check runs inside the target terminal
environment, so it uses the mounted or remote filesystem rather than assuming
the host operating system's behavior.

Hermes first probes whether that filesystem distinguishes case. If it does not,
Hermes skips the extra check because the filesystem already treats `Desktop` and
`desktop` as the same path. If it does distinguish case and the requested target
has an existing case-only variant, Hermes returns a `case_conflict` result and
does not execute the command.

The caller must then explicitly choose one of:

- `case_resolution: "use_existing"` — use the existing path spelling.
- `case_resolution: "create_variant"` — intentionally create/use the requested
  differently-cased path.

Hermes does not choose between these options. This is particularly important
when speech-to-text is used: spoken file and directory names do not reliably
encode capitalization, so a dictated `desktop` can be intended to refer to an
existing `Desktop` directory. Surfacing the conflict prevents Hermes from
silently creating or targeting the wrong case variant.

The check is deliberately limited to commands with unambiguous, directly
identifiable write destinations. Arbitrary shell programs and compound shell
syntax remain the user's responsibility; Hermes does not rewrite unknown shell
commands.
