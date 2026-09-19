---
title: Desktop subagent image previews
---

# Desktop subagent image previews

The existing `subagent.tail` Desktop RPC includes optional image metadata while retaining its text-log fields:

- `images`: ordered, deduplicated local image references.
- `image_session_id`: the child session used by the protected file API.
- `image_revision`: an event-driven revision for invalidating overwritten image files. Text-only progress does not advance it.
- `images_truncated`: some references were omitted by a supported-source or resource limit.

These are live, best-effort inspection metadata, not archived image history or an extension of the public subagent lifecycle API. Clients must tolerate absent fields. Desktop pins the parent gateway/profile route but reads files through the child session; it never falls back to the parent file scope or an arbitrary local disk read.

## Capture and lifecycle

The live-log writer captures structured tool arguments/results before text truncation. Known image fields, explicit image arrays, native image blocks and supported local media markers are extracted with bounded traversal. Relative paths require a reliable child working directory; transient terminal working directories and sandbox-to-host mappings are not guessed. Native pixels, including crops, take precedence over provisional source references and use the existing image cache.

The tail handler validates parent ownership and transport before reading, then revalidates the registry record, child identity and authority afterward. Retired or replaced children cannot leak stale image metadata through an in-flight tail request. The UI discards replies after child/session/owner changes, serializes polling and preserves image cache identity during text-only updates.

## Bounds

Per child, collection retains at most 32 references, 2,048 characters per reference and a 24 KiB JSON-escaped reference budget. Native caching permits at most 32 images, 4 MiB each and 16 MiB cumulatively. Parsing is capped at 512 nodes, depth 8, 8 MiB of JSON characters and 64 KiB of text scanning per container. Resource limits and unsupported image references set `images_truncated`.

Remote URL fetching, SVG and sandbox-path translation are not supported by this subagent path. These restrictions do not narrow existing normal tool galleries. The shared gallery mounts five thumbnails per page and shares a global maximum of three concurrent file reads. A pinned side preview retains already-loaded pixels independently of the live child registry; image bytes are not persisted in the preview-tab metadata.

## Verification

Run the focused backend boundary tests with the repository runner:

```bash
scripts/run_tests.sh tests/tools/test_delegation_live_log.py \
  tests/tools/test_delegation_image_refs.py \
  tests/tui_gateway/test_subagent_snapshot.py \
  tests/tools/test_delegate_child_transcript_release.py \
  tests/tui_gateway/test_subagent_child_mirror.py --file-retries 0 -q
```

Desktop tests cover routing, owner/child changes, legacy payloads, slow reads, text-only polls, omission notices and same-path overwrite invalidation in `subagent-images.test.tsx`, together with the existing gallery and source-loader regressions. New image bytes do not appear in a previously pinned snapshot automatically: open the updated selected image again when needed.
