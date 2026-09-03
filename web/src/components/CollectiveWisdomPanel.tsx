import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Loader2, Search, Sparkles } from 'lucide-react'

import { api } from '@/lib/api'
import type {
  WisdomCandidate,
  WisdomCheckResult,
  WisdomDiscovery,
  WisdomDraft,
  WisdomDraftReview,
  WisdomPreparedDraft,
  WisdomReviewCheck,
  WisdomActionPlan,
  WisdomInstallations,
  WisdomSkillDetail,
  WisdomStatus,
  WisdomUpdateMode,
  WisdomVersionContent
} from '@/lib/api'
import { Button } from '@nous-research/ui/ui/components/button'
import { Input } from '@nous-research/ui/ui/components/input'
import { useI18n } from '@/i18n'
import { WisdomFileEditor } from './WisdomFileEditor'
import { WisdomSystemSpecificationEditor } from './WisdomManifestEditor'
import { WisdomCheckBadge, WisdomReviewTables } from './WisdomChecks'
import {
  parseWisdomSystemSpecification,
  wisdomManifestValidationError,
  wisdomSystemSpecificationValidationError
} from '@/lib/wisdom-manifest'
import type { WisdomSystemSpecification } from '@/lib/wisdom-manifest'

interface Props {
  profile?: string
}

const TERMINAL_DRAFT_STATES = new Set(['published', 'declined', 'invalidated', 'rejected'])
const UPDATE_CHECK_INTERVAL_MS = 5 * 60 * 1000

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function userFacingError(reason: unknown): string {
  const message = reason instanceof Error ? reason.message : String(reason)
  const jsonStart = message.indexOf('{')
  if (jsonStart >= 0) {
    try {
      const body = JSON.parse(message.slice(jsonStart)) as { detail?: unknown }
      if (typeof body.detail === 'string' && body.detail.trim()) return body.detail
    } catch {
      // Keep the original non-JSON error below.
    }
  }
  return message.replace(/^Error:\s*/, '')
}

const candidateDisplayName = (candidate: WisdomCandidate) => candidate.editorial_name?.trim() || candidate.name

const candidateDisplayDescription = (candidate: WisdomCandidate) => candidate.editorial_description?.trim() || ''

function wisdomActionFailure(status: { exit_code: number | null; lines: string[] }): string {
  const lastRunMarker = status.lines.findLastIndex(line => line.startsWith('==='))
  const latestRun = status.lines
    .slice(lastRunMarker + 1)
    .join('\n')
    .trim()
  const objectStart = latestRun.indexOf('{')
  const objectEnd = latestRun.lastIndexOf('}')

  if (objectStart >= 0 && objectEnd > objectStart) {
    try {
      const body = asRecord(JSON.parse(latestRun.slice(objectStart, objectEnd + 1)))
      for (const key of ['error', 'detail', 'message']) {
        const value = body[key]
        if (typeof value === 'string' && value.trim()) return value
      }
    } catch {
      // Action logs can also contain ordinary command output. Use the stable
      // fallback below rather than leaking JSON punctuation into the UI.
    }
  }

  return `Collective Wisdom action failed (${status.exit_code ?? 'unknown'})`
}

async function waitForWisdomAction(name: string): Promise<void> {
  for (let attempt = 0; attempt < 1200; attempt += 1) {
    const status = await api.getActionStatus(name, 80)
    if (!status.running) {
      if (status.exit_code !== 0) {
        throw new Error(wisdomActionFailure(status))
      }
      return
    }
    await new Promise(resolve => setTimeout(resolve, 500))
  }
  throw new Error('Collective Wisdom action timed out')
}

export function CollectiveWisdomPanel({ profile }: Props) {
  const { t } = useI18n()
  const copy = t.skills.wisdom
  const [status, setStatus] = useState<WisdomStatus | null>(null)
  const [discovery, setDiscovery] = useState<WisdomDiscovery>({ skills: [], next_cursor: null })
  const [candidates, setCandidates] = useState<WisdomCandidate[]>([])
  const [drafts, setDrafts] = useState<WisdomDraft[]>([])
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<WisdomSkillDetail | null>(null)
  const [content, setContent] = useState<WisdomVersionContent | null>(null)
  const [installations, setInstallations] = useState<WisdomInstallations>({ installations: [], notifications: [] })
  const [updateCheck, setUpdateCheck] = useState<WisdomCheckResult | null>(null)
  const [actionPlan, setActionPlan] = useState<
    (WisdomActionPlan & { action: 'install' | 'update' | 'uninstall' }) | null
  >(null)
  const [actionPlanReference, setActionPlanReference] = useState<string | null>(null)
  const [acceptSensitive, setAcceptSensitive] = useState(false)
  const [acceptPartial, setAcceptPartial] = useState(false)
  const [preserveModified, setPreserveModified] = useState(false)
  const [review, setReview] = useState<WisdomDraftReview | null>(null)
  const [reviewDescription, setReviewDescription] = useState('')
  const [reviewFiles, setReviewFiles] = useState<Record<string, string>>({})
  const [prepared, setPrepared] = useState<WisdomPreparedDraft | null>(null)
  const [preparedSkill, setPreparedSkill] = useState('')
  const [preparedSkillId, setPreparedSkillId] = useState('')
  const [approvedDescription, setApprovedDescription] = useState('')
  const [approvedSpecification, setApprovedSpecification] = useState<WisdomSystemSpecification | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showManualCandidates, setShowManualCandidates] = useState(false)
  const [installReference, setInstallReference] = useState('')
  const [installUpdateMode, setInstallUpdateMode] = useState<'' | WisdomUpdateMode>('')

  const loadConfiguredData = useCallback(async () => {
    return Promise.all([
      api.getWisdomDiscovery(profile),
      api.getWisdomCandidates(profile),
      api.getWisdomDrafts(profile),
      api.getWisdomInstallations(profile)
    ])
  }, [profile])

  useEffect(() => {
    let cancelled = false
    api
      .getWisdomStatus(profile)
      .then(async nextStatus => {
        if (cancelled) return
        setStatus(nextStatus)
        if (nextStatus.configured) {
          const [nextDiscovery, nextCandidates, nextDrafts, nextInstallations] = await loadConfiguredData()
          if (cancelled) return
          setDiscovery(nextDiscovery)
          setCandidates(nextCandidates.candidates)
          setDrafts(nextDrafts.drafts)
          setInstallations(nextInstallations)
        }
        if (!cancelled) setError(null)
      })
      .catch(reason => !cancelled && setError(userFacingError(reason)))
      .finally(() => !cancelled && setBusy(null))
    return () => {
      cancelled = true
    }
  }, [loadConfiguredData, profile])

  useEffect(() => {
    if (!status?.configured) return

    let cancelled = false
    let inFlight = false
    let lastCheckedAt = 0

    const checkForUpdates = async () => {
      if (cancelled || inFlight || document.visibilityState === 'hidden') return
      inFlight = true
      try {
        const nextCheck = await api.checkWisdom(profile)
        const nextInstallations = await api.getWisdomInstallations(profile)
        if (!cancelled) {
          setUpdateCheck(nextCheck)
          setInstallations(nextInstallations)
          lastCheckedAt = Date.now()
        }
      } catch {
        // Background checks are best effort. The explicit check action reports failures.
      } finally {
        inFlight = false
      }
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && Date.now() - lastCheckedAt >= UPDATE_CHECK_INTERVAL_MS) {
        void checkForUpdates()
      }
    }

    void checkForUpdates()
    const timer = window.setInterval(() => void checkForUpdates(), UPDATE_CHECK_INTERVAL_MS)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [profile, status?.configured])

  const setupProfile = async () => {
    setBusy('setup')
    setError(null)
    try {
      const action = await api.setupWisdom(profile)
      await waitForWisdomAction(action.name)
      const nextStatus = await api.getWisdomStatus(profile)
      setStatus(nextStatus)
      const [nextDiscovery, nextCandidates, nextDrafts, nextInstallations] = await loadConfiguredData()
      setDiscovery(nextDiscovery)
      setCandidates(nextCandidates.candidates)
      setDrafts(nextDrafts.drafts)
      setInstallations(nextInstallations)
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return discovery.skills.filter(
      skill =>
        !normalized ||
        skill.slug.toLowerCase().includes(normalized) ||
        (skill.author_description || '').toLowerCase().includes(normalized)
    )
  }, [discovery.skills, query])

  const pendingUpdates = useMemo(
    () =>
      new Map(
        (updateCheck?.installations ?? [])
          .filter(item => item.state === 'update_available')
          .map(item => [item.skill_id, item] as const)
      ),
    [updateCheck?.installations]
  )

  const activeDrafts = useMemo(() => drafts.filter(draft => !TERMINAL_DRAFT_STATES.has(draft.state)), [drafts])

  const { manualCandidates, qualifiedCandidates } = useMemo(
    () => ({
      manualCandidates: candidates.filter(candidate => candidate.qualification === 'manual_selection'),
      qualifiedCandidates: candidates.filter(candidate => candidate.qualification !== 'manual_selection')
    }),
    [candidates]
  )

  const filterCandidates = useCallback(
    (items: WisdomCandidate[]) => {
      const needle = query.trim().toLocaleLowerCase()

      return items
        .filter(candidate =>
          !needle
            ? true
            : [candidate.name, candidateDisplayName(candidate), candidateDisplayDescription(candidate)].some(value =>
                value.toLocaleLowerCase().includes(needle)
              )
        )
        .toSorted((left, right) => candidateDisplayName(left).localeCompare(candidateDisplayName(right)))
    },
    [query]
  )

  const visibleQualifiedCandidates = useMemo(
    () => filterCandidates(qualifiedCandidates),
    [filterCandidates, qualifiedCandidates]
  )
  const visibleManualCandidates = useMemo(
    () => filterCandidates(manualCandidates),
    [filterCandidates, manualCandidates]
  )

  const reviewDirty = useMemo(() => {
    if (!review) return false
    if (reviewDescription !== (review.draft.authorDescription || '')) return true
    return review.files.some(file => reviewFiles[file.path] !== file.content_utf8)
  }, [review, reviewDescription, reviewFiles])

  const reviewManifestError = useMemo(() => {
    if (!review) return null
    const manifest = reviewFiles['skill.manifest.json']
    return manifest === undefined
      ? 'The complete package must include skill.manifest.json.'
      : wisdomManifestValidationError(manifest)
  }, [review, reviewFiles])

  const reviewCanEdit = !!review && ['ready', 'changes_requested'].includes(review.draft.state)
  const approvedSpecificationError = approvedSpecification
    ? wisdomSystemSpecificationValidationError(approvedSpecification)
    : 'System Specification is unavailable.'

  const showReview = (nextReview: WisdomDraftReview) => {
    setReview(nextReview)
    setReviewDescription(nextReview.draft.authorDescription || '')
    setReviewFiles(Object.fromEntries(nextReview.files.map(file => [file.path, file.content_utf8])))
  }

  const closeReview = () => {
    setReview(null)
    setReviewDescription('')
    setReviewFiles({})
  }

  const resetReviewEdits = () => {
    if (!review) return
    setReviewDescription(review.draft.authorDescription || '')
    setReviewFiles(Object.fromEntries(review.files.map(file => [file.path, file.content_utf8])))
  }

  const refreshContributionData = useCallback(async () => {
    const [nextCandidates, nextDrafts, nextDiscovery] = await Promise.all([
      api.getWisdomCandidates(profile),
      api.getWisdomDrafts(profile),
      api.getWisdomDiscovery(profile)
    ])
    setCandidates(nextCandidates.candidates)
    setDrafts(nextDrafts.drafts)
    setDiscovery(nextDiscovery)
  }, [profile])

  const notificationText = (event: Record<string, unknown>): string => {
    const payload = asRecord(event.payload)
    const skillId = String(event.skill_id ?? payload.skill_id ?? '')
    const skill =
      String(payload.slug ?? '') ||
      discovery.skills.find(item => item.id === skillId)?.slug ||
      installations.installations.find(item => item.skill_id === skillId)?.slug ||
      'A Collective Wisdom skill'
    const versionValue = event.version ?? payload.version
    const version = versionValue ? `v${String(versionValue)}` : undefined
    const kind = String(event.kind ?? '')
    if (kind === 'owner_decision') {
      const state = String(payload.state ?? '')
      if (state === 'published' || state === 'approved') return copy.decisionPublished(skill)
      if (state === 'changes_requested') {
        const note = typeof payload.moderation_note === 'string' ? payload.moderation_note.trim() : ''
        return `${copy.decisionChanges(skill)}${note ? ` ${note}` : ''}`
      }
      if (state === 'declined' || state === 'rejected') return copy.decisionDeclined(skill)
      return copy.decisionChanged(skill, copy.draftState(state))
    }
    if (kind === 'installed') return copy.installedNotice(skill, version)
    if (kind === 'updated' || kind === 'update_available' || kind === 'required_update') {
      return copy.updateNotice(skill, version)
    }
    if (kind === 'new' || kind === 'published') return copy.newSkillNotice(skill)
    if (kind === 'archived') return copy.archivedNotice(skill)
    if (kind === 'takedown') return copy.takedownNotice(skill)
    return copy.decisionChanged(skill, kind.replaceAll('_', ' ') || 'updated')
  }

  const candidateSummary = (candidate: WisdomCandidate): string => {
    if (candidate.eligibility !== 'eligible') return candidate.reason || copy.localOnly
    const qualification =
      candidate.qualification === 'manual_selection'
        ? copy.localOnly
        : candidate.notice_variant === 'first'
          ? copy.qualificationFirst(candidate.organization_name)
          : copy.qualificationReturning
    return candidate.contribution_state === 'prepared' ? `${qualification} ${copy.savedLocally}` : qualification
  }

  const openSkill = async (skillId: string) => {
    setBusy(skillId)
    setError(null)
    try {
      const detail = await api.getWisdomSkill(skillId, profile)
      setSelected(detail)
      const versions = detail.versions
        .map(version => Number(version.version))
        .filter(version => Number.isInteger(version) && version > 0)
      setContent(
        versions.length > 0 ? await api.getWisdomVersionContent(skillId, Math.max(...versions), profile) : null
      )
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const selectedId = selected ? String(selected.skill.id || '') : ''
  const installed = installations.installations.find(item => item.skill_id === selectedId && item.state === 'active')
  const selectedUpdate = pendingUpdates.get(selectedId)

  const refreshSharedSkills = async () => {
    const [nextDiscovery, nextInstallations] = await Promise.all([
      api.getWisdomDiscovery(profile),
      api.getWisdomInstallations(profile)
    ])
    setDiscovery(nextDiscovery)
    setInstallations(nextInstallations)

    if (selectedId) {
      const detail = await api.getWisdomSkill(selectedId, profile)
      setSelected(detail)
      const versions = detail.versions
        .map(version => Number(version.version))
        .filter(version => Number.isInteger(version) && version > 0)
      setContent(
        versions.length > 0 ? await api.getWisdomVersionContent(selectedId, Math.max(...versions), profile) : null
      )
    }
  }

  const planManagedAction = async (action: 'install' | 'update' | 'uninstall') => {
    if (!selectedId) return
    setBusy(selectedId)
    setError(null)
    try {
      const plan =
        action === 'install'
          ? await api.planWisdomInstall(selectedId, profile, installUpdateMode || undefined)
          : action === 'update'
            ? await api.planWisdomUpdate(selectedId, profile)
            : { skill_id: selectedId, state: 'confirm_uninstall' }
      setActionPlan({ ...plan, action })
      setActionPlanReference(action === 'install' ? selectedId : null)
      setAcceptSensitive(false)
      setAcceptPartial(false)
      setPreserveModified(false)
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const planReferencedInstall = async () => {
    const reference = installReference.trim()
    if (!reference) return

    setBusy('install-reference')
    setError(null)
    try {
      const plan = await api.planWisdomInstall(reference, profile, installUpdateMode || undefined)
      setActionPlan({ ...plan, action: 'install' })
      setActionPlanReference(reference)
      setAcceptSensitive(false)
      setAcceptPartial(false)
      setPreserveModified(false)
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const replanInstallUpdateMode = async (value: string) => {
    if (!actionPlan || actionPlan.action !== 'install') return

    const previousMode = installUpdateMode
    const nextMode = value as '' | WisdomUpdateMode
    const reference = actionPlanReference || actionPlan.skill_id

    setInstallUpdateMode(nextMode)
    setBusy('install-mode')
    setError(null)
    try {
      const plan = await api.planWisdomInstall(reference, profile, nextMode || undefined)
      setActionPlan({ ...plan, action: 'install' })
      setAcceptSensitive(false)
      setAcceptPartial(false)
      setPreserveModified(false)
    } catch (reason) {
      setInstallUpdateMode(previousMode)
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const applyManagedAction = async () => {
    if (!actionPlan) return
    setBusy(actionPlan.skill_id)
    try {
      if (actionPlan.action === 'uninstall') {
        await api.uninstallWisdomSkill(actionPlan.skill_id, profile)
      } else if (!actionPlan.receipt) {
        throw new Error('Verified action receipt is missing')
      } else if (actionPlan.action === 'install') {
        await api.applyWisdomInstall(actionPlan.receipt, acceptPartial, profile)
      } else {
        await api.applyWisdomUpdate(actionPlan.receipt, { acceptSensitive, acceptPartial, preserveModified }, profile)
      }
      await refreshSharedSkills()
      if (actionPlan.action === 'update' || actionPlan.action === 'uninstall') {
        setUpdateCheck(current =>
          current
            ? { ...current, installations: current.installations.filter(item => item.skill_id !== actionPlan.skill_id) }
            : current
        )
      }
      if (actionPlan.action === 'install') {
        setInstallReference('')
        setInstallUpdateMode('')
      }
      setActionPlan(null)
      setActionPlanReference(null)
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const prepare = async (candidate: WisdomCandidate) => {
    setBusy(candidate.local_skill_id)
    setError(null)
    try {
      const result = await api.suggestWisdomSkill(
        candidate.name,
        profile,
        undefined,
        undefined,
        candidate.local_skill_id
      )
      if ('network_submission' in result) {
        setPrepared(result)
        setPreparedSkill(candidate.name)
        setPreparedSkillId(candidate.local_skill_id)
        setApprovedDescription(result.drafted_description)
        setApprovedSpecification(parseWisdomSystemSpecification(result.system_specification))
      }
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const openReview = async (draftId: string) => {
    setBusy(draftId)
    setError(null)
    try {
      showReview(await api.reviewWisdomDraft(draftId, false, profile))
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  const saveReviewRevision = async () => {
    if (!review || !reviewCanEdit || !reviewDirty) return
    setBusy(review.draft.id)
    setError(null)
    try {
      if (!reviewDescription.trim()) {
        throw new Error('Add a description before saving this revision.')
      }
      const manifest = reviewFiles['skill.manifest.json']
      if (manifest === undefined) {
        throw new Error('The complete package must include skill.manifest.json.')
      }
      const manifestError = wisdomManifestValidationError(manifest)
      if (manifestError) throw new Error(`Fix the System Specification before saving: ${manifestError}`)
      const revised = await api.reviseWisdomDraft(
        review.draft.id,
        reviewDescription,
        review.files.map(file => ({
          path: file.path,
          content_utf8: reviewFiles[file.path] ?? file.content_utf8
        })),
        review.hashes,
        profile
      )
      await refreshContributionData()
      showReview(await api.reviewWisdomDraft(revised.draft.id, false, profile))
    } catch (reason) {
      setError(userFacingError(reason))
    } finally {
      setBusy(null)
    }
  }

  if (!status && !error) {
    return (
      <div className="flex items-center justify-center py-24" aria-label={copy.loading}>
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    )
  }

  if (error && !status) {
    return (
      <div role="alert" className="border border-border px-4 py-8 text-sm text-text-secondary">
        {copy.unavailable} {error}
      </div>
    )
  }

  if (status && !status.configured) {
    return (
      <section className="mx-auto max-w-2xl space-y-4 border border-border bg-muted/10 p-6" aria-label={copy.title}>
        <div>
          <h2 className="font-mondwest text-lg text-text-primary">{copy.title}</h2>
          <p className="mt-1 text-sm text-text-secondary">{copy.setup}</p>
        </div>
        <p className="text-sm leading-6 text-text-secondary">{copy.setupDisclosure}</p>
        {status.error && (
          <div role="alert" className="text-sm text-red-500">
            {status.error}
          </div>
        )}
        {error && (
          <div role="alert" className="text-sm text-red-500">
            {error}
          </div>
        )}
        <Button onClick={setupProfile} disabled={busy === 'setup'}>
          {busy === 'setup' ? copy.settingUp : copy.setupAction}
        </Button>
      </section>
    )
  }

  return (
    <section className="space-y-4" aria-label={copy.title}>
      <div className="flex flex-col gap-3 border border-border bg-muted/10 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="font-mondwest text-lg text-text-primary">{copy.title}</h2>
          <p className="text-xs text-text-secondary">
            {status?.verified_org_id ? `${status.verified_org_id} · org-wide collective` : copy.setup}
          </p>
        </div>
        <div className="relative w-full sm:max-w-sm">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-text-tertiary" />
          <Input
            aria-label={copy.search}
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder={copy.search}
            className="pl-9"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            outlined
            onClick={async () => {
              setBusy('refresh-shared')
              setError(null)
              try {
                await refreshSharedSkills()
              } catch (reason) {
                setError(userFacingError(reason))
              } finally {
                setBusy(null)
              }
            }}
          >
            {busy === 'refresh-shared' ? copy.refreshingShared : copy.refreshShared}
          </Button>
          <Button
            size="sm"
            outlined
            onClick={async () => {
              setBusy('scan')
              try {
                const action = await api.scanWisdom(undefined, profile)
                await waitForWisdomAction(action.name)
                const next = await api.getWisdomCandidates(profile)
                setCandidates(next.candidates)
              } catch (reason) {
                setError(userFacingError(reason))
              } finally {
                setBusy(null)
              }
            }}
          >
            {busy === 'scan' ? copy.checking : copy.scanLocal}
          </Button>
          <Button
            size="sm"
            outlined
            onClick={async () => {
              setBusy('check')
              setError(null)
              try {
                setUpdateCheck(await api.checkWisdom(profile))
                await refreshSharedSkills()
              } catch (reason) {
                setError(userFacingError(reason))
              } finally {
                setBusy(null)
              }
            }}
          >
            {busy === 'check' ? copy.checking : copy.checkUpdates(pendingUpdates.size)}
          </Button>
        </div>
      </div>

      <form
        className="grid gap-3 border border-border bg-muted/5 p-3 sm:grid-cols-[minmax(0,1fr)_minmax(12rem,0.55fr)_auto] sm:items-end"
        onSubmit={event => {
          event.preventDefault()
          void planReferencedInstall()
        }}
      >
        <div className="min-w-0 flex-1">
          <label className="mb-1 block text-xs font-medium" htmlFor="dashboard-wisdom-install-reference">
            {copy.installReferenceLabel}
          </label>
          <Input
            aria-describedby="dashboard-wisdom-install-reference-help"
            id="dashboard-wisdom-install-reference"
            onChange={event => setInstallReference(event.target.value)}
            placeholder={copy.installReferencePlaceholder}
            value={installReference}
          />
          <p className="mt-1 text-xs text-text-tertiary" id="dashboard-wisdom-install-reference-help">
            {copy.installReferenceHelp}
          </p>
        </div>
        <div className="min-w-0">
          <label className="mb-1 block text-xs font-medium" htmlFor="dashboard-wisdom-update-mode">
            {copy.updateModeLabel}
          </label>
          <select
            aria-describedby="dashboard-wisdom-update-mode-help"
            className="h-9 w-full border border-border bg-background px-2 text-xs"
            id="dashboard-wisdom-update-mode"
            onChange={event => setInstallUpdateMode(event.target.value as '' | WisdomUpdateMode)}
            value={installUpdateMode}
          >
            <option value="">{copy.updateModeDefault}</option>
            <option value="MANUAL">{copy.updateModeManual}</option>
            <option value="AUTO_WITH_NOTICE">{copy.updateModeAutomatic}</option>
            <option value="REQUIRED">{copy.updateModeRequired}</option>
          </select>
          <p className="mt-1 text-xs text-text-tertiary" id="dashboard-wisdom-update-mode-help">
            {copy.updateModeHelp}
          </p>
        </div>
        <Button disabled={!installReference.trim() || busy === 'install-reference'} outlined size="sm" type="submit">
          {busy === 'install-reference' ? copy.planningInstall : copy.reviewInstall}
        </Button>
      </form>

      {error && (
        <div role="alert" className="flex items-center gap-2 border border-red-500/40 px-3 py-2 text-sm">
          <AlertTriangle className="h-4 w-4" />
          {error}
        </div>
      )}

      {installations.notifications.length > 0 && (
        <div className="border border-blue-500/40 p-3 text-xs" aria-label="Collective Wisdom notifications">
          <div className="flex items-center justify-between gap-3">
            <div>
              <span>{copy.activityReady(installations.notifications.length)}</span>
              <ul className="mt-2 space-y-1 text-text-tertiary">
                {installations.notifications.slice(0, 8).map((event, index) => (
                  <li key={String(event.event_id ?? index)}>{notificationText(event)}</li>
                ))}
              </ul>
            </div>
            <Button
              size="sm"
              outlined
              onClick={async () => {
                await api.acknowledgeWisdomNotifications(profile)
                setInstallations(await api.getWisdomInstallations(profile))
              }}
            >
              {copy.markSeen}
            </Button>
          </div>
        </div>
      )}

      {(candidates.length > 0 || activeDrafts.length > 0) && (
        <div className="grid gap-3 border border-border p-4 lg:grid-cols-2">
          <div>
            <h3 className="text-sm font-medium">{copy.potential}</h3>
            <p className="mb-2 mt-1 text-xs text-text-tertiary">{copy.potentialHelp}</p>
            <div className="space-y-2">
              {visibleQualifiedCandidates.map(candidate => (
                <div
                  key={candidate.local_skill_id}
                  className="flex items-start justify-between gap-3 border-t border-border py-2 first:border-0"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{candidateDisplayName(candidate)}</p>
                    {candidateDisplayDescription(candidate) && (
                      <p className="line-clamp-2 text-xs text-text-secondary">
                        {candidateDisplayDescription(candidate)}
                      </p>
                    )}
                    <p className="text-xs text-text-tertiary">{candidateSummary(candidate)}</p>
                    {candidate.professionalism_check && (
                      <div className="mt-1">
                        <WisdomCheckBadge label="Professionalism" value={candidate.professionalism_check} />
                      </div>
                    )}
                  </div>
                  <Button
                    size="sm"
                    outlined
                    disabled={busy === candidate.local_skill_id || candidate.eligibility !== 'eligible'}
                    onClick={() => prepare(candidate)}
                    prefix={busy === candidate.local_skill_id ? <Loader2 className="animate-spin" /> : <Sparkles />}
                  >
                    {candidate.contribution_state === 'prepared' ? copy.continueDraft : copy.prepare}
                  </Button>
                </div>
              ))}
              {visibleQualifiedCandidates.length === 0 && (
                <p className="py-2 text-xs text-text-tertiary">{copy.noSuggestions}</p>
              )}
              {manualCandidates.length > 0 && (
                <div className="border-t border-border py-2">
                  <button
                    aria-expanded={showManualCandidates}
                    className="text-xs font-medium"
                    onClick={() => setShowManualCandidates(value => !value)}
                    type="button"
                  >
                    {copy.browseLocal(manualCandidates.length)}
                  </button>
                  {showManualCandidates && (
                    <>
                      <p className="mb-2 mt-1 text-xs text-text-tertiary">{copy.browseLocalHelp}</p>
                      <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                        {visibleManualCandidates.map(candidate => (
                          <div
                            key={candidate.local_skill_id}
                            className="flex items-start justify-between gap-3 border-t border-border py-2 first:border-0"
                          >
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium">{candidateDisplayName(candidate)}</p>
                              {candidateDisplayDescription(candidate) && (
                                <p className="line-clamp-2 text-xs text-text-secondary">
                                  {candidateDisplayDescription(candidate)}
                                </p>
                              )}
                              <p className="text-xs text-text-tertiary">{candidateSummary(candidate)}</p>
                              {candidate.professionalism_check && (
                                <div className="mt-1">
                                  <WisdomCheckBadge
                                    label="Professionalism"
                                    value={candidate.professionalism_check}
                                  />
                                </div>
                              )}
                            </div>
                            <Button
                              size="sm"
                              outlined
                              disabled={busy === candidate.local_skill_id || candidate.eligibility !== 'eligible'}
                              onClick={() => prepare(candidate)}
                              prefix={
                                busy === candidate.local_skill_id ? <Loader2 className="animate-spin" /> : <Sparkles />
                              }
                            >
                              {candidate.contribution_state === 'prepared' ? copy.continueDraft : copy.prepare}
                            </Button>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium">{copy.ownerReview}</h3>
            <p className="mb-2 mt-1 text-xs text-text-tertiary">{copy.ownerReviewHelp}</p>
            {activeDrafts.length === 0 ? (
              <p className="text-xs text-text-tertiary">{copy.noDrafts}</p>
            ) : (
              activeDrafts.map(draft => (
                <button
                  key={draft.id}
                  type="button"
                  className="flex w-full items-center justify-between border-t border-border py-2 text-left first:border-0 focus-visible:outline focus-visible:outline-2"
                  onClick={() => openReview(draft.id)}
                >
                  <span>
                    <span className="block font-mono text-sm">{draft.slug}</span>
                    <span className="text-xs text-text-tertiary">{copy.draftState(draft.state)}</span>
                  </span>
                  <span className="text-xs">{copy.reviewExact}</span>
                </button>
              ))
            )}
          </div>
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {filtered.map(skill => (
          <button
            key={skill.id}
            type="button"
            onClick={() => openSkill(skill.id)}
            className="min-h-44 border border-border bg-muted/10 p-4 text-left transition-colors hover:bg-muted/30 focus-visible:outline focus-visible:outline-2"
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <span className="font-mono font-semibold">{skill.slug}</span>
              <span className="flex shrink-0 flex-col items-end gap-1">
                <WisdomCheckBadge label="Security" value={skill.security_check} />
                <WisdomCheckBadge label="Professionalism" value={skill.professionalism_check} />
                {pendingUpdates.has(skill.id) && (
                  <span className="text-xs text-amber-500">
                    {copy.updateAvailable(pendingUpdates.get(skill.id)?.plan?.version)}
                  </span>
                )}
              </span>
            </div>
            <p className="line-clamp-3 text-sm text-text-secondary">{skill.author_description || copy.noDescription}</p>
            <div className="mt-4 flex items-center justify-between border-t border-border pt-3 text-xs text-text-tertiary">
              <span>v{skill.latest_version ?? '—'}</span>
              <span>
                {skill.install_count} {copy.managedInstalls}
              </span>
            </div>
          </button>
        ))}
      </div>

      {selected && (
        <div className="border border-border p-4" aria-live="polite">
          <div className="flex items-center justify-between">
            <h3 className="font-mono text-base">
              {String(selected.skill.slug || selected.skill.id || 'Skill detail')}
            </h3>
            <Button size="sm" outlined onClick={() => setSelected(null)}>
              {copy.close}
            </Button>
          </div>
          <WisdomReviewTables
            security={
              asRecord(asRecord(selected.latest_version_detail).version).security_check as
                | WisdomReviewCheck
                | undefined
            }
            professionalism={
              asRecord(asRecord(selected.latest_version_detail).version).professionalism_check as
                | WisdomReviewCheck
                | undefined
            }
          />
          <pre className="mt-4 max-h-80 overflow-auto whitespace-pre-wrap text-xs text-text-secondary">
            {JSON.stringify(
              {
                skill: selected.skill,
                latest_version: selected.latest_version_detail,
                version_history: selected.versions,
                local_compatibility: selected.local_compatibility
              },
              null,
              2
            )}
          </pre>
          {content && (
            <div className="mt-4">
              <p className="break-all font-mono text-[11px]">content {content.content_hash}</p>
              {content.files.map(file => (
                <details key={file.path} className="border-t border-border py-2" open>
                  <summary className="cursor-pointer font-mono text-xs">
                    {file.path} · {file.hash}
                  </summary>
                  <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap text-xs">{file.content_utf8}</pre>
                </details>
              ))}
            </div>
          )}
          <div className="mt-4 flex flex-wrap justify-end gap-2">
            {installed ? (
              <>
                <span className="self-center text-xs text-text-tertiary">
                  {copy.installed(installed.version, installed.update_mode)}
                </span>
                {selectedUpdate && (
                  <span className="self-center text-xs font-medium text-amber-500">
                    {copy.updateAvailable(selectedUpdate.plan?.version)}
                  </span>
                )}
                <Button size="sm" outlined onClick={() => planManagedAction('uninstall')}>
                  {copy.uninstall}
                </Button>
                <Button size="sm" onClick={() => planManagedAction('update')}>
                  {selectedUpdate ? copy.reviewUpdate : copy.checkSkill}
                </Button>
              </>
            ) : (
              <Button size="sm" onClick={() => planManagedAction('install')}>
                {copy.install}
              </Button>
            )}
          </div>
        </div>
      )}

      {actionPlan && (
        <div className="border border-amber-500/50 p-4" role="dialog" aria-label="Verified managed action plan">
          <h3 className="font-mono text-base">{copy.confirmAction(actionPlan.action)}</h3>
          <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap text-xs">
            {JSON.stringify(actionPlan, null, 2)}
          </pre>
          {actionPlan.action === 'install' && (
            <div className="mt-3 max-w-sm">
              <label className="mb-1 block text-xs font-medium" htmlFor="dashboard-wisdom-plan-update-mode">
                {copy.updateModeLabel}
              </label>
              <select
                aria-describedby="dashboard-wisdom-plan-update-mode-help"
                className="h-9 w-full border border-border bg-background px-2 text-xs"
                disabled={busy === 'install-mode'}
                id="dashboard-wisdom-plan-update-mode"
                onChange={event => void replanInstallUpdateMode(event.target.value)}
                value={installUpdateMode}
              >
                <option value="">{copy.updateModeDefault}</option>
                <option value="MANUAL">{copy.updateModeManual}</option>
                <option value="AUTO_WITH_NOTICE">{copy.updateModeAutomatic}</option>
                <option value="REQUIRED">{copy.updateModeRequired}</option>
              </select>
              <p className="mt-1 text-xs text-text-tertiary" id="dashboard-wisdom-plan-update-mode-help">
                {copy.updateModeHelp}
              </p>
            </div>
          )}
          {actionPlan.state === 'current' && <p className="mt-3 text-xs">This managed skill is already current.</p>}
          {actionPlan.compatibility && actionPlan.compatibility.outcome !== 'compatible' && (
            <label className="mt-3 flex gap-2 text-xs">
              <input
                type="checkbox"
                checked={acceptPartial}
                onChange={event => setAcceptPartial(event.target.checked)}
              />
              {copy.acceptCompatibility}
            </label>
          )}
          {(actionPlan.sensitive_expansion?.length ?? 0) > 0 && (
            <label className="mt-2 flex gap-2 text-xs">
              <input
                type="checkbox"
                checked={acceptSensitive}
                onChange={event => setAcceptSensitive(event.target.checked)}
              />
              {copy.acceptSensitive}
            </label>
          )}
          {actionPlan.modified && actionPlan.update_mode !== 'REQUIRED' && (
            <label className="mt-2 flex gap-2 text-xs">
              <input
                type="checkbox"
                checked={preserveModified}
                onChange={event => setPreserveModified(event.target.checked)}
              />
              {copy.preserveModified}
            </label>
          )}
          <div className="mt-4 flex justify-end gap-2">
            <Button
              size="sm"
              outlined
              onClick={() => {
                setActionPlan(null)
                setActionPlanReference(null)
              }}
            >
              Cancel
            </Button>
            {actionPlan.state !== 'current' && (
              <Button disabled={busy === 'install-mode'} size="sm" onClick={applyManagedAction}>
                {busy === 'install-mode' ? copy.planningInstall : copy.confirmAction(actionPlan.action)}
              </Button>
            )}
          </div>
        </div>
      )}

      {prepared && (
        <div className="border border-cyan-500/40 p-4" aria-label="Prepare owner-private Wisdom draft">
          <h3 className="font-mono text-base">{copy.prepareTitle}</h3>
          <p className="mt-1 text-xs text-text-secondary">{copy.prepareNotice}</p>
          <WisdomReviewTables professionalism={prepared.professionalism_check} />
          <label className="mt-4 block text-xs font-medium" htmlFor="wisdom-author-description">
            {copy.ownerDescription}
          </label>
          <textarea
            id="wisdom-author-description"
            className="mt-1 min-h-24 w-full border border-border bg-transparent p-3 text-sm focus-visible:outline focus-visible:outline-2"
            maxLength={4096}
            value={approvedDescription}
            onChange={event => setApprovedDescription(event.target.value)}
          />
          <div className="mt-4 text-xs font-medium">{copy.systemSpecification}</div>
          {approvedSpecification && (
            <div className="mt-2">
              <WisdomSystemSpecificationEditor
                value={approvedSpecification}
                disabled={busy === prepared.local_draft_id}
                onChange={setApprovedSpecification}
              />
            </div>
          )}
          {approvedSpecificationError && (
            <div role="alert" className="mt-3 border border-amber-500/50 bg-amber-500/5 p-3 text-xs text-amber-200">
              {approvedSpecificationError}
            </div>
          )}
          <p className="mt-2 break-all font-mono text-[11px] text-text-tertiary">
            {copy.localOverlay}: {prepared.overlay_path}
          </p>
          <div className="mt-4 grid grid-cols-3 items-center gap-2">
            <div className="justify-self-start">
              <Button size="sm" outlined onClick={() => setPrepared(null)}>
                {copy.cancel}
              </Button>
            </div>
            <div className="justify-self-center">
              <Button
                size="sm"
                disabled={busy === prepared.local_draft_id || !!approvedSpecificationError}
                onClick={async () => {
                  setBusy(prepared.local_draft_id)
                  setError(null)
                  try {
                    if (!approvedSpecification || approvedSpecificationError) {
                      throw new Error(approvedSpecificationError || 'System Specification is unavailable')
                    }
                    await api.suggestWisdomSkill(
                      preparedSkill,
                      profile,
                      approvedDescription,
                      approvedSpecification,
                      preparedSkillId
                    )
                    await refreshContributionData()
                    setPrepared(null)
                  } catch (reason) {
                    setError(userFacingError(reason))
                  } finally {
                    setBusy(null)
                  }
                }}
              >
                {busy === prepared.local_draft_id ? copy.submitting : copy.submit}
              </Button>
            </div>
            <span aria-hidden="true" />
          </div>
        </div>
      )}

      {review && (
        <div className="border border-emerald-500/40 p-4" aria-label="Owner review exact content">
          <h3 className="font-mono text-base">{review.draft.slug}</h3>
          <p className="mt-1 text-xs text-text-secondary">{copy.readEvery}</p>
          {reviewCanEdit && <p className="mt-2 text-xs leading-5 text-text-secondary">{copy.editReview}</p>}
          {reviewDirty && (
            <div role="status" className="mt-3 border border-amber-500/50 bg-amber-500/5 p-3 text-xs text-amber-200">
              {copy.unsavedChanges}
            </div>
          )}
          <div className="mt-3 grid gap-3 border-y border-border py-3 text-xs">
            <div>
              <strong>Owner-authored description (not platform verified)</strong>
              {reviewCanEdit ? (
                <textarea
                  aria-label="Edit owner-authored description"
                  className="mt-2 min-h-24 w-full resize-y border border-border bg-background/40 px-3 py-2 text-sm leading-relaxed focus-visible:border-foreground/25 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30"
                  maxLength={4096}
                  value={reviewDescription}
                  onChange={event => setReviewDescription(event.target.value)}
                />
              ) : (
                <p className="mt-1 whitespace-pre-wrap text-text-secondary">
                  {review.draft.authorDescription || 'No description.'}
                </p>
              )}
            </div>
            <div>
              <strong>Server-enforced scan and server-derived facts</strong>
              <WisdomReviewTables
                security={review.draft.security_check}
                professionalism={review.draft.professionalism_check}
              />
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap text-text-secondary">
                {JSON.stringify(
                  {
                    verdict: review.draft.scanVerdict,
                    scan: review.draft.scan,
                    explanation: review.draft.explanation
                  },
                  null,
                  2
                )}
              </pre>
            </div>
            <div>
              <strong>System Specification (declarative only)</strong>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap text-text-secondary">
                {JSON.stringify(review.draft.systemSpec, null, 2)}
              </pre>
            </div>
          </div>
          <div className="my-3 grid gap-1 font-mono text-[11px]">
            <strong className="font-sans text-xs">{copy.reviewedHashes}</strong>
            <span>content {review.hashes.content}</span>
            <span>author description {review.hashes.author_description}</span>
            <span>package manifest {review.hashes.package_manifest}</span>
          </div>
          {review.files.map(file => (
            <WisdomFileEditor
              key={`${review.draft.id}:${file.path}`}
              file={file}
              value={reviewFiles[file.path] ?? file.content_utf8}
              disabled={!reviewCanEdit || busy === review.draft.id}
              onChange={value => setReviewFiles(current => ({ ...current, [file.path]: value }))}
            />
          ))}
          <div className="mt-4 grid grid-cols-3 items-center gap-2">
            <div className="justify-self-start">
              <Button
                size="sm"
                outlined
                disabled={busy === review.draft.id}
                onClick={async () => {
                  setBusy(review.draft.id)
                  try {
                    await api.decideWisdomDraft(review.draft.id, 'decline', profile)
                    await refreshContributionData()
                    closeReview()
                  } catch (reason) {
                    setError(userFacingError(reason))
                  } finally {
                    setBusy(null)
                  }
                }}
              >
                {copy.decline}
              </Button>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              <Button size="sm" outlined onClick={closeReview}>
                {copy.close}
              </Button>
              {reviewCanEdit && reviewDirty && (
                <>
                  <Button size="sm" outlined disabled={busy === review.draft.id} onClick={resetReviewEdits}>
                    {copy.resetChanges}
                  </Button>
                  <Button
                    size="sm"
                    disabled={busy === review.draft.id || !!reviewManifestError}
                    onClick={saveReviewRevision}
                  >
                    {busy === review.draft.id ? copy.savingRevision : copy.saveAndRescan}
                  </Button>
                </>
              )}
            </div>
            <div className="justify-self-end">
              <Button
                size="sm"
                disabled={busy === review.draft.id || reviewDirty}
                onClick={async () => {
                  setBusy(review.draft.id)
                  try {
                    await api.reviewWisdomDraft(review.draft.id, true, profile)
                    await api.decideWisdomDraft(review.draft.id, 'approve', profile)
                    await refreshContributionData()
                    closeReview()
                  } catch (reason) {
                    setError(userFacingError(reason))
                  } finally {
                    setBusy(null)
                  }
                }}
              >
                {busy === review.draft.id ? copy.publishing : copy.approve}
              </Button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
