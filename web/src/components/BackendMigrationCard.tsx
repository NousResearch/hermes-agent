import { Fragment, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowRightLeft } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  applyOutcome,
  backendMigrationApi,
  blockerAdvice,
  consentClears,
  cardVisible,
  findingSentence,
  foundBackend,
  impactLines,
  isInFlight,
  isProseStep,
  proseText,
  isThisRun,
  isUnresolved,
  migrateCommand,
  qualifyCommands,
  readPoll,
  replacesConfigFile,
  stepWords,
  viewOf,
  type BackendMigrationPlan,
  type BackendMigrationPlanResponse,
  type MigrateCommands,
  type PollRef,
  type PollView,
  type Refusal,
} from "@/lib/backend-migration";
import { HERMES_BASE_PATH, getManagementProfile } from "@/lib/api";
import { useI18n, type Translations } from "@/i18n";
import { formatTranslation } from "@/i18n/runtime";

const POLL_MS = 1500;

/** The view a freshly read plan shows by itself: the last run when nothing is left to move, or a run
 *  still going (followed, never offered again). */
function initialView(r: BackendMigrationPlanResponse | null | undefined): PollView | null {
  if (!r) return null;
  if (r.run.state === "running") return viewOf(r.run);
  return r.run.state !== "none" && !foundBackend(r.plan) ? viewOf(r.run) : null;
}

/**
 * System page card for `hermes gateway migrate --backends`: shown only when an older Hermes server
 * (hermes dashboard / serve) serves the managed profile's home, or the last move needs reporting.
 * Hidden otherwise. One plan read per page load (the server answers repeated loads from one scan).
 */
export function BackendMigrationCard({ initial }: { initial?: BackendMigrationPlanResponse | null }) {
  const { t } = useI18n();
  const copy = t.backendMigration;
  const [plan, setPlan] = useState<BackendMigrationPlanResponse | null>(initial ?? null);
  const [confirm, setConfirm] = useState<null | { force: boolean }>(null);
  const [view, setView] = useState<PollView | null>(() => initialView(initial));
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const alive = useRef(true);
  // Set synchronously on the first click, so a second click (or a second confirm) never sends a second apply.
  const applying = useRef(false);
  // The run the card follows (kept for "Check again"), and the poll loop that follows it.
  const [followed, setFollowed] = useState<PollRef | null>(null);
  const loop = useRef(0);

  /** Poll the same URL across the swap until the run `ref` names ends or its outcome cannot be known. */
  const follow = useCallback(async (pollRef: PollRef) => {
    setFollowed(pollRef);
    const id = ++loop.current;
    const startedAt = Date.now();
    let heardAt = startedAt;
    for (;;) {
      await new Promise((r) => setTimeout(r, POLL_MS));
      if (!alive.current || loop.current !== id) return;
      const read = await backendMigrationApi.status();
      if (!alive.current || loop.current !== id) return;
      const now = Date.now();
      // Only an answer about this run is news; the previous run's record read again is not.
      if (read.kind === "status" && isThisRun(read.status, pollRef)) heardAt = now;
      const next = readPoll(read, pollRef, { quietSeconds: (now - heardAt) / 1000, sinceSeconds: (now - startedAt) / 1000 });
      setView(next);
      if (!isInFlight(next)) return;
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  /** Follow a run the plan says is still going (a page opened, or Try again read, while a move runs). */
  const follows = useCallback((r: BackendMigrationPlanResponse) => {
    if (r.run.state === "running" && isInFlight(viewOf(r.run))) {
      void follow({ previousStartedAt: r.run.started_at - 1, requestedAt: r.run.started_at });
    }
  }, [follow]);

  // Each plan read is numbered: only the newest one's answer is shown (an older one may arrive later).
  const planReads = useRef(0);
  const adopt = useCallback((r: { status: number; body: BackendMigrationPlanResponse | null }, n: number) => {
    if (!alive.current || n !== planReads.current || r.status !== 200 || !r.body) return;
    setPlan(r.body);
    setRefusal(null);
    setView(initialView(r.body));
    follows(r.body);
  }, [follows]);
  const readPlan = useCallback(() => {
    const n = ++planReads.current;
    void backendMigrationApi.plan().then((r) => adopt(r, n));
  }, [adopt]);

  useEffect(() => {
    const followsInitial = (r: BackendMigrationPlanResponse) => follows(r);
    if (initial !== undefined) {
      if (initial) followsInitial(initial);
      return;
    }
    const n = ++planReads.current;
    void backendMigrationApi.plan().then((r) => adopt(r, n));
  }, [initial, follows, adopt]);

  const profile = plan?.profile ?? plan?.run.profile ?? (getManagementProfile() || null);
  const commands = plan?.commands ?? plan?.run.commands ?? null;

  const move = async (force: boolean) => {
    setConfirm(null);
    if (applying.current || isInFlight(view) || isUnresolved(view)) return;
    applying.current = true;
    setRefusal(null);
    setView({ kind: "starting" });
    // A record newer than the last one seen before asking is this move's.
    const previousStartedAt = plan?.run.started_at ?? 0;
    try {
      // The backend and address the user confirmed: the route refuses (`plan_changed`) when its fresh plan
      // would move anything else, and the card then shows that plan to be confirmed again.
      const shown = plan?.plan;
      const outcome = applyOutcome(await backendMigrationApi.apply({
        ...(force ? { force_interrupt: true } : {}),
        ...(shown?.old_address ? { address: shown.old_address } : {}),
        ...(shown?.expect_token ? { expect: shown.expect_token } : {}),
      }));
      if (outcome.kind === "refused") {
        // The route refused before starting anything: Move may be offered again.
        const fresh = outcome.refusal.plan;
        if (outcome.refusal.error === "plan_changed" && fresh) {
          setPlan((prev) => (prev ? { ...prev, plan: fresh } : prev));
        }
        setView(null);
        setRefusal(outcome.refusal);
        return;
      }
      if (outcome.kind === "started") {
        if (!outcome.confirmed) setView({ kind: "checking" });
        void follow({ previousStartedAt, requestedAt: outcome.requestedAt, confirmed: outcome.confirmed });
        return;
      }
      // Already starting or running (a second click, another window), or the answer was lost: whether a
      // run started is not known, so the status decides. Never "did not start".
      setView({ kind: "checking" });
      void follow({ previousStartedAt, requestedAt: null });
    } finally {
      applying.current = false;
    }
  };

  // A by-hand plan held by chat activity: the forced read-only plan lists its steps (no move is asked for).
  const showStepsAnyway = async () => {
    const r = await backendMigrationApi.plan({ force_interrupt: true });
    if (!alive.current || r.status !== 200 || !r.body) return;
    setPlan(r.body);
  };

  // After a rolled-back run (everything was put back): read the plan again, as a page load does, and show
  // the move as it stands now (or follow a run that started meanwhile).
  const tryAgain = readPlan;

  const checkAgain = () => {
    if (!followed) return;
    setView({ kind: followed.requestedAt !== null && followed.confirmed !== false ? "starting" : "checking" });
    void follow(followed);
  };

  if (!cardVisible(plan) && view === null) return null;
  const p = plan?.plan;
  const busy = isInFlight(view);
  const finished = view !== null && !busy;

  return (
    <Card data-testid="backend-migration-card">
      <CardContent className="flex flex-col gap-3 py-4 text-sm">
        <div className="flex items-start gap-2">
          <ArrowRightLeft className="mt-0.5 h-4 w-4 shrink-0" />
          <div className="flex flex-col gap-1">
            <span className="font-medium">{copy.title}</span>
            {p && foundBackend(p) && <span>{findingSentence(p, copy)}</span>}
          </div>
        </div>

        {p && foundBackend(p) && !finished && (
          <PlanDetails plan={p} profile={profile} busy={busy} onMove={(force) => setConfirm({ force })}
            onShowSteps={() => void showStepsAnyway()} />
        )}

        {refusal && <RefusalNotice refusal={refusal} />}

        {view && <Progress view={view} profile={profile} commands={commands} onCheckAgain={followed ? checkAgain : undefined}
          onTryAgain={tryAgain} />}
      </CardContent>

      <ConfirmDialog
        open={confirm !== null}
        onCancel={() => setConfirm(null)}
        onConfirm={() => void move(confirm?.force ?? false)}
        title={copy.confirmTitle}
        description={confirmText(p, confirm?.force ?? false, copy)}
        confirmLabel={copy.confirmMove}
        cancelLabel={t.common.cancel}
      />
    </Card>
  );
}

function confirmText(p: BackendMigrationPlan | null | undefined, force: boolean, copy: Translations["backendMigration"]): string {
  const stops = (p?.takes_over ?? []).map((item) => ` ${item}`).join("");
  // Forced: the chats and gateways the plan named are stopped; messaging is down until the gateway service runs.
  return formatTranslation(copy.confirmBody, {
    address: p?.old_address ?? copy.thisAddress,
    forced: force ? copy.confirmInterrupt : "",
    stops,
  });
}

/** What the move would change, what blocks it, and the Move / Move anyway buttons. */
function PlanDetails({ plan: p, profile, busy, onMove, onShowSteps }: {
  plan: BackendMigrationPlan;
  profile: string | null;
  busy: boolean;
  onMove: (force: boolean) => void;
  onShowSteps: () => void;
}) {
  const { t, format } = useI18n();
  const copy = t.backendMigration;
  const all = p.blockers ?? [];
  const byHand = p.unsupported.length > 0;
  // What consent clears is not a blocker on the card: the Move confirmation says it is stopped instead.
  const needsConsent = all.some((b) => consentClears(b, p));
  const blockers = byHand ? all : all.filter((b) => !consentClears(b, p));
  const canForce = byHand && all.length > 0 && all.every((b) => consentClears(b, p));
  return (
    <>
      <ul className="list-disc pl-6 text-muted-foreground">
        {impactLines(p, copy).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {p.unsupported.map((u) => (
        <div key={u.instructions} className="text-warning">
          {format(copy.unsupported, { instructions: qualifyCommands(u.instructions, profile) })}
          {blockers.length === 0 && !!u.steps?.length && (
            <div className="flex flex-col gap-1">
              <EngineSteps steps={u.steps} profile={null} />
            </div>
          )}
        </div>
      ))}
      {!byHand && !!p.takes_over?.length && (
        <ul className="list-disc pl-6 text-muted-foreground" data-slot="backend-migration-takes-over">
          {p.takes_over.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      )}
      {blockers.map((b) => (
        <div key={b} className="text-warning" data-slot="backend-migration-blocker">
          • {blockerAdvice(b, byHand).text}
        </div>
      ))}
      {!busy && (
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            className="uppercase"
            disabled={blockers.length > 0 || p.unsupported.length > 0}
            title={blockers.length ? copy.clearBlockers : undefined}
            onClick={() => onMove(needsConsent)}
          >
            {copy.move}
          </Button>
          {canForce && (
            <Button size="sm" ghost className="uppercase text-warning" onClick={onShowSteps}>
              {copy.showSteps}
            </Button>
          )}
        </div>
      )}
    </>
  );
}

function RefusalNotice({ refusal }: { refusal: Refusal }) {
  return (
    <div className="text-warning" data-slot="backend-migration-refusal">
      {refusal.message}
      {refusal.blockers?.map((b) => (
        <div key={b}>• {blockerAdvice(b).text}</div>
      ))}
    </div>
  );
}

function CommandText({ template, command }: { template: string; command: string }) {
  return template.split(/(\{command\})/).map((part, index) => (
    <Fragment key={index}>
      {part === "{command}" ? <code className="select-all">{command}</code> : part}
    </Fragment>
  ));
}

function Progress({ view, profile, commands, onCheckAgain, onTryAgain }: {
  view: PollView;
  profile: string | null;
  commands: MigrateCommands | null;
  onCheckAgain?: () => void;
  onTryAgain?: () => void;
}) {
  const { t, format } = useI18n();
  const copy = t.backendMigration;
  const status = migrateCommand(profile, "--status", commands);
  const checkAgain = onCheckAgain && (
    <div>
      <Button size="sm" ghost className="uppercase" onClick={onCheckAgain}>
        {copy.checkAgain}
      </Button>
    </div>
  );
  switch (view.kind) {
    case "starting":
      return <Line spin>{copy.starting}</Line>;
    case "checking":
      return <Line spin>{copy.checking}</Line>;
    case "running":
      return <Line spin>{format(copy.running, { step: stepWords(view.step, copy) })}</Line>;
    case "switching":
      return (
        <div className="flex flex-col gap-2">
          <Line spin>{copy.switching}</Line>
          <span className="text-muted-foreground">
            {copy.longWait}{" "}<CommandText template={copy.statusCommand} command={status} />
          </span>
        </div>
      );
    case "signInAgain":
      return (
        <div className="flex flex-col gap-2" data-slot="backend-migration-signin">
          <span>{copy.signInAgainDetail}</span>
          <div>
            <Button size="sm" className="uppercase" onClick={() => window.location.assign(`${HERMES_BASE_PATH}/`)}>
              {copy.signInAgain}
            </Button>
          </div>
        </div>
      );
    case "unknown":
    case "unreachable":
      return (
        <div className="flex flex-col gap-2 text-warning" data-slot="backend-migration-unknown">
          <span>
            {view.kind === "unknown" ? copy.startUnknown : copy.endUnknown}{" "}
            <CommandText template={copy.statusCommand} command={status} />
          </span>
          {checkAgain}
        </div>
      );
    case "stalled":
      return (
        <div className="flex flex-col gap-2 text-warning" data-slot="backend-migration-stalled">
          <span>
            {format(copy.stalled, { step: stepWords(view.step, copy).toLowerCase() })}{" "}
            <CommandText template={copy.restoreCommand} command={migrateCommand(profile, "--restore", commands)} />
          </span>
          {checkAgain}
        </div>
      );
    case "done":
      return (
        <div className="text-success" data-slot="backend-migration-done">
          {view.address ? format(copy.doneAt, { address: view.address }) : copy.doneSame}{" "}
          {copy.signInAtSameAddress}{" "}
          <CommandText template={copy.undoCommand} command={migrateCommand(profile, "--restore", commands)} />
        </div>
      );
    case "rolledBack":
      return (
        <div className="flex flex-col gap-2 text-warning" data-slot="backend-migration-rolled-back">
          <span>{format(copy.rolledBack, { reason: view.reason })}</span>
          {onTryAgain && (
            <div>
              <Button size="sm" ghost className="uppercase" onClick={onTryAgain}>
                {copy.tryAgain}
              </Button>
            </div>
          )}
        </div>
      );
    case "manual":
      return (
        <div className="flex flex-col gap-1 text-destructive" data-slot="backend-migration-manual">
          <span>{format(copy.incomplete, { reason: view.reason })}</span>
          {view.detail && <span>{qualifyCommands(view.detail, profile)}</span>}
          <EngineSteps steps={view.steps} profile={profile} />
        </div>
      );
    case "restored":
      return <div>{copy.restored}</div>;
    case "notStarted":
      return (
        <div className="flex flex-col gap-1 text-warning" data-slot="backend-migration-not-started">
          <span>{copy.notStarted}</span>
          <pre className="whitespace-pre-wrap text-xs">{view.lines.map((line) => qualifyCommands(line, profile)).join("\n")}</pre>
        </div>
      );
  }
}

/** Engine steps to follow by hand: the save-both-files warning first when a step replaces or removes
 *  config.yaml or .env, then the steps in order (prose as words, commands as code). *profile*: qualify the
 *  `hermes` commands (null: shown as the server gave them). */
function EngineSteps({ steps, profile }: { steps: string[]; profile: string | null }) {
  const { t } = useI18n();
  if (steps.length === 0) return null;
  return (
    <>
      {steps.some(replacesConfigFile) && <span>{t.backendMigration.saveConfig}</span>}
      <span>{steps.some(isProseStep) ? t.backendMigration.followSteps : t.backendMigration.runLocally}</span>
      <ol className="list-decimal pl-6">
        {steps.map((s, index) => (
          <li key={index}>
            {isProseStep(s) ? qualifyCommands(proseText(s), profile) : <code>{qualifyCommands(s, profile)}</code>}
          </li>
        ))}
      </ol>
    </>
  );
}

function Line({ children, spin }: { children: ReactNode; spin?: boolean }) {
  return (
    <div className="flex items-center gap-2" data-slot="backend-migration-progress">
      {spin && <Spinner className="h-3.5 w-3.5" />}
      <span>{children}</span>
    </div>
  );
}
