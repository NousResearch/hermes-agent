import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowRightLeft } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  applyOutcome,
  backendMigrationApi,
  blockerAdvice,
  cardVisible,
  findingSentence,
  foundBackend,
  impactLines,
  isInFlight,
  isThisRun,
  isUnresolved,
  migrateCommand,
  qualifyCommands,
  readPoll,
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

  useEffect(() => {
    const follows = (r: BackendMigrationPlanResponse) => {
      if (r.run.state === "running" && isInFlight(viewOf(r.run))) {
        void follow({ previousStartedAt: r.run.started_at - 1, requestedAt: r.run.started_at });
      }
    };
    if (initial !== undefined) {
      if (initial) follows(initial);
      return;
    }
    void backendMigrationApi.plan().then((r) => {
      if (!alive.current || r.status !== 200 || !r.body) return;
      setPlan(r.body);
      setView(initialView(r.body));
      follows(r.body);
    });
  }, [initial, follow]);

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
            <span className="font-medium">Move to a single Hermes</span>
            {p && foundBackend(p) && <span>{findingSentence(p)}</span>}
          </div>
        </div>

        {p && foundBackend(p) && !finished && (
          <PlanDetails plan={p} profile={profile} busy={busy} onMove={(force) => setConfirm({ force })} />
        )}

        {refusal && <RefusalNotice refusal={refusal} />}

        {view && <Progress view={view} profile={profile} commands={commands} onCheckAgain={followed ? checkAgain : undefined} />}
      </CardContent>

      <ConfirmDialog
        open={confirm !== null}
        onCancel={() => setConfirm(null)}
        onConfirm={() => void move(confirm?.force ?? false)}
        title="Move to a single Hermes?"
        description={confirmText(p?.old_address, confirm?.force ?? false, p?.running_turns === null)}
        confirmLabel="Move now"
      />
    </Card>
  );
}

function confirmText(oldAddress: string | null | undefined, force: boolean, turnsUnknown: boolean): string {
  // Forced past unknown activity: the engine's own warning, never read as idle.
  const forced = turnsUnknown
    ? " The old backend's turn activity could not be read: the move goes ahead and any turn running on it is interrupted."
    : " Replies being written right now are cut off.";
  return (
    `This page stops working while Hermes takes over ${oldAddress ?? "this address"}; ` +
    "afterwards you sign in again at this same address. If anything goes wrong, everything is put back." +
    (force ? forced : "")
  );
}

/** What the move would change, what blocks it, and the Move / Move anyway buttons. */
function PlanDetails({ plan: p, profile, busy, onMove }: {
  plan: BackendMigrationPlan;
  profile: string | null;
  busy: boolean;
  onMove: (force: boolean) => void;
}) {
  const blockers = p.blockers ?? [];
  const canForce = blockers.length > 0 && blockers.every((b) => blockerAdvice(b).canForce);
  return (
    <>
      <ul className="list-disc pl-6 text-muted-foreground">
        {impactLines(p).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {p.unsupported.map((u) => (
        <div key={u.instructions} className="text-warning">
          This server cannot be moved from here. {qualifyCommands(u.instructions, profile)}
          {blockers.length === 0 && !!u.steps?.length && (
            <ol className="list-decimal pl-6">
              {u.steps.map((step, index) => <li key={index}><code>{step}</code></li>)}
            </ol>
          )}
        </div>
      ))}
      {blockers.map((b) => (
        <div key={b} className="text-warning" data-slot="backend-migration-blocker">
          • {blockerAdvice(b).text}
        </div>
      ))}
      {!busy && (
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            className="uppercase"
            disabled={blockers.length > 0 || p.unsupported.length > 0}
            title={blockers.length ? "Sort out the points above first" : undefined}
            onClick={() => onMove(false)}
          >
            Move
          </Button>
          {canForce && (
            <Button size="sm" ghost className="uppercase text-warning" onClick={() => onMove(true)}>
              Move anyway
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

function Cmd({ children }: { children: string }) {
  return <code className="select-all">{children}</code>;
}

function Progress({ view, profile, commands, onCheckAgain }: {
  view: PollView;
  profile: string | null;
  commands: MigrateCommands | null;
  onCheckAgain?: () => void;
}) {
  const status = migrateCommand(profile, "--status", commands);
  const checkAgain = onCheckAgain && (
    <div>
      <Button size="sm" ghost className="uppercase" onClick={onCheckAgain}>
        Check again
      </Button>
    </div>
  );
  switch (view.kind) {
    case "starting":
      return <Line spin>Starting the move…</Line>;
    case "checking":
      return <Line spin>Finding out whether the move started…</Line>;
    case "running":
      return <Line spin>{stepWords(view.step)}…</Line>;
    case "switching":
      return <Line spin>Switching over: this address is unavailable for a few seconds…</Line>;
    case "signInAgain":
      return (
        <div className="flex flex-col gap-2" data-slot="backend-migration-signin">
          <span>
            This address restarted, so your sign-in ended. Sign in again at this same address to see how the move
            ended: if it worked, you will see the new Hermes web app.
          </span>
          <div>
            <Button size="sm" className="uppercase" onClick={() => window.location.assign(`${HERMES_BASE_PATH}/`)}>
              Sign in again
            </Button>
          </div>
        </div>
      );
    case "unknown":
    case "unreachable":
      return (
        <div className="flex flex-col gap-2 text-warning" data-slot="backend-migration-unknown">
          <span>
            {view.kind === "unknown"
              ? "It is not known yet whether the move started."
              : "This address has not answered for a while, so how the move ended is not known."}{" "}
            On this computer, see where it stands with <Cmd>{status}</Cmd>
          </span>
          {checkAgain}
        </div>
      );
    case "stalled":
      return (
        <div className="flex flex-col gap-2 text-warning" data-slot="backend-migration-stalled">
          <span>
            The move stopped part-way ({stepWords(view.step).toLowerCase()}). On this computer, run{" "}
            <Cmd>{migrateCommand(profile, "--restore", commands)}</Cmd> to put everything back.
          </span>
          {checkAgain}
        </div>
      );
    case "done":
      return (
        <div className="text-success" data-slot="backend-migration-done">
          Done. Hermes now answers {view.address ? `at ${view.address}` : "at the same address"}. Sign in again at this
          same address. To undo it, run{" "}
          <Cmd>{migrateCommand(profile, "--restore", commands)}</Cmd> on this computer.
        </div>
      );
    case "rolledBack":
      return (
        <div className="text-warning" data-slot="backend-migration-rolled-back">
          The move did not work, so everything was put back the way it was. Reason: {view.reason}
        </div>
      );
    case "manual":
      return (
        <div className="flex flex-col gap-1 text-destructive" data-slot="backend-migration-manual">
          <span>The move failed and could not be fully undone ({view.reason}).</span>
          {view.detail && <span>{qualifyCommands(view.detail, profile)}</span>}
          <span>On this computer, run:</span>
          <ol className="list-decimal pl-6">
            {view.steps.map((s) => (
              <li key={s}>
                <code>{qualifyCommands(s, profile)}</code>
              </li>
            ))}
          </ol>
        </div>
      );
    case "restored":
      return <div>The earlier move was undone; the older server is back.</div>;
    case "notStarted":
      return (
        <div className="flex flex-col gap-1 text-warning" data-slot="backend-migration-not-started">
          <span>The move did not start. Its last words:</span>
          <pre className="whitespace-pre-wrap text-xs">{view.lines.map((line) => qualifyCommands(line, profile)).join("\n")}</pre>
        </div>
      );
  }
}

function Line({ children, spin }: { children: ReactNode; spin?: boolean }) {
  return (
    <div className="flex items-center gap-2" data-slot="backend-migration-progress">
      {spin && <Spinner className="h-3.5 w-3.5" />}
      <span>{children}</span>
    </div>
  );
}
