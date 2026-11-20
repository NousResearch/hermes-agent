// @vitest-environment jsdom
// The System page's "move to a single Hermes" card and the poll reading behind it.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

import { BackendMigrationCard } from "./BackendMigrationCard";
import { containerPlans, systemSystemdPlans } from "./BackendMigrationCard.plans.fixture";
import {
  killedWhileWritingStatus,
  lateTurnStatus,
  refusalBlockers,
  refusedUrlPlans,
  relativeLauncherStatus,
  restoreRefusedStatus,
  unknownActivityBlocker,
  unknownStateStatus,
} from "./BackendMigrationCard.engine.fixture";
import {
  applyOutcome,
  blockerAdvice,
  readPoll,
  type BackendMigrationPlanResponse,
  type BackendMigrationStatus,
  type PollRef,
} from "@/lib/backend-migration";

let container: HTMLDivElement;
let root: Root;

async function render(initial: BackendMigrationPlanResponse) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<BackendMigrationCard initial={initial} />));
}

const none: BackendMigrationStatus = {
  state: "none", step: "", steps: [], message: "", old_address: "", new_address: "", started_at: 0, finished_at: null,
};

function planWith(
  overrides: Partial<BackendMigrationPlanResponse["plan"]>,
  run = none,
  profile: string | null = "work",
): BackendMigrationPlanResponse {
  return {
    exit_code: 0,
    run,
    profile,
    plan: {
      old_address: "0.0.0.0:9119", new_address: "0.0.0.0:9119",
      old_backend: { supervisor: "systemd", unit: "hermes-dashboard.service", scope: "user", pid: 1, was_active: true },
      keep_webapp_address: false, copy_env_keys: [], generate_secret: true, public_url: null,
      auth_before: { auth_required: true, providers: ["basic"] }, gateway: { installed: false, active: false },
      linger: false, webapp_bundle_missing: "missing", running_turns: 0, blockers: [], unsupported: [], notices: [],
      nothing_to_do: false, text: [], ...overrides,
    },
  };
}

const button = (label: string) => [...document.querySelectorAll("button")].find((b) => b.textContent === label);
const moveButton = () => button("Move");

/** A server for the card: `apply` answers as each test says; `status` returns the queued records (the
 *  last one again and again). */
function server(apply: (init?: RequestInit) => Promise<Response>, statuses: BackendMigrationStatus[]) {
  const calls: string[] = [];
  vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
    const url = String(input);
    calls.push(`${init?.method ?? "GET"} ${url}`);
    if (url.includes("/apply")) return apply(init);
    const next = statuses.length > 1 ? statuses.shift()! : statuses[0];
    return new Response(JSON.stringify(next), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  return { applies: () => calls.filter((c) => c.startsWith("POST") && c.includes("/apply")).length };
}

const json = (status: number, body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

async function confirmMove(times = 1) {
  await act(async () => moveButton()!.click());
  await act(async () => {
    for (let i = 0; i < times; i++) button("Move now")!.click();
  });
}

async function wait(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("BackendMigrationCard", () => {
  it("is hidden when no older server was found and no run needs reporting", async () => {
    await render(planWith({ nothing_to_do: true }));
    expect(container.querySelector('[data-testid="backend-migration-card"]')).toBeNull();
  });

  it("stays hidden for a blocker without any old backend, and shows only the run when one needs reporting", async () => {
    const blockerOnly = { old_backend: null, old_address: null, new_address: null, nothing_to_do: false, found: false,
      blockers: ["/h/config.yaml cannot be read or parsed (ParserError): fix it and re-run the dry run. Nothing changed."] };
    await render(planWith(blockerOnly));
    expect(container.querySelector('[data-testid="backend-migration-card"]')).toBeNull();
    await act(async () => root.unmount());
    container.remove();

    await render(planWith(blockerOnly, { ...none, state: "rolled_back", started_at: 10, message: "verify failed; rolled back" }));
    expect(container.querySelector('[data-slot="backend-migration-rolled-back"]')).not.toBeNull();
    expect(container.textContent).not.toContain("An older Hermes server is running");
    expect(moveButton()).toBeUndefined();
  });

  it("says what was found in plain words and offers the move", async () => {
    await render(planWith({}));
    expect(container.textContent).toContain(
      "An older Hermes server is running on port 9119. Phones and browsers that use it talk to a second copy of Hermes.",
    );
    expect(container.textContent).toContain("The same username and password keep working.");
    expect(moveButton()?.disabled).toBe(false);
  });

  it("holds the move while a blocker stands and says what to do", async () => {
    await render(planWith({
      blockers: ["1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted)."],
    }));
    expect(moveButton()?.disabled).toBe(true);
    expect(container.textContent).toContain("Wait for them to finish. Or move anyway");
    expect(container.textContent).not.toContain("--force-interrupt");
    expect(button("Move anyway")).toBeDefined();
  });

  it("shows unknown turn activity as the engine's blocker, and Move anyway with the engine's warning", async () => {
    // The plan route's blocker for an unreadable turn-marker file: it names the file.
    const marker = unknownActivityBlocker.split("checked: ")[1].split(". To move it")[0];
    await render(planWith({ running_turns: null, blockers: [unknownActivityBlocker] }));
    expect(moveButton()?.disabled).toBe(true);
    expect(container.textContent).toContain(`${marker}. To move it, confirm it is idle, then move anyway (any reply `
      + "being written is cut off).");
    expect(container.textContent).not.toContain("--force-interrupt");
    await act(async () => button("Move anyway")!.click());
    expect(document.body.textContent).toContain(
      "The old backend's turn activity could not be read: the move goes ahead and any turn running on it is interrupted.");
  });

  it("shows what an incomplete rollback could not do, in the engine's words, next to the steps", async () => {
    const detail = "restored config.yaml: /h/config.yaml changed since the migration wrote it, so it was not written "
      + "back and your edits were kept; merge them with /h/backend_migration/config.yaml.bak by hand; did not start "
      + "the old backend again: this background run is in hermes-backend-migration-1.service, which stops what it "
      + "starts when the run ends; start it again from a terminal, without --detach: hermes gateway migrate --backends --restore";
    await render(planWith({ nothing_to_do: true }, {
      ...none, state: "failed", step: "verify", started_at: 10, finished_at: 20,
      message: `verify failed: no HTTP 200; rollback incomplete: ${detail}`,
      reason: "verify failed: no HTTP 200", manual_steps: ["hermes -p work gateway stop"],
    }));
    const manual = container.querySelector('[data-slot="backend-migration-manual"]')?.textContent ?? "";
    expect(manual).toContain("(verify failed: no HTTP 200)");
    expect(manual).toContain("so it was not written back and your edits were kept; merge them with /h/backend_migration/config.yaml.bak by hand");
    expect(manual).toContain("start it again from a terminal, without --detach: hermes -p work gateway migrate --backends --restore");
  });

  it("shows the engine's per-file notes as words to follow, never as commands to copy", async () => {
    const notes = [
      "/h/config.yaml was restored from its backup (nothing to do for it)",
      "keep /h/.env as it is (it was edited after the migration wrote it): merge the lines you need from "
        + "/h/backend_migration/env.bak into it by hand before you start the old backend",
    ];
    const missing = "the backup /h/backend_migration/env.bak of .env is missing or cannot be read: recover it, or "
      + "put /h/.env back "
      + "as it was before the migration by hand, before you start the old backend";
    const commands = ["hermes -p work gateway uninstall", "systemctl --user start hermes-dashboard.service"];
    await render(planWith({ nothing_to_do: true }, {
      ...none, state: "failed", step: "verify", started_at: 10, finished_at: 20,
      message: "verify failed; rollback incomplete: x", reason: "verify failed",
      manual_steps: [commands[0], ...notes, missing, commands[1]],
    }));
    const manual = container.querySelector('[data-slot="backend-migration-manual"]')!;
    expect([...manual.querySelectorAll("li")].map((li) => li.textContent))
      .toEqual([commands[0], ...notes, missing, commands[1]]);
    expect([...manual.querySelectorAll("li code")].map((c) => c.textContent)).toEqual(commands);
  });

  describe("a by-hand plan blocked by chat activity", () => {
    // The real plan route's answers (see BackendMigrationCard.plans.fixture.ts): unforced, then with
    // ?force_interrupt=true, for a container backend and a systemd system unit on Linux.
    const asPlan = (payload: unknown) => payload as BackendMigrationPlanResponse;
    const cases: [string, { held: unknown; forced: unknown }][] = [
      ["a running chat on a container backend", containerPlans.running],
      ["a running chat on a systemd system unit", systemSystemdPlans.running],
      ["unknown chat activity on a container backend", containerPlans.unknown],
      ["unknown chat activity on a systemd system unit", systemSystemdPlans.unknown],
    ];
    it.each(cases)("with %s offers to show the steps anyway: one forced plan read, never a move", async (_what, payloads) => {
      const initial = asPlan(payloads.held);
      const forced = asPlan(payloads.forced);
      const [entry] = forced.plan.unsupported;
      const calls: string[] = [];
      vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
        calls.push(`${init?.method ?? "GET"} ${String(input)}`);
        return new Response(JSON.stringify(forced), { status: 200, headers: { "Content-Type": "application/json" } });
      }));
      await render(initial);
      expect(button("Move anyway")).toBeUndefined();
      expect(container.textContent).toMatch(/[Oo]r show the steps anyway|then show the steps anyway/);
      expect(container.textContent).not.toContain("move anyway");
      expect(container.textContent).not.toContain("--force-interrupt");
      expect(container.querySelectorAll("ol li code")).toHaveLength(0);
      await act(async () => button("Show the steps anyway")!.click());
      expect(calls).toHaveLength(1);
      expect(calls[0]).toMatch(/^GET .*\/api\/gateway\/migrate\/backends\/plan\?force_interrupt=true/);
      // The engine's instructions, verbatim, whatever they say.
      expect(container.textContent).toContain(entry.instructions);
      expect([...container.querySelectorAll("ol li code")].map((c) => c.textContent)).toEqual(entry.steps);
      expect(button("Show the steps anyway")).toBeUndefined();
    });

    // The engine adds the running-turn sentence to every forced by-hand plan, on Linux as on Windows.
    it.each(cases.slice(0, 2))("with %s says the steps cut the running reply off once they are shown", async (_what, payloads) => {
      vi.stubGlobal("fetch", vi.fn<typeof fetch>(async () => new Response(JSON.stringify(payloads.forced),
        { status: 200, headers: { "Content-Type": "application/json" } })));
      await render(asPlan(payloads.held));
      await act(async () => button("Show the steps anyway")!.click());
      expect(container.textContent).toContain("1 turn(s) are running on it now: stopping it interrupts them");
    });

    it("leaves a movable plan's Move anyway as it is", async () => {
      await render(planWith({ blockers: asPlan(containerPlans.running.held).plan.blockers }));
      expect(button("Move anyway")).toBeDefined();
      expect(button("Show the steps anyway")).toBeUndefined();
    });
  });

  it("reports a rolled-back run with its reason even when nothing is left to move", async () => {
    await render(planWith({ nothing_to_do: true }, {
      ...none, state: "rolled_back", step: "verify", started_at: 10, finished_at: 20,
      message: "verify failed: no HTTP 200 on 127.0.0.1:9119/api/status; rolled back, old backend verified on 0.0.0.0:9119",
    }));
    expect(container.querySelector('[data-slot="backend-migration-rolled-back"]')?.textContent).toContain(
      "Reason: verify failed: no HTTP 200 on 127.0.0.1:9119/api/status",
    );
  });

  it.each([["config.yaml", killedWhileWritingStatus.config], [".env", killedWhileWritingStatus.env]])(
    "shows a run killed while writing %s as incomplete, its note as words and only the commands as code",
    async (_name, record) => {
      const killed = record as unknown as BackendMigrationStatus;
      const steps = killed.manual_steps!;
      const note = steps.find((s) => s.startsWith("the migration stopped while writing "))!;
      await render(planWith({ nothing_to_do: true }, killed));
      const manual = container.querySelector('[data-slot="backend-migration-manual"]')!;
      expect(manual.textContent).toContain("so it may hold the migration's changes; it was not written back");
      expect([...manual.querySelectorAll("li")].map((li) => li.textContent)).toEqual(steps);
      expect([...manual.querySelectorAll("li code")].map((c) => c.textContent))
        .toEqual(steps.filter((s) => s.startsWith("hermes ") || s.startsWith("systemctl ")));
      expect(note).toBeDefined();
    });

  it("shows an unrecognized record state in the engine's words, with no empty list of steps", async () => {
    const record = unknownStateStatus as unknown as BackendMigrationStatus;
    await render(planWith({ nothing_to_do: true }, record));
    const manual = container.querySelector('[data-slot="backend-migration-manual"]')!;
    expect(manual.textContent).toContain(record.message);
    expect(manual.textContent).not.toContain("On this computer, run:");
    expect(manual.querySelectorAll("ol")).toHaveLength(0);
    expect(button("Move anyway")).toBeUndefined();
  });

  // The plan route's blockers for the engine's newer preflight refusals: shown as given, Move held, and never
  // Move anyway (none of them is cleared by --force-interrupt).
  it.each(Object.entries(refusalBlockers))("shows the %s refusal in the engine's words, with no Move anyway", async (_what, b) => {
    await render(planWith({ blockers: [b] }));
    expect(container.querySelector('[data-slot="backend-migration-blocker"]')?.textContent).toBe(`• ${b}`);
    expect(moveButton()?.disabled).toBe(true);
    expect(button("Move anyway")).toBeUndefined();
  });

  it("never offers to show the steps of a by-hand plan held by a refused public URL", async () => {
    const held = refusedUrlPlans.held as unknown as BackendMigrationPlanResponse;
    const [blocker] = held.plan.blockers;
    expect(blocker).toContain("public URL");
    expect((refusedUrlPlans.forced as unknown as BackendMigrationPlanResponse).plan.blockers).toEqual([blocker]);
    await render(held);
    expect(container.querySelector('[data-slot="backend-migration-blocker"]')?.textContent).toBe(`• ${blocker}`);
    expect(button("Show the steps anyway")).toBeUndefined();
    expect(button("Move anyway")).toBeUndefined();
    expect(container.textContent).not.toMatch(/show the steps anyway/i);
  });

  it("shows a background move refused for a relative launcher as not started, in the engine's words", async () => {
    const refused = relativeLauncherStatus as unknown as BackendMigrationStatus;
    await render(planWith({ nothing_to_do: true }, refused));
    const shown = container.querySelector('[data-slot="backend-migration-not-started"]')?.textContent ?? "";
    expect(shown).toContain(refused.message);
    expect(shown).not.toContain("Before running these steps");
    expect(button("Move anyway")).toBeUndefined();
  });

  describe("a chat that starts after the preflight (exit 4, the old backend never stopped)", () => {
    // The status route's records after the engine stopped the move right before the old backend's stop.
    it.each([
      ["a running turn", lateTurnStatus.running,
        "1 turn(s) are running on the old backend. Wait for them to finish, then try again. The old backend has not "
          + "been stopped."],
      ["unreadable activity", lateTurnStatus.unknown,
        "Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, "
          + "confirm it is idle, then try again. The old backend has not been stopped."],
    ])("with %s reads as rolled back with the engine's reason and offers to try again", async (_what, record, reason) => {
      const late = record as unknown as BackendMigrationStatus;
      const calls: string[] = [];
      vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
        const url = String(input);
        calls.push(`${init?.method ?? "GET"} ${url}`);
        if (url.includes("/apply")) {
          return new Response(JSON.stringify({ ok: true, confirmed: true, requested_at: late.started_at }),
            { status: 200, headers: { "Content-Type": "application/json" } });
        }
        const body = url.includes("/plan") ? planWith({}, late) : late;
        return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
      }));
      await render(planWith({}));
      await confirmMove();
      await wait(1600);
      const shown = container.querySelector('[data-slot="backend-migration-rolled-back"]')?.textContent ?? "";
      expect(shown).toContain(`Reason: ${reason}`);
      expect(shown).not.toContain("--force-interrupt");
      expect(shown).not.toContain("gateway_service failed");
      await act(async () => button("Try again")!.click());
      expect(calls.filter((c) => c.startsWith("GET") && c.includes("/plan"))).toHaveLength(1);
      expect(container.querySelector('[data-slot="backend-migration-rolled-back"]')).toBeNull();
      expect(moveButton()?.disabled).toBe(false);
    });
  });

  it("sends one apply for a double confirm and follows a 409 'already running' instead of applying again", async () => {
    const running = { ...none, state: "running" as const, step: "stop_old_backend", started_at: 50, alive: true };
    const srv = server(() => json(409, { ok: false, error: "already_running", message: "A move is already starting." }),
      [running]);
    await render(planWith({}));

    await confirmMove(2);
    expect(srv.applies()).toBe(1);
    expect(container.textContent).toContain("Finding out whether the move started");
    expect(moveButton()).toBeUndefined();
    expect(container.querySelector('[data-slot="backend-migration-refusal"]')).toBeNull();

    await wait(1600);
    expect(container.textContent).toContain("Switching off the older server");
    expect(srv.applies()).toBe(1);
  });

  it("treats a lost apply answer as an unknown outcome, never as 'did not start'", async () => {
    const srv = server(() => Promise.reject(new TypeError("Failed to fetch")), [{ ...none, state: "restored", started_at: 0 }]);
    await render(planWith({}));

    await confirmMove();
    expect(container.textContent).toContain("Finding out whether the move started");
    await wait(65_000);
    const unknown = container.querySelector('[data-slot="backend-migration-unknown"]');
    expect(unknown?.textContent).toContain("It is not known yet whether the move started.");
    expect(unknown?.textContent).toContain("hermes -p work gateway migrate --backends --status");
    expect(container.textContent).not.toContain("did not start");
    expect(moveButton()).toBeUndefined();
    expect(button("Check again")).toBeDefined();
    expect(srv.applies()).toBe(1);
  });

  it("names the profile in the undo command, and in the recovery command of a run whose process is gone", async () => {
    await render(planWith({ nothing_to_do: true }, { ...none, state: "done", started_at: 100, new_address: "0.0.0.0:9119" }));
    expect(container.querySelector('[data-slot="backend-migration-done"]')?.textContent).toContain(
      "hermes -p work gateway migrate --backends --restore",
    );
    await act(async () => root.unmount());
    container.remove();

    // The engine may leave the address out of a record: the card never prints "null".
    await render(planWith({ nothing_to_do: true }, { ...none, state: "done", started_at: 100, new_address: null }));
    expect(container.querySelector('[data-slot="backend-migration-done"]')?.textContent).toContain(
      "Hermes now answers at the same address",
    );
    await act(async () => root.unmount());
    container.remove();

    await render(planWith({ nothing_to_do: true },
      { ...none, state: "running", step: "start_gateway", started_at: 100, alive: false }));
    expect(container.querySelector('[data-slot="backend-migration-stalled"]')?.textContent).toContain(
      "hermes -p work gateway migrate --backends --restore",
    );
  });

  it("shows the server's commands as given: a profile outside the standard root keeps its HERMES_HOME pin", async () => {
    const pinned = "HERMES_HOME=/srv/team hermes -p work gateway migrate --backends";
    await render({
      ...planWith({ nothing_to_do: true }, { ...none, state: "done", started_at: 100, new_address: "0.0.0.0:9119" }),
      commands: { status: `${pinned} --status`, restore: `${pinned} --restore` },
    });
    const done = container.querySelector('[data-slot="backend-migration-done"]')?.textContent ?? "";
    expect(done).toContain(`${pinned} --restore`);
    expect(done).not.toContain("hermes -p work gateway migrate --backends --restore to");
  });

  it("shows a manual plan's own steps in order, as given, only when nothing blocks it", async () => {
    const steps = ["hermes -p work config set webapp.port 9119", "move \"Hermes Dashboard.lnk\" %TEMP%\\"];
    const manual = {
      unsupported: [{ supervisor: "startup", unit: null, instructions: "Run the steps below.", steps }],
    };
    await render(planWith(manual));
    const lines = [...container.querySelectorAll("ol li code")].map((c) => c.textContent);
    expect(lines).toEqual(steps);
    await act(async () => root.unmount());
    container.remove();

    await render(planWith({ ...manual, blockers: ["Something else blocks the move."] }));
    expect(container.querySelectorAll("ol li code")).toHaveLength(0);
  });

  it("never prints an unknown new address", async () => {
    await render(planWith({ new_address: null }));
    expect(container.textContent).toContain("Hermes itself takes over its address.");
    expect(container.textContent).not.toContain("null");
  });
});

describe("readPoll", () => {
  const run = (state: BackendMigrationStatus["state"], started_at = 100): BackendMigrationStatus =>
    ({ ...none, state, started_at, new_address: "0.0.0.0:9119", step: "verify" });
  const asked: PollRef = { previousStartedAt: 50, requestedAt: 100 };
  const lost: PollRef = { previousStartedAt: 50, requestedAt: null };
  const at = (sinceSeconds: number, quietSeconds = sinceSeconds) => ({ sinceSeconds, quietSeconds });

  it("ignores the previous run's final status until the new run writes its own", () => {
    expect(readPoll({ kind: "status", status: run("restored", 50) }, asked, at(2)).kind).toBe("starting");
    // No record for a minute proves nothing: unknown, never "did not start".
    expect(readPoll({ kind: "status", status: run("restored", 50) }, asked, at(90)).kind).toBe("unknown");
    expect(readPoll({ kind: "status", status: run("done") }, asked, at(2))).toEqual({ kind: "done", address: "0.0.0.0:9119" });
  });

  it("never calls a run whose request answer was lost 'not started': it is unknown", () => {
    expect(readPoll({ kind: "status", status: run("restored", 50) }, lost, at(2)).kind).toBe("checking");
    expect(readPoll({ kind: "status", status: run("restored", 50) }, lost, at(90)).kind).toBe("unknown");
    expect(readPoll({ kind: "status", status: { ...run("running", 51), alive: true } }, lost, at(2)).kind).toBe("running");
  });

  it("shows a refused background run as not started with its reason, never as a failed undo", () => {
    const refused = { ...run("failed"), refused: true, message: "the background migration did not start: x" };
    expect(readPoll({ kind: "status", status: refused }, asked, at(2))).toEqual({
      kind: "notStarted",
      lines: ["the background migration did not start: x"],
    });
  });

  it("warns to save config.yaml and .env and keep edits before showing manual steps on a refused run", () => {
    const step = "cp /h/backend_migration/config.yaml.bak /h/config.yaml";
    const refused = {
      ...run("failed"),
      refused: true,
      message: "config.yaml was edited after the move; a restore would overwrite it, so the gateway keeps serving",
      manual_steps: [step],
    };
    const view = readPoll({ kind: "status", status: refused }, asked, at(2));
    expect(view.kind).toBe("notStarted");
    if (view.kind !== "notStarted") return;
    expect(view.lines).toHaveLength(3);
    expect(view.lines[0]).toBe(refused.message);
    // As the CLI says: save both files first, then merge into the backups before, or re-apply after.
    const warning = view.lines[1];
    expect(warning).toMatch(/^Before running these steps, save a copy of your current config\.yaml and \.env/);
    expect(warning).toMatch(/merge them into any backup copy a step copies back \(before you run it\), or re-apply them after the last step/);
    // Also shown for a refused apply, where nothing was moved.
    expect(warning).not.toMatch(/since the move/i);
    expect(view.lines[2]).toBe(step);
  });

  it("leaves the save-both-files warning out when the steps already say what to do with each file", () => {
    const keep = "keep /h/.env as it is (it was edited after the migration wrote it): merge the lines you need from "
      + "/h/backend_migration/env.bak into it by hand before you start the old backend";
    const refused = { ...run("failed"), refused: true, message: "did not start", manual_steps: ["hermes -p work gateway uninstall", keep] };
    const view = readPoll({ kind: "status", status: refused }, asked, at(2));
    expect(view).toEqual({ kind: "notStarted", lines: ["did not start", "hermes -p work gateway uninstall", keep] });
  });

  it.each([
    ["env.bak", ".env", "cp /h/backend_migration/config.yaml.bak /h/config.yaml"],
    ["config.yaml.bak", "config.yaml", "cp /h/backend_migration/env.bak /h/.env && chmod 600 /h/.env"],
  ])("keeps the save-both-files warning when %s is missing and a step still copies the other file back", (bak, name, copy) => {
    const missing = `the backup /h/backend_migration/${bak} of ${name} is missing or cannot be read: recover it, or `
      + `put /h/${name} back `
      + "as it was before the migration by hand, before you start the old backend";
    const refused = { ...run("failed"), refused: true, message: "did not start", manual_steps: [copy, missing] };
    const view = readPoll({ kind: "status", status: refused }, asked, at(2));
    expect(view.kind).toBe("notStarted");
    if (view.kind !== "notStarted") return;
    expect(view.lines[1]).toMatch(/^Before running these steps, save a copy of your current config\.yaml and \.env/);
    expect(view.lines.slice(2)).toEqual([copy, missing]);
  });

  // The engine's own refusal records (BackendMigrationCard.engine.fixture.ts): a move to the gateway, then
  // `restore(host, yes=True, detached=True)` with one or both backups deleted (config.yaml or .env edited too),
  // read back through the status route (exit 3 each time).
  const envBackupMissing = restoreRefusedStatus.envMissing as unknown as BackendMigrationStatus;
  const configBackupMissing = restoreRefusedStatus.configMissing as unknown as BackendMigrationStatus;
  const bothBackupsMissing = restoreRefusedStatus.bothMissing as unknown as BackendMigrationStatus;

  it.each([["the .env backup", envBackupMissing], ["the config.yaml backup", configBackupMissing]])(
    "keeps the save-both-files warning on the engine's restore refusal when %s is missing", (_what, refused) => {
      const view = readPoll({ kind: "status", status: refused }, asked, at(2));
      expect(view.kind).toBe("notStarted");
      if (view.kind !== "notStarted") return;
      expect(view.lines[0]).toBe(refused.message);
      expect(view.lines[1]).toMatch(/^Before running these steps, save a copy of your current config\.yaml and \.env/);
      expect(view.lines.slice(2)).toEqual(refused.manual_steps);
    });

  it("leaves the warning out of the engine's restore refusal when no step copies a backup back", () => {
    const view = readPoll({ kind: "status", status: bothBackupsMissing }, asked, at(2));
    expect(view).toEqual({ kind: "notStarted", lines: [bothBackupsMissing.message, ...bothBackupsMissing.manual_steps!] });
  });

  it("reads the swap's gap as switching and a refused sign-in as sign in again, never as failure", () => {
    expect(readPoll({ kind: "unreachable" }, asked, at(5)).kind).toBe("switching");
    expect(readPoll({ kind: "auth" }, asked, at(5)).kind).toBe("signInAgain");
    expect(readPoll({ kind: "unreachable" }, asked, at(200)).kind).toBe("unreachable");
  });

  it("reads a running record whose process is gone as stalled, from alive or the engine's note", () => {
    expect(readPoll({ kind: "status", status: { ...run("running"), alive: false } }, asked, at(2)).kind).toBe("stalled");
    expect(readPoll({ kind: "status", status: { ...run("running"), alive: true } }, asked, at(2)).kind).toBe("running");
    const old = { ...run("running"), message: "x; interrupted: the next run rolls it back" };
    expect(readPoll({ kind: "status", status: old }, asked, at(2)).kind).toBe("stalled");
  });

  it("prefers the route's reason over the engine's message", () => {
    const rolled = { ...run("rolled_back"), reason: "verify failed", message: "verify failed; rolled back, old backend verified" };
    expect(readPoll({ kind: "status", status: rolled }, asked, at(2))).toEqual({ kind: "rolledBack", reason: "verify failed" });
  });

  it("keeps the engine's blocker sentences but drops flags the card does not offer", () => {
    expect(blockerAdvice("This command runs inside x itself; stopping it would end this run. Re-run with --detach.").text)
      .not.toContain("--detach");
  });
});

describe("applyOutcome", () => {
  it("knows a started move, a refusal, a move already going and a lost answer apart", () => {
    expect(applyOutcome({ status: 200, body: { ok: true, requested_at: 5, old_address: null, new_address: null } }))
      .toEqual({ kind: "started", requestedAt: 5, confirmed: true });
    expect(applyOutcome({ status: 202, body: { ok: true, confirmed: false, requested_at: 5, old_address: null, new_address: null } }))
      .toEqual({ kind: "started", requestedAt: 5, confirmed: false });
    expect(applyOutcome({ status: 409, body: { ok: false, error: "blocked", message: "m" } }).kind).toBe("refused");
    expect(applyOutcome({ status: 409, body: { ok: false, error: "unfinished_run", message: "m" } }).kind).toBe("refused");
    expect(applyOutcome({ status: 409, body: { ok: false, error: "already_running", message: "m" } }).kind).toBe("running");
    expect(applyOutcome({ status: 0, body: null }).kind).toBe("lost");
    expect(applyOutcome({ status: 502, body: null }).kind).toBe("lost");
    expect(applyOutcome({ status: 500, body: { ok: false, error: "spawn_failed", message: "m" } }).kind).toBe("lost");
  });
});

describe("BackendMigrationCard review round", () => {
  it("sends the confirmed backend with the apply and asks again when the server's plan changed", async () => {
    const bodies: unknown[] = [];
    const fresh = { ...planWith({ old_address: "0.0.0.0:9200", new_address: "0.0.0.0:9200" }).plan, expect_token: "t-9200" };
    const srv = server((init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return json(409, { ok: false, error: "plan_changed", message: "What this server would move changed since it was shown.", plan: fresh });
    }, [none]);
    await render(planWith({ expect_token: "t-9119" }));

    await confirmMove();
    expect(bodies).toEqual([{ address: "0.0.0.0:9119", expect: "t-9119" }]);
    expect(container.textContent).toContain("What this server would move changed");
    expect(container.textContent).toContain("running on port 9200");
    expect(moveButton()?.disabled).toBe(false); // a new confirmation is needed, and offered
    expect(srv.applies()).toBe(1);

    // Confirming the new plan sends ITS token, not the old one.
    await confirmMove();
    expect(bodies[1]).toEqual({ address: "0.0.0.0:9200", expect: "t-9200" });
  });

  it("keeps an unconfirmed start unknown until a record says otherwise, never 'did not start'", async () => {
    server(() => json(202, { ok: true, confirmed: false, requested_at: 100, old_address: null, new_address: null }),
      [{ ...none, state: "restored", started_at: 10 }]);
    await render(planWith({}));

    await confirmMove();
    expect(container.textContent).toContain("Finding out whether the move started");
    await wait(30_000);
    expect(container.textContent).toContain("Finding out whether the move started");
    await wait(40_000);
    expect(container.querySelector('[data-slot="backend-migration-unknown"]')?.textContent).toContain(
      "hermes -p work gateway migrate --backends --status",
    );
    expect(container.textContent).not.toContain("did not start");
  });

  it("never shows an engine step without the profile", async () => {
    await render(planWith({ nothing_to_do: true }, {
      ...none, state: "failed", started_at: 10, message: "verify failed; rollback incomplete: x",
      manual_steps: ["hermes gateway stop", "cp /b/config.yaml.bak /h/.hermes/config.yaml", "hermes -p work gateway start"],
    }));
    const manual = container.querySelector('[data-slot="backend-migration-manual"]')?.textContent ?? "";
    expect(manual).toContain("hermes -p work gateway stop");
    expect(manual).toContain("cp /b/config.yaml.bak /h/.hermes/config.yaml");
    expect(manual).toContain("hermes -p work gateway start");
    expect(manual).not.toMatch(/hermes gateway/);
  });
});
