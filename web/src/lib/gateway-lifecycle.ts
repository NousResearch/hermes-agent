export interface GatewayRunningStatus {
  gateway_running?: boolean;
}

export interface GatewayActionStatus {
  running: boolean;
  exit_code: number | null;
  pid: number | null;
}

interface PollOptions {
  intervalMs?: number;
  timeoutMs?: number;
  sleep?: (milliseconds: number) => Promise<void>;
}

const defaultSleep = (milliseconds: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, milliseconds));

async function beforeDeadline<T>(
  promise: Promise<T>,
  deadline: number,
  timeoutMessage: string,
): Promise<T> {
  const remainingMs = deadline - Date.now();
  if (remainingMs <= 0) throw new Error(timeoutMessage);

  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const result = await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(timeoutMessage)), remainingMs);
      }),
    ]);
    if (Date.now() >= deadline) throw new Error(timeoutMessage);
    return result;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

async function pollUntil<T>(
  getValue: () => Promise<T>,
  isComplete: (value: T) => boolean,
  timeoutMessage: string,
  options: PollOptions,
): Promise<T> {
  const intervalMs = options.intervalMs ?? 1_000;
  const deadline = Date.now() + (options.timeoutMs ?? 30_000);
  const sleep = options.sleep ?? defaultSleep;

  while (true) {
    const value = await beforeDeadline(getValue(), deadline, timeoutMessage);
    if (Date.now() >= deadline) throw new Error(timeoutMessage);
    if (isComplete(value)) return value;
    await beforeDeadline(sleep(intervalMs), deadline, timeoutMessage);
  }
}

export function waitForGatewayState<T extends GatewayRunningStatus>(
  getStatus: () => Promise<T>,
  expectedRunning: boolean,
  options: PollOptions = {},
): Promise<T> {
  return pollUntil(
    getStatus,
    (status) => status.gateway_running === expectedRunning,
    "Timed out waiting for gateway state transition",
    options,
  );
}

export async function waitForActionCompletion<T extends GatewayActionStatus>(
  getActionStatus: () => Promise<T>,
  expectedPid: number,
  options: PollOptions = {},
): Promise<T> {
  const status = await pollUntil(
    async () => {
      const candidate = await getActionStatus();
      if (candidate.pid !== expectedPid) {
        throw new Error(`Gateway action was replaced by process ${candidate.pid}`);
      }
      return candidate;
    },
    (candidate) => !candidate.running,
    "Timed out waiting for gateway action completion",
    options,
  );
  if (status.exit_code !== 0) {
    throw new Error(`Gateway action failed with exit code ${status.exit_code}`);
  }
  return status;
}
