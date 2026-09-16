const APPROVAL_CHOICES = ["once", "session", "always", "deny"] as const;

type ApprovalChoice = (typeof APPROVAL_CHOICES)[number];

export interface RemoteApproval {
  allow_permanent?: boolean;
  choices: ApprovalChoice[];
  command: string;
  created_at?: number;
  description: string;
  expires_at?: number;
  profile: string;
  request_id: string;
  session_id: string;
  session_key: string;
  source: string;
  status: "waiting_approval";
  stored_session_id: string;
  title: string;
}

export interface PendingApprovalsResponse {
  approvals?: unknown;
  profile?: unknown;
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeChoices(row: Record<string, unknown>): ApprovalChoice[] {
  const rawChoices = Array.isArray(row.choices) ? row.choices : [];
  const choices = APPROVAL_CHOICES.filter((choice) => rawChoices.includes(choice));
  const filtered = choices.filter((choice) => {
    if (choice === "session" && row.allow_session === false) return false;
    if (choice === "always" && row.allow_permanent === false) return false;
    if ((choice === "session" || choice === "always") && row.smart_denied === true) return false;
    return true;
  });

  // A malformed/legacy response must never manufacture an approval scope. A
  // deny-only action is the safe recovery affordance until the server refreshes.
  return filtered.length > 0 ? filtered : ["deny"];
}

function normalizeOne(row: unknown, profile: string): RemoteApproval | null {
  if (!row || typeof row !== "object" || Array.isArray(row)) return null;
  const value = row as Record<string, unknown>;
  if (
    !nonEmptyString(value.request_id) ||
    value.profile !== profile ||
    !nonEmptyString(value.session_id) ||
    !nonEmptyString(value.session_key) ||
    !nonEmptyString(value.stored_session_id) ||
    value.session_key !== value.stored_session_id
  ) {
    return null;
  }

  const result: RemoteApproval = {
    allow_permanent: value.allow_permanent === false ? false : undefined,
    choices: normalizeChoices(value),
    command: typeof value.command === "string" ? value.command : "",
    description: typeof value.description === "string" ? value.description : "",
    profile,
    request_id: value.request_id,
    session_id: value.session_id,
    session_key: value.session_key,
    source: typeof value.source === "string" && value.source ? value.source : "unknown",
    status: "waiting_approval",
    stored_session_id: value.stored_session_id,
    title: typeof value.title === "string" ? value.title : "",
  };

  if (finiteNumber(value.created_at)) result.created_at = value.created_at;
  if (finiteNumber(value.expires_at)) result.expires_at = value.expires_at;
  return result;
}

export function normalizePendingApprovals(
  response: PendingApprovalsResponse | unknown,
  profile: string,
): RemoteApproval[] {
  if (!response || typeof response !== "object" || !Array.isArray((response as PendingApprovalsResponse).approvals)) {
    return [];
  }
  return ((response as PendingApprovalsResponse).approvals as unknown[])
    .map((row) => normalizeOne(row, profile))
    .filter((row): row is RemoteApproval => row !== null);
}

export function isRemoteApprovalExpired(
  approval: RemoteApproval | null | undefined,
  nowMs = Date.now(),
): boolean {
  return Boolean(approval && finiteNumber(approval.expires_at) && approval.expires_at * 1000 <= nowMs);
}
