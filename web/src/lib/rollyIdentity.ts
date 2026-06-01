export type RollyDashboardUser = {
  slug: string;
  label: string;
  admin?: boolean;
};

export const ROLLY_DASHBOARD_USERS: RollyDashboardUser[] = [
  { slug: "deniz", label: "Deniz", admin: true },
  { slug: "arman", label: "Arman", admin: true },
  { slug: "buket", label: "Buket" },
  { slug: "metin", label: "Metin" },
  { slug: "guest", label: "Guest" },
];

const STORAGE_KEY = "rolly-dashboard-user";
const COOKIE_KEY = "rolly_user";
const USER_SLUGS = new Set(ROLLY_DASHBOARD_USERS.map((user) => user.slug));

export function normalizeRollyUserSlug(value: string | null | undefined): string {
  const slug = (value ?? "").trim().toLowerCase();
  return USER_SLUGS.has(slug) ? slug : "";
}

function readCookie(): string {
  if (typeof document === "undefined") return "";
  const prefix = `${COOKIE_KEY}=`;
  const match = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix));
  return normalizeRollyUserSlug(match?.slice(prefix.length));
}

export function getRollyUserSlug(): string {
  if (typeof window === "undefined") return "";
  return normalizeRollyUserSlug(window.localStorage.getItem(STORAGE_KEY)) || readCookie();
}

export function getRollyUser(slug = getRollyUserSlug()): RollyDashboardUser | null {
  const normalized = normalizeRollyUserSlug(slug);
  return ROLLY_DASHBOARD_USERS.find((user) => user.slug === normalized) ?? null;
}

export function setRollyUserSlug(slug: string): string {
  const normalized = normalizeRollyUserSlug(slug);
  if (typeof window === "undefined" || !normalized) return normalized;
  window.localStorage.setItem(STORAGE_KEY, normalized);
  document.cookie = `${COOKIE_KEY}=${normalized}; path=/; max-age=31536000; SameSite=Lax`;
  window.dispatchEvent(new CustomEvent("rolly-user-change", { detail: normalized }));
  return normalized;
}
