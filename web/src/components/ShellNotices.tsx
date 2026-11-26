import type { StatusResponse } from "@/lib/api";
import { DashboardMovingModal } from "@/components/DashboardMovingModal";
import { MemoryPressureBanner } from "@/components/MemoryPressureBanner";
import { MultiplexStandaloneBanner } from "@/components/MultiplexStandaloneBanner";
import { ProfileScopeBanner } from "@/components/ProfileScopeBanner";
import { SharedMetricsConsentBanner } from "@/components/SharedMetricsConsentBanner";

/** The app shell's notices: the banner stack above the content, and the Hermes Cloud modal. */
export function ShellNotices({ status }: { status: StatusResponse | null }) {
  return (
    <>
      <ProfileScopeBanner />
      <MemoryPressureBanner status={status} />
      <MultiplexStandaloneBanner status={status} />
      <SharedMetricsConsentBanner />
      <DashboardMovingModal status={status} />
    </>
  );
}
