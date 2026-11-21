import type { StatusResponse } from "@/lib/api";
import { ConfigParseErrorBanner } from "./ConfigParseErrorBanner";
import { MemoryPressureBanner } from "./MemoryPressureBanner";
import { MultiplexStandaloneBanner } from "./MultiplexStandaloneBanner";

/** The app-wide warning banners driven by /api/status, in display order. */
export function StatusBanners({ status }: { status: StatusResponse | null }) {
  return (
    <>
      <MemoryPressureBanner status={status} />
      <MultiplexStandaloneBanner status={status} />
      <ConfigParseErrorBanner status={status} />
    </>
  );
}
