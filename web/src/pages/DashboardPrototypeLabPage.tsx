import { Boxes, GalleryVerticalEnd } from "lucide-react";
import {
  DashboardHeader,
  DashboardPrototypeReview,
  DashboardShell,
  DashboardSidebar,
  StatusPill,
  assessDashboardPrototypeSet,
} from "@hermes/dashboard-kit";
import { dashboardPrototypeSets } from "./dashboard-prototype-data";

export default function DashboardPrototypeLabPage() {
  const assessments = dashboardPrototypeSets.map(assessDashboardPrototypeSet);
  const missingRequired = assessments.flatMap((assessment) => assessment.missingRequirements);

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="Prototype Lab"
          description="Design spine V6 review surface."
          items={[
            { id: "overview", label: "Overview", href: "#overview", active: true, icon: GalleryVerticalEnd },
            ...dashboardPrototypeSets.map((set) => ({ id: set.id, label: set.dashboardName, href: `#${set.id}`, icon: Boxes })),
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="Dashboard Prototype Lab"
          eyebrow="Design system spine"
          description="Compare operator-first dashboard variants before production UI changes. This page is backed by the shared prototype registry."
          meta={(
            <>
              <StatusPill tone="info">{dashboardPrototypeSets.length} prototype sets</StatusPill>
              <StatusPill tone={missingRequired.length ? "warning" : "success"}>{missingRequired.length} missing requirements</StatusPill>
            </>
          )}
        />
      )}
    >
      <DashboardPrototypeReview prototypeSets={dashboardPrototypeSets} />
    </DashboardShell>
  );
}
