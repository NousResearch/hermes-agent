export interface PackageNativeMigrationTarget {
  id: string;
  dashboard: string;
  recipe: string;
  currentSurface: string;
  targetSurface: string;
  packageNativeRoute: string | null;
  snapshotEndpoint: string | null;
  healthEndpoint: string | null;
  adapterPath: string;
  completion: number;
  status: "ready" | "in-progress" | "blocked" | "planned";
  nextStep: string;
  retirementAllowed: boolean;
  parity: Record<string, boolean>;
}

export const packageNativeMigrationTargets: PackageNativeMigrationTarget[] = [
  {
    "id": "media-engine.ops",
    "dashboard": "Media Engine Ops",
    "recipe": "pipeline-workflow-dashboard",
    "currentSurface": "static-adapter",
    "targetSurface": "package-native-react",
    "packageNativeRoute": "/package-native/media-engine",
    "snapshotEndpoint": "https://media.tlccapitalgroup.com/dashboard-snapshot",
    "healthEndpoint": "https://media.tlccapitalgroup.com/health",
    "adapterPath": "../media-engine/core/operations/hermes-dashboard-kit.css",
    "completion": 88,
    "status": "blocked",
    "nextStep": "Capture live production screenshot parity before adapter retirement.",
    "retirementAllowed": false,
    "parity": {
      "authPreserved": true,
      "commandsPreserved": true,
      "apiBehaviorPreserved": true,
      "snapshotEndpointExists": true,
      "packageNativeShadowRoute": true,
      "playwrightCoverage": true,
      "productionScreenshotEvidence": false,
      "rollbackPath": true
    }
  },
  {
    "id": "khashi-vc.roc",
    "dashboard": "Khashi VC ROC",
    "recipe": "operations-control-room + market-asset-explorer",
    "currentSurface": "static-adapter",
    "targetSurface": "package-native-react",
    "packageNativeRoute": "/package-native/khashi-vc",
    "snapshotEndpoint": "https://roc.tlccapitalgroup.com/api/dashboard-snapshot",
    "healthEndpoint": "https://roc.tlccapitalgroup.com/readyz",
    "adapterPath": "../khashi-vc/public/roc/hermes-dashboard-kit.css",
    "completion": 88,
    "status": "blocked",
    "nextStep": "Capture live production screenshot parity and validate command behavior before adapter retirement.",
    "retirementAllowed": false,
    "parity": {
      "authPreserved": true,
      "commandsPreserved": true,
      "apiBehaviorPreserved": true,
      "snapshotEndpointExists": true,
      "packageNativeShadowRoute": true,
      "playwrightCoverage": true,
      "productionScreenshotEvidence": false,
      "rollbackPath": true
    }
  },
  {
    "id": "hermes.executive-summary",
    "dashboard": "Hermes Executive Summary",
    "recipe": "executive-command-center",
    "currentSurface": "package-native-reference",
    "targetSurface": "live-enterprise-command-center",
    "packageNativeRoute": "/executive-summary",
    "snapshotEndpoint": "",
    "healthEndpoint": "https://agent.tlccapitalgroup.com/api/status",
    "adapterPath": "not-applicable: package-native reference route",
    "completion": 75,
    "status": "blocked",
    "nextStep": "Connect live project feeds and capture production screenshot evidence before treating this as the enterprise command source of truth.",
    "retirementAllowed": false,
    "parity": {
      "authPreserved": true,
      "commandsPreserved": true,
      "apiBehaviorPreserved": true,
      "snapshotEndpointExists": false,
      "packageNativeShadowRoute": true,
      "playwrightCoverage": true,
      "productionScreenshotEvidence": false,
      "rollbackPath": true
    }
  },
  {
    "id": "media-business-operations.main",
    "dashboard": "Media Business Operations",
    "recipe": "brand-business-performance",
    "currentSurface": "static-adapter",
    "targetSurface": "package-native-react",
    "packageNativeRoute": null,
    "snapshotEndpoint": "",
    "healthEndpoint": "https://media-business-operations.tlccapitalgroup.com/health",
    "adapterPath": "../media-business-operations/public/dashboard/hermes-dashboard-kit.css",
    "completion": 0,
    "status": "planned",
    "nextStep": "Define source analytics contract and project-owned dashboard snapshot endpoint.",
    "retirementAllowed": false,
    "parity": {
      "authPreserved": false,
      "commandsPreserved": false,
      "apiBehaviorPreserved": false,
      "snapshotEndpointExists": false,
      "playwrightCoverage": false,
      "productionScreenshotEvidence": false,
      "rollbackPath": false
    }
  }
];
