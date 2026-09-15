from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BASE_SHA = "057dcdf236f8a6a26721c10fcc6ccb72726e272a"


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def insert_after(text: str, anchor: str, insertion: str, *, marker: str, label: str) -> str:
    if marker in text:
        return text
    count = text.count(anchor)
    if count != 1:
        raise PatchError(f"{label}: expected one anchor, found {count}")
    return text.replace(anchor, anchor + insertion, 1)


def patch_file(path: Path, transform) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = transform(original)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def patch_main(text: str) -> str:
    return insert_after(
        text,
        "} from 'electron'\n\n",
        "import './workstation-browser-runtime'\n\n",
        marker="import './workstation-browser-runtime'",
        label="desktop main runtime import",
    )


PRELOAD_BRIDGE = r"""  workstationBrowser: {
    status: () => ipcRenderer.invoke('hermes:workstation-browser:status'),
    ensure: () => ipcRenderer.invoke('hermes:workstation-browser:ensure'),
    newTab: target => ipcRenderer.invoke('hermes:workstation-browser:new-tab', target),
    activateTab: tabId => ipcRenderer.invoke('hermes:workstation-browser:activate-tab', tabId),
    closeTab: tabId => ipcRenderer.invoke('hermes:workstation-browser:close-tab', tabId),
    navigate: target => ipcRenderer.invoke('hermes:workstation-browser:navigate', target),
    back: () => ipcRenderer.invoke('hermes:workstation-browser:back'),
    forward: () => ipcRenderer.invoke('hermes:workstation-browser:forward'),
    reload: () => ipcRenderer.invoke('hermes:workstation-browser:reload'),
    stop: () => ipcRenderer.invoke('hermes:workstation-browser:stop'),
    focus: () => ipcRenderer.invoke('hermes:workstation-browser:focus'),
    attach: bounds => ipcRenderer.invoke('hermes:workstation-browser:attach', bounds),
    setBounds: bounds => ipcRenderer.invoke('hermes:workstation-browser:set-bounds', bounds),
    detach: () => ipcRenderer.invoke('hermes:workstation-browser:detach'),
    pause: () => ipcRenderer.invoke('hermes:workstation-browser:pause'),
    resume: () => ipcRenderer.invoke('hermes:workstation-browser:resume'),
    takeControl: () => ipcRenderer.invoke('hermes:workstation-browser:take-control'),
    releaseControl: () => ipcRenderer.invoke('hermes:workstation-browser:release-control'),
    cleanupCache: force => ipcRenderer.invoke('hermes:workstation-browser:cleanup-cache', force),
    onState: callback => {
      const listener = (_event, state) => callback(state)
      ipcRenderer.on('hermes:workstation-browser:state', listener)

      return () => ipcRenderer.removeListener('hermes:workstation-browser:state', listener)
    }
  },
"""


def patch_preload(text: str) -> str:
    return insert_after(
        text,
        "contextBridge.exposeInMainWorld('hermesDesktop', {\n",
        PRELOAD_BRIDGE,
        marker="workstationBrowser: {",
        label="desktop preload bridge",
    )


def patch_global_types(text: str) -> str:
    text = insert_after(
        text,
        "import type { TranslucencyState } from '@hermes/shared/translucency'\n",
        "import type { WorkstationBrowserBridge } from './app/browser/types'\n",
        marker="WorkstationBrowserBridge",
        label="desktop global browser type import",
    )
    return insert_after(
        text,
        "    hermesDesktop: {\n",
        "      workstationBrowser: WorkstationBrowserBridge\n",
        marker="workstationBrowser: WorkstationBrowserBridge",
        label="desktop global browser bridge",
    )


def patch_routes(text: str) -> str:
    text = insert_after(
        text,
        "export const ARTIFACTS_ROUTE = '/artifacts'\n",
        "export const BROWSER_ROUTE = '/browser'\n",
        marker="BROWSER_ROUTE",
        label="browser route constant",
    )
    text = replace_once(
        text,
        "  | 'artifacts'\n  | 'chat'",
        "  | 'artifacts'\n  | 'browser'\n  | 'chat'",
        label="browser AppView",
    )
    text = replace_once(
        text,
        "  | 'artifacts'\n  | 'command-center'",
        "  | 'artifacts'\n  | 'browser'\n  | 'command-center'",
        label="browser AppRouteId",
    )
    text = insert_after(
        text,
        "  { id: 'artifacts', path: ARTIFACTS_ROUTE, view: 'artifacts' },\n",
        "  { id: 'browser', path: BROWSER_ROUTE, view: 'browser' },\n",
        marker="{ id: 'browser', path: BROWSER_ROUTE",
        label="browser APP_ROUTES row",
    )
    return text


BROWSER_NAV = r"""  {
    id: 'browser',
    label: 'Browser',
    icon: props => <Codicon name="globe" {...props} />,
    route: BROWSER_ROUTE
  },
"""


def patch_sidebar(text: str) -> str:
    text = replace_once(
        text,
        "  ARTIFACTS_ROUTE,\n  CRON_ROUTE,",
        "  ARTIFACTS_ROUTE,\n  BROWSER_ROUTE,\n  CRON_ROUTE,",
        label="sidebar Browser route import",
    )
    text = insert_after(
        text,
        "  {\n    id: 'new-session',\n    label: '',\n    icon: props => <Codicon name=\"robot\" {...props} />,\n    action: 'new-session',\n    keybindActionId: 'session.new'\n  },\n",
        BROWSER_NAV,
        marker="id: 'browser'",
        label="sidebar Browser nav",
    )
    text = replace_once(
        text,
        "{isNewSession || item.route ? (",
        "{isNewSession || (item.route && item.id !== 'browser') ? (",
        label="disable Browser split route",
    )
    return text


def patch_surfaces(text: str) -> str:
    text = replace_once(
        text,
        "import { contributedRoutes, NEW_CHAT_ROUTE, ROUTES_AREA, sessionRoute } from '../routes'",
        "import { BROWSER_ROUTE, contributedRoutes, NEW_CHAT_ROUTE, ROUTES_AREA, sessionRoute } from '../routes'",
        label="surface Browser route import",
    )
    text = insert_after(
        text,
        "const ArtifactsView = lazy(async () => ({ default: (await import('../artifacts')).ArtifactsView }))\n",
        "const BrowserView = lazy(async () => ({ default: (await import('../browser')).BrowserView }))\n",
        marker="const BrowserView = lazy",
        label="Browser lazy view",
    )
    text = insert_after(
        text,
        '      <Route element={page(<ArtifactsView setStatusbarItemGroup={setStatusbarItemGroup} />)} path="artifacts" />\n',
        '      <Route element={page(<BrowserView />)} path={BROWSER_ROUTE.slice(1)} />\n',
        marker="<BrowserView />",
        label="Browser workspace route",
    )
    return text


def patch_kanban(text: str) -> str:
    text = text.replace(
        " * Ships OFF by default (`defaultEnabled: false`):",
        " * Hermes Workstation ships this bundled plugin ON by default (`defaultEnabled: true`):",
        1,
    )
    return replace_once(
        text,
        "  defaultEnabled: false,\n",
        "  defaultEnabled: true,\n",
        label="Workstation Kanban default",
    )


def patch_browser_tool(text: str) -> str:
    text = insert_after(
        text,
        "from tools.browser_extension_router import (\n    extension_controller_available,\n    routed_browser_handler,\n)\n",
        "from tools.browser_workstation import (\n    workstation_controller_available,\n    workstation_routed_browser_handler,\n)\n",
        marker="workstation_routed_browser_handler",
        label="Workstation browser router import",
    )
    text = insert_after(
        text,
        "def _browser_router_kw(kw: dict) -> dict:\n    \"\"\"Identity kwargs forwarded to the extension router wrapper.\"\"\"\n    return {\n        \"task_id\": kw.get(\"task_id\"),\n        \"session_id\": kw.get(\"session_id\"),\n    }\n",
        "\n\ndef _workstation_or_legacy(action: str, args: dict, kw: dict, fallback):\n    \"\"\"Prefer the embedded Hermes Browser; preserve legacy fallback for unbound tasks.\"\"\"\n    return workstation_routed_browser_handler(\n        action,\n        args,\n        fallback=fallback,\n        task_id=kw.get(\"task_id\"),\n        session_id=kw.get(\"session_id\"),\n    )\n",
        marker="def _workstation_or_legacy",
        label="Workstation browser fallback helper",
    )
    text = replace_once(
        text,
        "    return check_browser_requirements() or extension_controller_available(action)\n",
        "    return (\n        check_browser_requirements()\n        or extension_controller_available(action)\n        or workstation_controller_available()\n    )\n",
        label="Workstation browser availability",
    )
    # Workstation must be the OUTERMOST router. The official extension router
    # and legacy backend are fallbacks only while an unbound task is allowed to
    # route elsewhere. This also makes routing_enabled=false truly internal-only.
    route_specs = {
        "browser_navigate": 'browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id"))',
        "browser_snapshot": 'browser_snapshot(\n            full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task"))',
        "browser_click": 'browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id"))',
        "browser_type": 'browser_type(ref=args.get("ref", ""), text=args.get("text", ""), clear=args.get("clear", True), append=args.get("append", False), task_id=kw.get("task_id"))',
        "browser_scroll": 'browser_scroll(direction=args.get("direction", "down"), task_id=kw.get("task_id"))',
        "browser_back": 'browser_back(task_id=kw.get("task_id"))',
        "browser_press": 'browser_press(key=args.get("key", ""), task_id=kw.get("task_id"))',
        "browser_get_images": 'browser_get_images(task_id=kw.get("task_id"))',
        "browser_vision": 'browser_vision(question=args.get("question", ""), annotate=args.get("annotate", False), task_id=kw.get("task_id"))',
        "browser_console": 'browser_console(clear=args.get("clear", False), expression=args.get("expression"), task_id=kw.get("task_id"))',
    }
    for action, legacy_call in route_specs.items():
        old = (
            f'handler=lambda args, **kw: routed_browser_handler(\n'
            f'        "{action}",\n'
            f'        args,\n'
            f'        fallback=lambda: {legacy_call},\n'
            f'        **_browser_router_kw(kw),\n'
            f'    ),'
        )
        new = (
            f'handler=lambda args, **kw: _workstation_or_legacy(\n'
            f'        "{action}",\n'
            f'        args,\n'
            f'        kw,\n'
            f'        lambda: routed_browser_handler(\n'
            f'            "{action}",\n'
            f'            args,\n'
            f'            fallback=lambda: {legacy_call},\n'
            f'            **_browser_router_kw(kw),\n'
            f'        ),\n'
            f'    ),'
        )
        if new not in text:
            if old not in text:
                raise PatchError(f"browser tool route anchor missing for {action}")
            text = text.replace(old, new, 1)
    text = text.replace("    check_fn=check_browser_requirements,\n    emoji=\"🖼️\",", "    check_fn=lambda: check_browser_routed_requirements(\"browser_get_images\"),\n    emoji=\"🖼️\",")
    text = text.replace("    check_fn=check_browser_vision_requirements,\n    emoji=\"👁️\",", "    check_fn=lambda: check_browser_vision_requirements() or workstation_controller_available(),\n    emoji=\"👁️\",")
    text = text.replace("    check_fn=check_browser_requirements,\n    emoji=\"🖥️\",", "    check_fn=lambda: check_browser_routed_requirements(\"browser_console\"),\n    emoji=\"🖥️\",")
    return text


def patch_cli_config(text: str) -> str:
    return insert_after(
        text,
        "  inactivity_timeout: 120\n",
        "  # Hermes Workstation embedded Chromium. When routing_enabled is false,\n  # browser_* tools use only the internal persistent browser and fail closed\n  # if the Desktop controller is unavailable.\n  workstation:\n    enabled: true\n    routing_enabled: true\n    fail_closed_after_binding: true\n",
        marker="fail_closed_after_binding",
        label="Workstation browser config",
    )


TARGETS = {
    "apps/desktop/electron/main.ts": patch_main,
    "apps/desktop/electron/preload.ts": patch_preload,
    "apps/desktop/src/global.d.ts": patch_global_types,
    "apps/desktop/src/app/routes.ts": patch_routes,
    "apps/desktop/src/app/chat/sidebar/index.tsx": patch_sidebar,
    "apps/desktop/src/app/contrib/surfaces.tsx": patch_surfaces,
    "apps/desktop/src/plugins/kanban/plugin.tsx": patch_kanban,
    "tools/browser_tool.py": patch_browser_tool,
    "cli-config.yaml.example": patch_cli_config,
}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true", help="Validate anchors without writing.")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    if not (root / ".git").exists():
        print("INFO: .git directory not found; applying against source tree without Git base check.")
    else:
        try:
            head = git_output(root, "rev-parse", "HEAD")
            dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
            if dirty and args.require_clean:
                raise PatchError("Tracked files are already modified; commit/stash them before applying.")
            if dirty and not args.require_clean:
                print("WARNING: tracked edits exist; patching only validated Workstation anchors.")
            try:
                subprocess.check_call(
                    ["git", "merge-base", "--is-ancestor", BASE_SHA, head],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as exc:
                raise PatchError(
                    f"Current HEAD {head} is not based on expected Hermes base {BASE_SHA}. "
                    "Sync/review before applying this initial patch."
                ) from exc
        except FileNotFoundError:
            print("WARNING: git is unavailable; skipping base verification.")

    changed: list[str] = []
    for rel, transform in TARGETS.items():
        path = root / rel
        if not path.exists():
            raise PatchError(f"Required Hermes file not found: {rel}")
        original = path.read_text(encoding="utf-8")
        updated = transform(original)
        if updated != original:
            if not args.check:
                path.write_text(updated, encoding="utf-8", newline="\n")
            changed.append(rel)

    if args.check:
        print("OK: all Hermes core integration anchors are valid.")
    elif changed:
        print("Applied Hermes Workstation core integration:")
        for rel in changed:
            print(f"  - {rel}")
    else:
        print("Hermes Workstation core integration is already applied.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        raise SystemExit(f"ERROR: {exc}")
