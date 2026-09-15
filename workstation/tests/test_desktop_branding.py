import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hermes_work_branding_preserves_update_and_profile_identity():
    package = json.loads((ROOT / "apps" / "desktop" / "package.json").read_text(encoding="utf-8"))
    build = package["build"]

    assert package["productName"] == "Hermes Work"
    assert build["productName"] == "Hermes Work"
    assert build["nsis"]["shortcutName"] == "Hermes Work"
    assert build["nsis"]["uninstallDisplayName"] == "Hermes Work"

    # These are persisted/update identities, not display branding.
    assert build["appId"] == "com.nousresearch.hermes"
    assert build["executableName"] == "Hermes"
    assert build["artifactName"] == "Hermes-${version}-${os}-${arch}.${ext}"
    assert build["protocols"][0]["schemes"] == ["hermes"]

    main_source = (ROOT / "apps" / "desktop" / "electron" / "main.ts").read_text(encoding="utf-8")
    assert "app.setPath('userData', path.join(app.getPath('appData'), 'Hermes'))" in main_source
    assert "app.setAppUserModelId('com.nousresearch.hermes')" in main_source
    assert "process.env.HERMES_DESKTOP_APP_NAME || 'Hermes Work'" in main_source
