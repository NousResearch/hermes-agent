---
title: Android Release Pipeline
---

# Android Release Pipeline

Hermes Android release assets are produced by GitHub Actions, not by `scripts/release.py` directly.

Release flow:
1. Run `python scripts/release.py --bump <patch|minor|major> --publish`
2. The script creates the Git tag and GitHub release
3. `.github/workflows/android-release.yml` triggers on release publication
4. CI builds signed Android artifacts:
   - release APK
   - release AAB
5. `scripts/android_release_manifest.py` renames artifacts and emits SHA256 files
6. GitHub Actions uploads the APK, AAB, and checksum files to the release

Required GitHub secrets:
- `ANDROID_KEYSTORE_BASE64`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

Local files which must stay untracked:
- `android/keystore.properties`
- `android/release.keystore`
- `android/local.properties`

The Android build uses the version metadata already managed by Hermes release tooling and derives Android `versionCode` from the CalVer tag format.
