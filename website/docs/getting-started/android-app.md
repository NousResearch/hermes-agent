---
title: Android App
---

# Hermes Android App

Hermes now has an Android APK workstream separate from the Termux path.

Use the Android app when you want:
- a native Android UI
- an embedded Python runtime
- the local Hermes API server inside the app process
- GitHub Actions to build APK/AAB artifacts

Use the Termux path when you want:
- the terminal-first Hermes CLI on Android
- direct shell workflows in Termux
- the existing `.[termux]` install path

Current MVP boundaries:
- native Android shell under `android/`
- embedded Python runtime and local API server boot
- mobile-safe default tool profile
- CI-built debug APKs and release APK/AAB artifacts

The Android app does not replace the Termux workflow. They are separate product surfaces with different constraints.
