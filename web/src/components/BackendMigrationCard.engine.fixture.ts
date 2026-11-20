// Engine and route output captured on this branch with the engine tests' fake host and the route test client
// (real engine, real routes; a scratch home per case): the plan route's blockers for the new refusals, the
// status route's records for a late turn (exit 4), a run killed while writing a file (exit 5), refused
// restores with missing backups, a relative launcher refused in the background, and an unrecognized record
// state. Pasted as returned.

export const unknownActivityBlocker = "Cannot tell whether turns are running on the old backend: these turn-marker files cannot be read, or hold a marker whose writer cannot be checked: /var/tmp/hermes-pytest-1000/r-hnrt3wof/pytest-of-mark/pytest-0/test_unreadable_marker_automat0/hermes-home/desktop/interrupted_turns.json. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted).";

export const refusedUrlPlans = {
  "held": {
    "applied": false,
    "dry_run": true,
    "exit_code": 3,
    "plan": {
      "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_refused_url0/hermes-home",
      "old_backend": null,
      "old_address": null,
      "new_address": null,
      "keep_webapp_address": false,
      "copy_env_keys": [],
      "generate_secret": false,
      "public_url": null,
      "auth_before": null,
      "gateway": {},
      "linger": null,
      "webapp_bundle_missing": null,
      "running_turns": null,
      "blockers": [
        "The old backend's public URL (dashboard.public_url) is not one the gateway accepts as webapp.public_url: it must be http:// or https:// with no credentials, query or fragment. Fix it, or set webapp.public_url yourself (its value is not printed here)."
      ],
      "unsupported": [
        {
          "supervisor": "systemd",
          "unit": "hermes-dashboard.service",
          "pid": 4242,
          "instructions": "The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root.",
          "steps": []
        }
      ],
      "notices": [],
      "nothing_to_do": false,
      "steps": [],
      "text": [
        "Backend migration plan (dry run — nothing changed)",
        "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_refused_url0/hermes-home",
        "  ✗ systemd hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
        "    The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root.",
        "    Steps, in this order (one command per line):",
        "",
        "  ✗ Blockers (nothing will be changed):",
        "    • The old backend's public URL (dashboard.public_url) is not one the gateway accepts as webapp.public_url: it must be http:// or https:// with no credentials, query or fragment. Fix it, or set webapp.public_url yourself (its value is not printed here)."
      ],
      "expect_token": null,
      "found": true
    },
    "report": [
      {
        "supervisor": "systemd",
        "running": true,
        "pid": 4242,
        "unit": "hermes-dashboard.service",
        "scope": "system",
        "state": "enabled",
        "this_home": true,
        "not_applied": "",
        "notes": []
      }
    ],
    "run": {
      "state": "none",
      "step": "",
      "steps": [],
      "message": "",
      "old_address": "",
      "new_address": "",
      "started_at": 0.0,
      "finished_at": null,
      "manual_steps": [],
      "reason": null
    },
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  },
  "forced": {
    "applied": false,
    "dry_run": true,
    "exit_code": 3,
    "plan": {
      "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_refused_url0/hermes-home",
      "old_backend": null,
      "old_address": null,
      "new_address": null,
      "keep_webapp_address": false,
      "copy_env_keys": [],
      "generate_secret": false,
      "public_url": null,
      "auth_before": null,
      "gateway": {},
      "linger": null,
      "webapp_bundle_missing": null,
      "running_turns": null,
      "blockers": [
        "The old backend's public URL (dashboard.public_url) is not one the gateway accepts as webapp.public_url: it must be http:// or https:// with no credentials, query or fragment. Fix it, or set webapp.public_url yourself (its value is not printed here)."
      ],
      "unsupported": [
        {
          "supervisor": "systemd",
          "unit": "hermes-dashboard.service",
          "pid": 4242,
          "instructions": "The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root.",
          "steps": []
        }
      ],
      "notices": [],
      "nothing_to_do": false,
      "steps": [],
      "text": [
        "Backend migration plan (dry run — nothing changed)",
        "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_refused_url0/hermes-home",
        "  ✗ systemd hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
        "    The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root.",
        "    Steps, in this order (one command per line):",
        "",
        "  ✗ Blockers (nothing will be changed):",
        "    • The old backend's public URL (dashboard.public_url) is not one the gateway accepts as webapp.public_url: it must be http:// or https:// with no credentials, query or fragment. Fix it, or set webapp.public_url yourself (its value is not printed here)."
      ],
      "expect_token": null,
      "found": true
    },
    "report": [
      {
        "supervisor": "systemd",
        "running": true,
        "pid": 4242,
        "unit": "hermes-dashboard.service",
        "scope": "system",
        "state": "enabled",
        "this_home": true,
        "not_applied": "",
        "notes": []
      }
    ],
    "run": {
      "state": "none",
      "step": "",
      "steps": [],
      "message": "",
      "old_address": "",
      "new_address": "",
      "started_at": 0.0,
      "finished_at": null,
      "manual_steps": [],
      "reason": null
    },
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  }
};

export const refusalBlockers = {
  "configBackend": "HERMES_CONFIG_BACKEND=remote keeps the settings outside config.yaml, but the migration backs up, rolls back and restores only config.yaml, so it needs the file config backend. Move this home by hand with the steps in the docs. Nothing changed.",
  "providers": "The old backend requires sign-in but its sign-in providers could not be read (127.0.0.1:9119/api/auth/providers did not answer), so the move could not check that the gateway offers the same ones. Check that the old backend answers there, then run this again. Nothing changed.",
  "managed": "The managed config /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_managed0/managed/config.yaml sets webapp.port, which the gateway applies after this home's config.yaml: after the move its Webapp would serve 0.0.0.0:8000 instead of serving 0.0.0.0:9119, so the old backend would be stopped for nothing. Have the administrator change that setting, then run this again. Nothing changed.",
  "guardPath": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_refusals_guard_path_0/systemd-user/hermes-dashboard.service.d/hermes-migrated.conf already exists; move it aside before migrating this backend."
};

export const relativeLauncherStatus = {
  "state": "failed",
  "step": "preflight",
  "steps": [
    {
      "name": "preflight",
      "state": "failed"
    }
  ],
  "message": "the background migration did not start (exit 3): the old backend was started by a relative path and its working directory could not be read; nothing changed",
  "old_address": "",
  "new_address": "",
  "started_at": 1795209898.7127728,
  "finished_at": 1795209898.7127728,
  "manual_steps": [],
  "refused": true,
  "reason": null,
  "log_tail": [],
  "launcher": null,
  "served_by": "dashboard",
  "now": 1795209898.741727,
  "profile": "default",
  "commands": {
    "status": "hermes -p default gateway migrate --backends --status",
    "restore": "hermes -p default gateway migrate --backends --restore"
  }
};

export const lateTurnStatus = {
  "running": {
    "state": "rolled_back",
    "step": "gateway_service",
    "steps": [
      {
        "name": "preflight",
        "state": "done"
      },
      {
        "name": "manifest",
        "state": "done"
      },
      {
        "name": "webapp_bundle",
        "state": "undone"
      },
      {
        "name": "write_config",
        "state": "undone"
      },
      {
        "name": "gateway_service",
        "state": "undone"
      },
      {
        "name": "stop_old_backend",
        "state": "pending"
      },
      {
        "name": "start_gateway",
        "state": "pending"
      },
      {
        "name": "verify",
        "state": "pending"
      }
    ],
    "message": "gateway_service failed: 1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted). The old backend has not been stopped.; rolled back, old backend verified on 0.0.0.0:9119",
    "old_address": "0.0.0.0:9119",
    "new_address": "0.0.0.0:9119",
    "started_at": 1795209897.5860546,
    "finished_at": 1795209897.6516888,
    "manual_steps": [],
    "reason": "gateway_service failed: 1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted). The old backend has not been stopped.",
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209897.6841922,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  },
  "unknown": {
    "state": "rolled_back",
    "step": "gateway_service",
    "steps": [
      {
        "name": "preflight",
        "state": "done"
      },
      {
        "name": "manifest",
        "state": "done"
      },
      {
        "name": "webapp_bundle",
        "state": "undone"
      },
      {
        "name": "write_config",
        "state": "undone"
      },
      {
        "name": "gateway_service",
        "state": "undone"
      },
      {
        "name": "stop_old_backend",
        "state": "pending"
      },
      {
        "name": "start_gateway",
        "state": "pending"
      },
      {
        "name": "verify",
        "state": "pending"
      }
    ],
    "message": "gateway_service failed: Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted). The old backend has not been stopped.; rolled back, old backend verified on 0.0.0.0:9119",
    "old_address": "0.0.0.0:9119",
    "new_address": "0.0.0.0:9119",
    "started_at": 1795209897.7477152,
    "finished_at": 1795209897.8543587,
    "manual_steps": [],
    "reason": "gateway_service failed: Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted). The old backend has not been stopped.",
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209897.8841934,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  }
};

export const killedWhileWritingStatus = {
  "config": {
    "state": "failed",
    "step": "write_config",
    "steps": [
      {
        "name": "preflight",
        "state": "done"
      },
      {
        "name": "manifest",
        "state": "done"
      },
      {
        "name": "webapp_bundle",
        "state": "done"
      },
      {
        "name": "write_config",
        "state": "running"
      },
      {
        "name": "gateway_service",
        "state": "pending"
      },
      {
        "name": "stop_old_backend",
        "state": "pending"
      },
      {
        "name": "start_gateway",
        "state": "pending"
      },
      {
        "name": "verify",
        "state": "pending"
      }
    ],
    "message": "interrupted at write_config; rollback incomplete: restored config.yaml: the migration stopped while writing /var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing_conf0/hermes-home/config.yaml, so it may hold the migration's changes; it was not written back",
    "old_address": "0.0.0.0:9119",
    "new_address": "0.0.0.0:9119",
    "started_at": 1795209912.180449,
    "finished_at": 1795209912.7922745,
    "manual_steps": [
      "hermes -p default gateway stop",
      "the migration stopped while writing /var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing_conf0/hermes-home/config.yaml, so it may hold the migration's changes: compare it with /var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing_conf0/hermes-home/backend_migration/config.yaml.bak and put back by hand what the migration changed, before you start the old backend",
      "/var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing_conf0/hermes-home/.env was restored from its backup (nothing to do for it)",
      "hermes -p default gateway uninstall",
      "systemctl --user enable hermes-dashboard.service",
      "systemctl --user start hermes-dashboard.service"
    ],
    "reason": "interrupted at write_config",
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209912.9019082,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  },
  "env": {
    "state": "failed",
    "step": "write_config",
    "steps": [
      {
        "name": "preflight",
        "state": "done"
      },
      {
        "name": "manifest",
        "state": "done"
      },
      {
        "name": "webapp_bundle",
        "state": "done"
      },
      {
        "name": "write_config",
        "state": "running"
      },
      {
        "name": "gateway_service",
        "state": "pending"
      },
      {
        "name": "stop_old_backend",
        "state": "pending"
      },
      {
        "name": "start_gateway",
        "state": "pending"
      },
      {
        "name": "verify",
        "state": "pending"
      }
    ],
    "message": "interrupted at write_config; rollback incomplete: restored .env: the migration stopped while writing /var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing__env0/hermes-home/.env, so it may hold the migration's changes; it was not written back",
    "old_address": "0.0.0.0:9119",
    "new_address": "0.0.0.0:9119",
    "started_at": 1795209913.6149676,
    "finished_at": 1795209914.2343907,
    "manual_steps": [
      "hermes -p default gateway stop",
      "/var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing__env0/hermes-home/config.yaml was restored from its backup (nothing to do for it)",
      "the migration stopped while writing /var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing__env0/hermes-home/.env, so it may hold the migration's changes: compare it with /var/tmp/hermes-pytest-1000/r-6k0fwp96/pytest-of-mark/pytest-0/test_killed_while_writing__env0/hermes-home/backend_migration/env.bak and put back by hand what the migration changed, before you start the old backend",
      "hermes -p default gateway uninstall",
      "systemctl --user enable hermes-dashboard.service",
      "systemctl --user start hermes-dashboard.service"
    ],
    "reason": "interrupted at write_config",
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209914.3737178,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  }
};

export const restoreRefusedStatus = {
  "envMissing": {
    "state": "failed",
    "step": "preflight",
    "steps": [
      {
        "name": "preflight",
        "state": "failed"
      }
    ],
    "message": "the background restore did not start (exit 3): the backup of .env (/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_env_missing_0/hermes-home/backend_migration/env.bak) cannot be read: No such file or directory; a restore would overwrite it, so nothing was written: your edits were kept and the gateway keeps serving; nothing changed",
    "old_address": "",
    "new_address": "",
    "started_at": 1795209898.007876,
    "finished_at": 1795209898.007876,
    "manual_steps": [
      "hermes -p default gateway stop",
      "cp /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_env_missing_0/hermes-home/backend_migration/config.yaml.bak /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_env_missing_0/hermes-home/config.yaml",
      "the backup /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_env_missing_0/hermes-home/backend_migration/env.bak of .env is missing or cannot be read: recover it, or put /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_env_missing_0/hermes-home/.env back as it was before the migration by hand, before you start the old backend",
      "hermes -p default gateway uninstall",
      "rm -f /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_env_missing_0/systemd-user/hermes-dashboard.service.d/hermes-migrated.conf && systemctl --user daemon-reload",
      "systemctl --user enable hermes-dashboard.service",
      "systemctl --user start hermes-dashboard.service"
    ],
    "refused": true,
    "reason": null,
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209898.0234776,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  },
  "configMissing": {
    "state": "failed",
    "step": "preflight",
    "steps": [
      {
        "name": "preflight",
        "state": "failed"
      }
    ],
    "message": "the background restore did not start (exit 3): the backup of config.yaml (/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/hermes-home/backend_migration/config.yaml.bak) cannot be read: No such file or directory; a restore would overwrite it, so nothing was written: your edits were kept and the gateway keeps serving; nothing changed",
    "old_address": "",
    "new_address": "",
    "started_at": 1795209898.158249,
    "finished_at": 1795209898.158249,
    "manual_steps": [
      "hermes -p default gateway stop",
      "the backup /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/hermes-home/backend_migration/config.yaml.bak of config.yaml is missing or cannot be read: recover it, or put /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/hermes-home/config.yaml back as it was before the migration by hand, before you start the old backend",
      "cp /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/hermes-home/backend_migration/env.bak /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/hermes-home/.env && chmod 600 /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/hermes-home/.env",
      "hermes -p default gateway uninstall",
      "rm -f /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_config_missing_0/systemd-user/hermes-dashboard.service.d/hermes-migrated.conf && systemctl --user daemon-reload",
      "systemctl --user enable hermes-dashboard.service",
      "systemctl --user start hermes-dashboard.service"
    ],
    "refused": true,
    "reason": null,
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209898.1767595,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  },
  "bothMissing": {
    "state": "failed",
    "step": "preflight",
    "steps": [
      {
        "name": "preflight",
        "state": "failed"
      }
    ],
    "message": "the background restore did not start (exit 3): the backup of config.yaml (/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_both_missing_0/hermes-home/backend_migration/config.yaml.bak) cannot be read: No such file or directory; a restore would overwrite it, so nothing was written: your edits were kept and the gateway keeps serving; nothing changed",
    "old_address": "",
    "new_address": "",
    "started_at": 1795209898.316791,
    "finished_at": 1795209898.316791,
    "manual_steps": [
      "hermes -p default gateway stop",
      "the backup /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_both_missing_0/hermes-home/backend_migration/config.yaml.bak of config.yaml is missing or cannot be read: recover it, or put /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_both_missing_0/hermes-home/config.yaml back as it was before the migration by hand, before you start the old backend",
      "the backup /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_both_missing_0/hermes-home/backend_migration/env.bak of .env is missing or cannot be read: recover it, or put /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_both_missing_0/hermes-home/.env back as it was before the migration by hand, before you start the old backend",
      "hermes -p default gateway uninstall",
      "rm -f /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_restore_both_missing_0/systemd-user/hermes-dashboard.service.d/hermes-migrated.conf && systemctl --user daemon-reload",
      "systemctl --user enable hermes-dashboard.service",
      "systemctl --user start hermes-dashboard.service"
    ],
    "refused": true,
    "reason": null,
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795209898.3347783,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  }
};

export const unknownStateStatus = {
  "state": "failed",
  "step": "",
  "steps": [],
  "message": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_unknown_state0/hermes-home/backend_migration.json has an unrecognized migration state; nothing can be undone automatically",
  "old_address": "",
  "new_address": "",
  "started_at": 0.0,
  "finished_at": null,
  "manual_steps": [],
  "reason": null,
  "log_tail": [],
  "launcher": null,
  "served_by": "dashboard",
  "now": 1795209898.4772065,
  "profile": "default",
  "commands": {
    "status": "hermes -p default gateway migrate --backends --status",
    "restore": "hermes -p default gateway migrate --backends --restore"
  }
};

// The status route's record after the old backend changed between the check and the stop (exit 4).
export const changedBackendStatus = {
  "state": "rolled_back",
  "step": "gateway_service",
  "steps": [
    {
      "name": "preflight",
      "state": "done"
    },
    {
      "name": "manifest",
      "state": "done"
    },
    {
      "name": "webapp_bundle",
      "state": "undone"
    },
    {
      "name": "write_config",
      "state": "undone"
    },
    {
      "name": "gateway_service",
      "state": "undone"
    },
    {
      "name": "stop_old_backend",
      "state": "pending"
    },
    {
      "name": "start_gateway",
      "state": "pending"
    },
    {
      "name": "verify",
      "state": "pending"
    }
  ],
  "message": "gateway_service failed: the old backend changed since the check (it was pid 4242 at the check and is pid 5242 now), so the stop would reach a process the checks never saw. The old backend has not been stopped; run this command again.; rolled back, old backend verified on 0.0.0.0:9119",
  "old_address": "0.0.0.0:9119",
  "new_address": "0.0.0.0:9119",
  "started_at": 1795212431.7260187,
  "finished_at": 1795212431.921434,
  "manual_steps": [],
  "reason": "gateway_service failed: the old backend changed since the check (it was pid 4242 at the check and is pid 5242 now), so the stop would reach a process the checks never saw. The old backend has not been stopped; run this command again.",
  "log_tail": [],
  "launcher": null,
  "served_by": "dashboard",
  "now": 1795212431.943505,
  "profile": "default",
  "commands": {
    "status": "hermes -p default gateway migrate --backends --status",
    "restore": "hermes -p default gateway migrate --backends --restore"
  }
};

// The apply route's answers with Remote Config selected: its own pre-check, and the launcher's refusal (the
// --detach parent exits 3 and writes no refusal record) when the pre-check let it through.
export const nonFileBackendApply = {
  "precheck": {
    "ok": false,
    "error": "blocked",
    "message": "The move cannot start yet.",
    "blockers": [
      "HERMES_CONFIG_BACKEND=remote keeps the settings outside config.yaml, but the migration backs up, rolls back and restores only config.yaml, so it needs the file config backend. Move this home by hand with the steps in the docs. Nothing changed."
    ]
  },
  "launcher": {
    "ok": false,
    "error": "not_started",
    "message": "HERMES_CONFIG_BACKEND=remote keeps the settings outside config.yaml, but the migration backs up, rolls back and restores only config.yaml, so it needs the file config backend. Move this home by hand with the steps in the docs. Nothing changed.",
    "status_command": "hermes -p default gateway migrate --backends --status"
  }
};

// A move whose Webapp build fails while the old backend goes offline: rollback incomplete (exit 5), as the
// status route returns it, for a systemd user unit and a hand-started backend.
export const exit5BundleStatus = {
  "systemd": {
    "state": "failed",
    "step": "webapp_bundle",
    "steps": [
      {
        "name": "preflight",
        "state": "done"
      },
      {
        "name": "manifest",
        "state": "done"
      },
      {
        "name": "webapp_bundle",
        "state": "failed"
      },
      {
        "name": "write_config",
        "state": "pending"
      },
      {
        "name": "gateway_service",
        "state": "pending"
      },
      {
        "name": "stop_old_backend",
        "state": "pending"
      },
      {
        "name": "start_gateway",
        "state": "pending"
      },
      {
        "name": "verify",
        "state": "pending"
      }
    ],
    "message": "webapp_bundle failed: the Webapp bundle is still missing after the build: dist/index.html is missing; rollback incomplete: old backend not verified: hermes-dashboard.service is not active",
    "old_address": "0.0.0.0:9119",
    "new_address": "0.0.0.0:9119",
    "started_at": 1795213477.5999467,
    "finished_at": 1795213537.6100283,
    "manual_steps": [
      "hermes -p default gateway stop",
      "cp /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back0/hermes-home/backend_migration/config.yaml.bak /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back0/hermes-home/config.yaml",
      "cp /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back0/hermes-home/backend_migration/env.bak /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back0/hermes-home/.env && chmod 600 /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back0/hermes-home/.env",
      "hermes -p default gateway uninstall",
      "systemctl --user enable hermes-dashboard.service",
      "systemctl --user start hermes-dashboard.service"
    ],
    "reason": "webapp_bundle failed: the Webapp bundle is still missing after the build: dist/index.html is missing",
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795213537.622485,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  },
  "manual": {
    "state": "failed",
    "step": "webapp_bundle",
    "steps": [
      {
        "name": "preflight",
        "state": "done"
      },
      {
        "name": "manifest",
        "state": "done"
      },
      {
        "name": "webapp_bundle",
        "state": "failed"
      },
      {
        "name": "write_config",
        "state": "pending"
      },
      {
        "name": "gateway_service",
        "state": "pending"
      },
      {
        "name": "stop_old_backend",
        "state": "pending"
      },
      {
        "name": "start_gateway",
        "state": "pending"
      },
      {
        "name": "verify",
        "state": "pending"
      }
    ],
    "message": "webapp_bundle failed: the Webapp bundle is still missing after the build: dist/index.html is missing; rollback incomplete: old backend not verified: hermes-dashboard.service is not active",
    "old_address": "0.0.0.0:9119",
    "new_address": "0.0.0.0:9119",
    "started_at": 1795213537.6783073,
    "finished_at": 1795213597.6890085,
    "manual_steps": [
      "hermes -p default gateway stop",
      "cp /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back1/hermes-home/backend_migration/config.yaml.bak /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back1/hermes-home/config.yaml",
      "cp /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back1/hermes-home/backend_migration/env.bak /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back1/hermes-home/.env && chmod 600 /var/tmp/hermes-pytest-1000/r-ua90v9hb/pytest-of-mark/pytest-0/test_bundle_fails_and_old_back1/hermes-home/.env",
      "hermes -p default gateway uninstall",
      "start the old backend again the way you started it before (it was pid 4242), so it serves 0.0.0.0:9119 again"
    ],
    "reason": "webapp_bundle failed: the Webapp bundle is still missing after the build: dist/index.html is missing",
    "log_tail": [],
    "launcher": null,
    "served_by": "dashboard",
    "now": 1795213597.6973193,
    "profile": "default",
    "commands": {
      "status": "hermes -p default gateway migrate --backends --status",
      "restore": "hermes -p default gateway migrate --backends --restore"
    }
  }
};

// The plan route's blockers for a Hermes home whose path holds the engine's remedy text: a symbolic-link
// config.yaml, and an unreadable turn-marker file under that home.
export const flagInPathBlockers = {
  "symlink": "/var/tmp/hermes-pytest-1000/r-3n0nc6rb/pytest-of-mark/pytest-0/test_flag_text_inside_a_path0/--force-interrupt/, then re-run with --force-interrupt (any running turn is interrupted)./h/config.yaml is a symbolic link: the migration would change the file it points to, which may be shared. Replace the link with a copy of the file (or move this home off the shared file), then run this again.",
  "unknown": "Cannot tell whether turns are running on the old backend: these turn-marker files cannot be read, or hold a marker whose writer cannot be checked: /var/tmp/hermes-pytest-1000/r-3n0nc6rb/pytest-of-mark/pytest-0/test_flag_text_inside_a_path0/--force-interrupt/, then re-run with --force-interrupt (any running turn is interrupted)./h/desktop/interrupted_turns.json. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted)."
};

// Every distinct step the engine's by-hand planners (POSIX: systemd user and system, launchd, Nix, s6,
// container, hand-started, no backend; Windows: Scheduled Task, Startup folder, hand-started) and its
// rollback steps (every old-backend kind and file state) emit, from the real functions (probe_steps_all.py).
export const engineStepStrings = [
  "# (an 'already exists' error from mkdir is fine)",
  "# and Disable that task in Task Scheduler",
  "# and remove whatever starts it at sign-in",
  "# copy K1 from the old backend's environment into C:\\Users\\u\\.hermes/.env (values are never printed here)",
  "# copy K1, K2 from the old launcher's environment into /h/.env, then: chmod 600 /h/.env  (values are never printed here)",
  "# copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into /h/.env (chmod 600)",
  "# copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h/.env (chmod 600)",
  "# copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into the Hermes .env (chmod 600)",
  "# copy the old backend's public URL (dashboard.public_url, or HERMES_DASHBOARD_PUBLIC_URL in its environment) into webapp.public_url: open config.yaml with $env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes config edit",
  "# dashboard.public_url is not a URL the gateway accepts (http:// or https://, no credentials, query or fragment): set webapp.public_url yourself (the value is not printed here)",
  "# enable the gateway service there, then run home-manager switch",
  "# enable the gateway service there, then run nixos-rebuild switch or home-manager switch",
  "# give the gateway K1, K2 through the environment file your Nix configuration provides (values are never printed here)",
  "# give the gateway every HERMES_DASHBOARD_* variable the backend had through the environment file your Nix configuration provides (values are never printed here)",
  "# in Task Scheduler, End the task that starts it",
  "# in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host and webapp.port to the old backend's address",
  "# in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119",
  "# in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119, webapp.public_url: https://dash.example",
  "# in your Nix configuration: disable the hermes backend service and set webapp.enabled: true, webapp.host and webapp.port to the old backend's address in the Hermes settings",
  "# in your Nix configuration: disable the hermes backend service and set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119 in the Hermes settings",
  "# in your Nix configuration: disable the hermes backend service and set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119, webapp.public_url: https://dash.example in the Hermes settings",
  "# more than one Startup-folder entry starts this exact command: find the one that starts this backend (shell:startup) and move it aside",
  "# move it to a folder outside shell:startup, and move it back to undo",
  "# publish 9119 on the gateway's container and restart it, so its gateway serves the address",
  "# publish the old backend's port on the gateway's container and restart it, so its gateway serves the address",
  "# remove the dashboard: recreate the container without HERMES_DASHBOARD=1, or drop the Compose `dashboard` service and run: docker compose up -d --remove-orphans",
  "# run the next line from your own supervisor (restart on exit and at boot):",
  "# set the old backend's address: $env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes config set webapp.host HOST, then webapp.port PORT",
  "# set the old backend's address: HERMES_HOME=/h hermes config set webapp.host HOST, then webapp.port PORT",
  "# stop the old backend (Ctrl+C in its window)",
  "# stop the old backend through its supervisor and keep it from starting at boot",
  "# the gateway takes a few seconds to serve the address: if the next line prints 'The gateway is not running', wait 5 seconds and run it again",
  "# then by what it prints: 'Gateway is not running' (no service installed) -> $env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes gateway install; 'No gateway process detected' -> $env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes gateway start; 'Gateway process running' or 'Running manually' -> $env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes gateway restart",
  "# then by what it prints: 'Gateway is not running' (no service installed) -> $env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes gateway install; 'No gateway process detected' -> $env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes gateway start; 'Gateway process running' or 'Running manually' -> $env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes gateway restart",
  "# then by what it prints: 'Gateway is not running' (no service installed) -> HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes gateway restart",
  "# then by what it prints: 'Gateway is not running' (no service installed) -> HERMES_HOME=/h hermes gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> HERMES_HOME=/h hermes gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> HERMES_HOME=/h hermes gateway restart",
  "# then by what it prints: 'Gateway is not running' (no service installed) -> hermes -p default gateway install; 'No gateway process detected' -> hermes -p default gateway start; 'Gateway process running' or 'Running manually' -> hermes -p default gateway restart",
  "# then by what it prints: 'Gateway is not running' (no service installed) -> hermes -p default gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> hermes -p default gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> hermes -p default gateway restart",
  "# these lines set HERMES_HOME for this Hermes root: run them in PowerShell, not Command Prompt",
  "# this Startup-folder entry starts the same command; if it is what starts this backend at sign-in, move it out of the Startup folder (Windows tries to open any file left there, even a renamed one):",
  "# to undo later: move \"C:\\Users\\u\\.hermes\\backend_migration\\hermes.lnk\" \"C:\\Users\\u\\Startup\\hermes.lnk\"",
  "$env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes config set webapp.enabled true",
  "$env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes config set webapp.host 0.0.0.0",
  "$env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes config set webapp.port 9119",
  "$env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes gateway status",
  "$env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes webapp --build-only",
  "$env:HERMES_HOME='C:\\Users\\u\\.hermes'; hermes webapp --status",
  "$env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes config set webapp.enabled true",
  "$env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes config set webapp.host 0.0.0.0",
  "$env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes config set webapp.port 9119",
  "$env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes gateway status",
  "$env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes webapp --build-only",
  "$env:HERMES_HOME='\\var\\tmp\\hermes-pytest-1000\\r-7q_qoome\\pytest-of-mark\\pytest-0\\test_collect0\\w h'; hermes webapp --status",
  "/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env was restored from its backup (nothing to do for it)",
  "/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml was restored from its backup (nothing to do for it)",
  "HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes config set webapp.enabled true",
  "HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes config set webapp.host 0.0.0.0",
  "HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes config set webapp.port 9119",
  "HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes gateway status",
  "HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes webapp --build-only",
  "HERMES_HOME='/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/x h' hermes webapp --status",
  "HERMES_HOME=/h hermes config set webapp.enabled true",
  "HERMES_HOME=/h hermes config set webapp.host 0.0.0.0",
  "HERMES_HOME=/h hermes config set webapp.port 9119",
  "HERMES_HOME=/h hermes config set webapp.public_url https://dash.example",
  "HERMES_HOME=/h hermes gateway run",
  "HERMES_HOME=/h hermes gateway status",
  "HERMES_HOME=/h hermes webapp --build-only",
  "HERMES_HOME=/h hermes webapp --status",
  "HERMES_HOME=/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home hermes gateway start",
  "HERMES_HOME=/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home hermes gateway stop",
  "HERMES_HOME=/var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home hermes gateway uninstall",
  "cp /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/backend_migration/config.yaml.bak /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml",
  "cp /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/backend_migration/env.bak /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env && chmod 600 /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env",
  "hermes -p default config set webapp.enabled true",
  "hermes -p default config set webapp.host 0.0.0.0",
  "hermes -p default config set webapp.port 9119",
  "hermes -p default gateway status",
  "hermes -p default webapp --build-only",
  "hermes -p default webapp --status",
  "keep /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env as it is (it was edited after the migration wrote it): merge the lines you need from /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/backend_migration/env.bak into it by hand before you start the old backend",
  "keep /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env as it is (it was edited after the migration wrote it): merge the lines you need from nothing (it did not exist before the migration) into it by hand before you start the old backend",
  "keep /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml as it is (it was edited after the migration wrote it): merge the lines you need from /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/backend_migration/config.yaml.bak into it by hand before you start the old backend",
  "keep /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml as it is (it was edited after the migration wrote it): merge the lines you need from nothing (it did not exist before the migration) into it by hand before you start the old backend",
  "launchctl bootout gui/501/ai.hermes.dashboard; launchctl disable gui/501/ai.hermes.dashboard",
  "mkdir \"C:\\Users\\u\\.hermes\\backend_migration\"",
  "move \"C:\\Users\\u\\Startup\\hermes.lnk\" \"C:\\Users\\u\\.hermes\\backend_migration\\hermes.lnk\"",
  "mv /Users/u/Library/LaunchAgents/ai.hermes.dashboard.plist.disabled /Users/u/Library/LaunchAgents/ai.hermes.dashboard.plist; launchctl enable gui/501/ai.hermes.dashboard; launchctl bootstrap gui/501 /Users/u/Library/LaunchAgents/ai.hermes.dashboard.plist",
  "rm -f /u/.config/systemd/user/hermes-dashboard.service.d/hermes-migrated.conf && systemctl --user daemon-reload",
  "rm /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env",
  "rm /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml",
  "schtasks /Change /TN \"Hermes Dashboard\" /DISABLE",
  "schtasks /End /TN \"Hermes Dashboard\"",
  "start the old backend again the way you started it before (it was pid 4242), so it serves 0.0.0.0:9119 again",
  "start the old backend again the way you started it before, so it serves 0.0.0.0:9119 again",
  "sudo systemctl disable --now hermes-dashboard.service",
  "systemctl --user disable --now hermes-dashboard.service",
  "systemctl --user enable hermes-dashboard.service",
  "systemctl --user start hermes-dashboard.service",
  "systemctl enable --runtime hermes-dashboard.service",
  "systemctl is-active hermes-dashboard.service  # must print inactive (or failed) before the next line",
  "taskkill /PID 4242 /T /F",
  "the backup /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/gone-env.bak of .env is missing or cannot be read: recover it, or put /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/.env back as it was before the migration by hand, before you start the old backend",
  "the backup /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/gone.bak of config.yaml is missing or cannot be read: recover it, or put /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml back as it was before the migration by hand, before you start the old backend",
  "the migration stopped while writing /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/config.yaml, so it may hold the migration's changes: compare it with /var/tmp/hermes-pytest-1000/r-7q_qoome/pytest-of-mark/pytest-0/test_collect0/home/backend_migration/config.yaml.bak and put back by hand what the migration changed, before you start the old backend"
];
