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
