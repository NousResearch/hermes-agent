// Plan payloads captured from the real plan route (GET /api/gateway/migrate/backends/plan, then with
// ?force_interrupt=true): the route test client over the engine tests' fake host, one turn running
// (`running`) or turn activity unreadable (`unknown`). The container backend is seen from the host (another
// network namespace), the systemd system unit shares the planner's. Pasted as the route returned them.

export const containerPlans = {
  "running": {
    "held": {
      "applied": false,
      "dry_run": true,
      "exit_code": 3,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_1_0/hermes-home",
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
        "running_turns": 1,
        "blockers": [
          "1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted)."
        ],
        "unsupported": [
          {
            "supervisor": "container",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
            "steps": []
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_1_0/hermes-home",
          "  ✗ container hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
          "    Steps, in this order (one command per line):",
          "",
          "  ✗ Blockers (nothing will be changed):",
          "    • 1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted)."
        ],
        "expect_token": null,
        "found": true
      },
      "report": [
        {
          "supervisor": "container",
          "running": true,
          "pid": 4242,
          "unit": "hermes-dashboard.service",
          "scope": "user",
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
      "exit_code": 6,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_1_0/hermes-home",
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
        "running_turns": 1,
        "blockers": [],
        "unsupported": [
          {
            "supervisor": "container",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
            "steps": [
              "# in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119",
              "# copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_1_0/hermes-home/.env (chmod 600)",
              "# remove the dashboard: recreate the container without HERMES_DASHBOARD=1, or drop the Compose `dashboard` service and run: docker compose up -d --remove-orphans",
              "# publish 9119 on the gateway's container and restart it, so its gateway serves the address"
            ]
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_1_0/hermes-home",
          "  ✗ container hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
          "    Steps, in this order (one command per line):",
          "      # in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119",
          "      # copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_1_0/hermes-home/.env (chmod 600)",
          "      # remove the dashboard: recreate the container without HERMES_DASHBOARD=1, or drop the Compose `dashboard` service and run: docker compose up -d --remove-orphans",
          "      # publish 9119 on the gateway's container and restart it, so its gateway serves the address"
        ],
        "expect_token": null,
        "found": true
      },
      "report": [
        {
          "supervisor": "container",
          "running": true,
          "pid": 4242,
          "unit": "hermes-dashboard.service",
          "scope": "user",
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
  },
  "unknown": {
    "held": {
      "applied": false,
      "dry_run": true,
      "exit_code": 3,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_None_0/hermes-home",
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
          "Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted)."
        ],
        "unsupported": [
          {
            "supervisor": "container",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it.",
            "steps": []
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_None_0/hermes-home",
          "  ✗ container hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it.",
          "    Steps, in this order (one command per line):",
          "",
          "  ✗ Blockers (nothing will be changed):",
          "    • Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted)."
        ],
        "expect_token": null,
        "found": true
      },
      "report": [
        {
          "supervisor": "container",
          "running": true,
          "pid": 4242,
          "unit": "hermes-dashboard.service",
          "scope": "user",
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
      "exit_code": 6,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_None_0/hermes-home",
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
        "blockers": [],
        "unsupported": [
          {
            "supervisor": "container",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it. Its turn activity could not be read, so turns may be running on it now: stopping it interrupts them.",
            "steps": [
              "# in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119",
              "# copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_None_0/hermes-home/.env (chmod 600)",
              "# remove the dashboard: recreate the container without HERMES_DASHBOARD=1, or drop the Compose `dashboard` service and run: docker compose up -d --remove-orphans",
              "# publish 9119 on the gateway's container and restart it, so its gateway serves the address"
            ]
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_None_0/hermes-home",
          "  ✗ container hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend runs in a container (Compose sidecar or docker run). Set webapp.enabled/host/port (below) in the mounted Hermes home, drop the dashboard service from the Compose file, publish the same port on the gateway container, then `docker compose up -d`. Its Hermes version could not be checked from outside its container, and a Hermes older than v2026.7.30 does not record running chats: check inside the container that no reply is being written before you stop it. Its turn activity could not be read, so turns may be running on it now: stopping it interrupts them.",
          "    Steps, in this order (one command per line):",
          "      # in the mounted Hermes home's config.yaml set webapp.enabled: true, webapp.host: 0.0.0.0, webapp.port: 9119",
          "      # copy every HERMES_DASHBOARD_* line from the old launcher's environment, if any, into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_container_None_0/hermes-home/.env (chmod 600)",
          "      # remove the dashboard: recreate the container without HERMES_DASHBOARD=1, or drop the Compose `dashboard` service and run: docker compose up -d --remove-orphans",
          "      # publish 9119 on the gateway's container and restart it, so its gateway serves the address"
        ],
        "expect_token": null,
        "found": true
      },
      "report": [
        {
          "supervisor": "container",
          "running": true,
          "pid": 4242,
          "unit": "hermes-dashboard.service",
          "scope": "user",
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
  }
};

export const systemSystemdPlans = {
  "running": {
    "held": {
      "applied": false,
      "dry_run": true,
      "exit_code": 3,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home",
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
        "running_turns": 1,
        "blockers": [
          "1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted)."
        ],
        "unsupported": [
          {
            "supervisor": "systemd",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
            "steps": []
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home",
          "  ✗ systemd hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
          "    Steps, in this order (one command per line):",
          "",
          "  ✗ Blockers (nothing will be changed):",
          "    • 1 turn(s) are running on the old backend. Wait for them to finish, or re-run with --force-interrupt (they are interrupted)."
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
      "exit_code": 6,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home",
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
        "running_turns": 1,
        "blockers": [],
        "unsupported": [
          {
            "supervisor": "systemd",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
            "steps": [
              "hermes -p default webapp --build-only",
              "hermes -p default config set webapp.enabled true",
              "hermes -p default config set webapp.host 0.0.0.0",
              "hermes -p default config set webapp.port 9119",
              "# copy HERMES_DASHBOARD_BASIC_AUTH_PASSWORD, HERMES_DASHBOARD_BASIC_AUTH_SECRET, HERMES_DASHBOARD_BASIC_AUTH_USERNAME from the old launcher's environment into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home/.env, then: chmod 600 /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home/.env  (values are never printed here)",
              "sudo systemctl disable --now hermes-dashboard.service",
              "systemctl is-active hermes-dashboard.service  # must print inactive (or failed) before the next line",
              "hermes -p default gateway status",
              "# then by what it prints: 'Gateway is not running' (no service installed) -> hermes -p default gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> hermes -p default gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> hermes -p default gateway restart",
              "# the gateway takes a few seconds to serve the address: if the next line prints 'The gateway is not running', wait 5 seconds and run it again",
              "hermes -p default webapp --status"
            ]
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home",
          "  ✗ systemd hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root. 1 turn(s) are running on it now: stopping it interrupts them, so let them finish first.",
          "    Steps, in this order (one command per line):",
          "      hermes -p default webapp --build-only",
          "      hermes -p default config set webapp.enabled true",
          "      hermes -p default config set webapp.host 0.0.0.0",
          "      hermes -p default config set webapp.port 9119",
          "      # copy HERMES_DASHBOARD_BASIC_AUTH_PASSWORD, HERMES_DASHBOARD_BASIC_AUTH_SECRET, HERMES_DASHBOARD_BASIC_AUTH_USERNAME from the old launcher's environment into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home/.env, then: chmod 600 /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_1_0/hermes-home/.env  (values are never printed here)",
          "      sudo systemctl disable --now hermes-dashboard.service",
          "      systemctl is-active hermes-dashboard.service  # must print inactive (or failed) before the next line",
          "      hermes -p default gateway status",
          "      # then by what it prints: 'Gateway is not running' (no service installed) -> hermes -p default gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> hermes -p default gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> hermes -p default gateway restart",
          "      # the gateway takes a few seconds to serve the address: if the next line prints 'The gateway is not running', wait 5 seconds and run it again",
          "      hermes -p default webapp --status"
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
  },
  "unknown": {
    "held": {
      "applied": false,
      "dry_run": true,
      "exit_code": 3,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home",
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
          "Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted)."
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
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home",
          "  ✗ systemd hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root.",
          "    Steps, in this order (one command per line):",
          "",
          "  ✗ Blockers (nothing will be changed):",
          "    • Cannot tell whether turns are running on the old backend: its turn activity could not be read. To move it, confirm it is idle, then re-run with --force-interrupt (any running turn is interrupted)."
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
      "exit_code": 6,
      "plan": {
        "home": "/var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home",
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
        "blockers": [],
        "unsupported": [
          {
            "supervisor": "systemd",
            "unit": "hermes-dashboard.service",
            "pid": 4242,
            "instructions": "The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root. Its turn activity could not be read, so turns may be running on it now: stopping it interrupts them.",
            "steps": [
              "hermes -p default webapp --build-only",
              "hermes -p default config set webapp.enabled true",
              "hermes -p default config set webapp.host 0.0.0.0",
              "hermes -p default config set webapp.port 9119",
              "# copy HERMES_DASHBOARD_BASIC_AUTH_PASSWORD, HERMES_DASHBOARD_BASIC_AUTH_SECRET, HERMES_DASHBOARD_BASIC_AUTH_USERNAME from the old launcher's environment into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home/.env, then: chmod 600 /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home/.env  (values are never printed here)",
              "sudo systemctl disable --now hermes-dashboard.service",
              "systemctl is-active hermes-dashboard.service  # must print inactive (or failed) before the next line",
              "hermes -p default gateway status",
              "# then by what it prints: 'Gateway is not running' (no service installed) -> hermes -p default gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> hermes -p default gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> hermes -p default gateway restart",
              "# the gateway takes a few seconds to serve the address: if the next line prints 'The gateway is not running', wait 5 seconds and run it again",
              "hermes -p default webapp --status"
            ]
          }
        ],
        "notices": [],
        "nothing_to_do": false,
        "steps": [],
        "text": [
          "Backend migration plan (dry run — nothing changed)",
          "  home:          /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home",
          "  ✗ systemd hermes-dashboard.service cannot be migrated automatically. Nothing changed.",
          "    The backend is a systemd SYSTEM unit: this command changes no system unit, and hermes -p default cannot run as root on a per-user install. Run the steps below as yourself; only the one `sudo systemctl` line needs root. Its turn activity could not be read, so turns may be running on it now: stopping it interrupts them.",
          "    Steps, in this order (one command per line):",
          "      hermes -p default webapp --build-only",
          "      hermes -p default config set webapp.enabled true",
          "      hermes -p default config set webapp.host 0.0.0.0",
          "      hermes -p default config set webapp.port 9119",
          "      # copy HERMES_DASHBOARD_BASIC_AUTH_PASSWORD, HERMES_DASHBOARD_BASIC_AUTH_SECRET, HERMES_DASHBOARD_BASIC_AUTH_USERNAME from the old launcher's environment into /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home/.env, then: chmod 600 /var/tmp/hermes-pytest-1000/r-deb5velf/pytest-of-mark/pytest-0/test_plan_systemd_system_None_0/hermes-home/.env  (values are never printed here)",
          "      sudo systemctl disable --now hermes-dashboard.service",
          "      systemctl is-active hermes-dashboard.service  # must print inactive (or failed) before the next line",
          "      hermes -p default gateway status",
          "      # then by what it prints: 'Gateway is not running' (no service installed) -> hermes -p default gateway install; 'service is stopped', 'service is not loaded' or 'registered with launchd' -> hermes -p default gateway start; 'service is running', 'supervised by launchd' or 'Running manually' -> hermes -p default gateway restart",
          "      # the gateway takes a few seconds to serve the address: if the next line prints 'The gateway is not running', wait 5 seconds and run it again",
          "      hermes -p default webapp --status"
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
  }
};
