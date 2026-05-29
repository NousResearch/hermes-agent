# Cavalry Windows Worker

This folder is the durable Windows-side worker bridge for Signal Room Cavalry jobs.

Run `run-cavalry-worker.ps1` on the Windows laptop from PowerShell. It polls `windows/cavalry/jobs/queued`, claims one job at a time, and moves job manifests through `running`, `done`, or `failed`.

Create jobs from the repo root with:

```powershell
python scripts/signal_room_cavalry_job_runner.py submit --job-id fee-machine-motion-pass --cwd C:\path\to\repo -- "C:\Program Files\Cavalry\Cavalry.exe" --run motion-pass.cav
```

The runner uses command arrays, not shell strings. Treat queued job JSON as trusted local automation input.
