# Moho Windows Worker

This folder is the durable Windows-side worker bridge for Signal Room Moho pose-export jobs.

Run `run-moho-worker.ps1` on the Windows laptop from PowerShell. It polls `windows/moho/jobs/queued`, claims one job at a time, and moves job manifests through `running`, `done`, or `failed`.

Create pose-export jobs from the repo root with:

```powershell
python scripts/signal_room_moho_job_runner.py submit-pose-export --job-id suit-male-pose-export --project C:\path\to\scene.moho --output-dir C:\path\to\frames --candidate-name Suit_Male --expected-frame-count 8 --license-status "licensed internal review" -- "C:\Program Files\Moho 14\Moho.exe" --export C:\path\to\scene.moho
```

The job manifest records pose-export metadata for the later rig acting and pose installer gates.

The worker marks jobs left in `running/` for more than two hours as failed on the next poll. Moho pose exports also require the expected output directory to exist before a job can land in `done/`.
