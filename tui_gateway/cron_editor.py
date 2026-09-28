"""Full single-job reads behind the profile-scoped cron.manage RPC."""


def get_cron_job(params: dict) -> dict:
    from cron.jobs import AmbiguousJobReference, resolve_job_ref

    try:
        job = resolve_job_ref(params.get("job_id") or params.get("name") or "")
    except AmbiguousJobReference as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "job": job} if job else {"success": False, "error": "Job not found."}
