import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapi.errors import register_error_handlers
from webapi.routes.chat import router as chat_router
from webapi.routes.config import router as config_router
from webapi.routes.health import router as health_router
from webapi.routes.jobs import router as jobs_router
from webapi.routes.memory import router as memory_router
from webapi.routes.models import router as models_router
from webapi.routes.sessions import router as sessions_router
from webapi.routes.skills import router as skills_router

# Default origins cover common local dev ports (3000-3010) + any explicitly
# configured origin via HERMES_CORS_ORIGINS (comma-separated).
_DEFAULT_ORIGINS = [f"http://localhost:{p}" for p in range(3000, 3011)] + \
                   [f"http://127.0.0.1:{p}" for p in range(3000, 3011)]

def _get_cors_origins() -> list[str]:
    extra = os.environ.get("HERMES_CORS_ORIGINS", "").strip()
    if extra:
        return [o.strip() for o in extra.split(",") if o.strip()]
    return _DEFAULT_ORIGINS


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hermes Web API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(sessions_router)
    app.include_router(chat_router)
    app.include_router(memory_router)
    app.include_router(skills_router)
    app.include_router(config_router)
    app.include_router(jobs_router)

    return app


app = create_app()
