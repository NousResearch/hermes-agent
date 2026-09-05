"""Tests for agent/verify/recipes.py — static run-recipe detection."""

import json

import pytest

from agent.verify.recipes import Recipe, detect_package_manager, detect_recipe


def write_pkg(root, data):
    (root / "package.json").write_text(json.dumps(data), encoding="utf-8")


class TestPackageManagerDetection:
    def test_pnpm_lock_wins(self, tmp_path):
        (tmp_path / "pnpm-lock.yaml").touch()
        (tmp_path / "yarn.lock").touch()
        assert detect_package_manager(tmp_path) == "pnpm"

    @pytest.mark.parametrize(
        "lockfile,manager",
        [
            ("bun.lock", "bun"),
            ("bun.lockb", "bun"),
            ("yarn.lock", "yarn"),
            ("package-lock.json", "npm"),
            ("uv.lock", "uv"),
            ("poetry.lock", "poetry"),
            ("Pipfile.lock", "pipenv"),
        ],
    )
    def test_single_lockfile(self, tmp_path, lockfile, manager):
        (tmp_path / lockfile).touch()
        assert detect_package_manager(tmp_path) == manager

    def test_no_lockfile(self, tmp_path):
        assert detect_package_manager(tmp_path) is None


class TestNodeDetection:
    def test_nextjs_with_pnpm(self, tmp_path):
        write_pkg(
            tmp_path,
            {
                "dependencies": {"next": "14.0.0"},
                "scripts": {"dev": "next dev", "build": "next build", "test": "jest"},
            },
        )
        (tmp_path / "pnpm-lock.yaml").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "nextjs"
        assert recipe.bootstrap == ["pnpm install"]
        assert recipe.build == ["pnpm build"]
        assert recipe.test == ["pnpm test"]
        assert recipe.start == "pnpm dev"
        assert recipe.port == 3000

    def test_vite_with_yarn(self, tmp_path):
        write_pkg(
            tmp_path,
            {
                "devDependencies": {"vite": "5.0.0"},
                "scripts": {"dev": "vite", "build": "vite build"},
            },
        )
        (tmp_path / "yarn.lock").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "vite"
        assert recipe.bootstrap == ["yarn install"]
        assert recipe.start == "yarn dev"
        assert recipe.port == 5173

    def test_bun_runner(self, tmp_path):
        write_pkg(tmp_path, {"scripts": {"start": "node server.js", "build": "tsc"}})
        (tmp_path / "bun.lockb").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "node"
        assert recipe.bootstrap == ["bun install"]
        assert recipe.start == "bun run start"

    def test_generic_node_defaults_to_npm(self, tmp_path):
        write_pkg(tmp_path, {"scripts": {"test": "mocha"}})
        recipe = detect_recipe(tmp_path)
        assert recipe.bootstrap == ["npm install"]
        assert recipe.test == ["npm run test"]
        assert recipe.start is None
        assert recipe.port is None

    def test_port_inferred_from_start_command(self, tmp_path):
        write_pkg(tmp_path, {"scripts": {"dev": "node server.js --port 4111"}})
        recipe = detect_recipe(tmp_path)
        assert recipe.port == 4111

    def test_cra(self, tmp_path):
        write_pkg(
            tmp_path,
            {"dependencies": {"react-scripts": "5.0"}, "scripts": {"start": "react-scripts start"}},
        )
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "cra"
        assert recipe.port == 3000

    def test_malformed_package_json_falls_through(self, tmp_path):
        (tmp_path / "package.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "go"


class TestPythonDetection:
    def test_django_via_manage_py(self, tmp_path):
        (tmp_path / "manage.py").touch()
        (tmp_path / "requirements.txt").write_text("django\n", encoding="utf-8")
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "django"
        assert recipe.test == ["python manage.py test"]
        assert recipe.port == 8000
        assert "runserver" in recipe.start

    def test_fastapi_uvicorn(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
        (tmp_path / "main.py").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "fastapi"
        assert recipe.start.startswith("uvicorn main:app")
        assert recipe.port == 8000

    def test_flask(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        (tmp_path / "app.py").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "flask"
        assert recipe.port == 5000

    def test_generic_python_uv(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / "uv.lock").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "python"
        assert recipe.bootstrap == ["uv sync"]

    def test_generic_python_pyproject_editable_install(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        recipe = detect_recipe(tmp_path)
        assert recipe.bootstrap == ["pip install -e ."]
        assert recipe.test == ["python -m unittest discover"]

    def test_pytest_when_tests_dir(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
        (tmp_path / "tests").mkdir()
        recipe = detect_recipe(tmp_path)
        assert recipe.test == ["pytest"]


class TestOtherEcosystems:
    def test_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x\n", encoding="utf-8")
        (tmp_path / "main.go").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "go"
        assert recipe.build == ["go build ./..."]
        assert recipe.start == "go run ."

    def test_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.rs").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "rust"
        assert recipe.start == "cargo run"

    def test_rust_library_has_no_start(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        recipe = detect_recipe(tmp_path)
        assert recipe.start is None

    def test_maven(self, tmp_path):
        (tmp_path / "pom.xml").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "maven"
        assert recipe.build == ["mvn package"]

    def test_gradle_wrapper(self, tmp_path):
        (tmp_path / "build.gradle").touch()
        (tmp_path / "gradlew").touch()
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "gradle"
        assert recipe.build == ["./gradlew build"]

    def test_makefile(self, tmp_path):
        (tmp_path / "Makefile").write_text(
            "install:\n\tpip install .\nbuild:\n\tmake -C src\ntest:\n\tpytest\nrun:\n\t./app\n",
            encoding="utf-8",
        )
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "make"
        assert recipe.bootstrap == ["make install"]
        assert recipe.build == ["make build"]
        assert recipe.test == ["make test"]
        assert recipe.start == "make run"

    def test_docker_compose(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        recipe = detect_recipe(tmp_path)
        assert recipe.kind == "compose"
        assert recipe.start == "docker compose up"

    def test_empty_dir_returns_none(self, tmp_path):
        assert detect_recipe(tmp_path) is None


class TestRecipeFromDict:
    def test_roundtrip(self):
        recipe = Recipe(name="X", kind="node", bootstrap=["npm install"], port=3000, start="npm run dev")
        restored = Recipe.from_dict(recipe.to_dict())
        assert restored == recipe

    def test_tolerates_garbage(self):
        assert Recipe.from_dict(None) is None
        assert Recipe.from_dict([]) is None
        assert Recipe.from_dict({"kind": "x"}) is None  # no name
        assert Recipe.from_dict({"name": "  "}) is None

    def test_grok_style_keys(self):
        recipe = Recipe.from_dict(
            {
                "appLabel": "Next.js",
                "appKind": "nextjs",
                "installCommands": ["npm install"],
                "buildCommands": ["npm run build"],
                "testCommands": ["npm test"],
                "startCommand": "npm run dev",
                "startPort": "3000",
            }
        )
        assert recipe.name == "Next.js"
        assert recipe.port == 3000
        assert recipe.bootstrap == ["npm install"]

    def test_invalid_port_dropped(self):
        recipe = Recipe.from_dict({"name": "x", "port": "not-a-port"})
        assert recipe.port is None
        recipe = Recipe.from_dict({"name": "x", "port": 99999999})
        assert recipe.port is None

    def test_bad_readiness_path_normalized(self):
        recipe = Recipe.from_dict({"name": "x", "readinessPath": "health"})
        assert recipe.readiness_path == "/"
        recipe = Recipe.from_dict({"name": "x", "readinessPath": "/health"})
        assert recipe.readiness_path == "/health"
