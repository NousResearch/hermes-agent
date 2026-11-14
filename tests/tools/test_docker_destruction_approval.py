"""Container and volume destruction verbs need approval like the lifecycle verbs (#132483)."""

import pytest

from tools.approval import detect_dangerous_command


@pytest.mark.parametrize("command", [
    "docker rm web",
    "docker rm -f web",
    "podman rm web",
    "docker container rm web",
    "podman container rm web",
    "docker compose rm web",
    "docker-compose rm -f",
    "docker volume rm appdata",
    "docker container remove web",
    "podman container remove web",
    "docker volume remove appdata",
    "docker volume prune",
    "docker volume prune -f",
    "docker container prune",
    "docker system prune",
    "docker system prune -af",
    "docker container stop web",
    "docker container kill web",
    "docker container restart web",
    # A whitespace run between a global flag and its value must not hide the verb (#130511).
    "docker --log-level  warn container rm web",
    "docker --log-level\t\twarn system prune",
])
def test_container_destruction_requires_approval(command):
    dangerous, key, desc = detect_dangerous_command(command)
    assert dangerous is True, command
    assert key and desc, command


@pytest.mark.parametrize("command", [
    "docker ps",
    "docker ps -a",
    "docker container ls",
    "docker container inspect web",
    "docker volume ls",
    "docker system df",
    "docker run --rm alpine true",
    # Boundary from #132483: images and networks come back with a pull or a recreate.
    "docker image rm img",
    "docker rmi img",
    "docker network rm net",
])
def test_reads_and_recreatable_objects_stay_ungated(command):
    assert detect_dangerous_command(command) == (False, None, None), command
