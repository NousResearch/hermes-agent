from tools.claude import register as register_claude
from tools.codex import register as register_codex
from tools.files import register as register_files
from tools.system import register as register_system
from tools.status import register as register_status
from tools.hermes import register as register_hermes
from tools.grok import register as register_grok

ALL_TOOLS = [register_claude, register_codex, register_files, register_system, register_status, register_hermes, register_grok]
