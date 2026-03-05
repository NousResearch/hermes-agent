---
sidebar_position: 1
title: "Installation"
description: "Install Hermes Agent on Linux, macOS, or WSL2"
---

# Installation

Get Hermes Agent up and running in under two minutes with the one-line installer, or follow the manual steps for full control.

## Quick Install

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::warning Windows
Native Windows is **not supported**. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run Hermes Agent from there. The install command above works inside WSL2.
:::

### What the Installer Does

The installer handles everything automatically:

- Installs [uv](https://docs.astral.sh/uv/) (fast Python package manager) if not present
- Installs Python 3.11 via uv if not already available (no sudo needed)
- Clones to `~/.hermes/hermes-agent` (with submodules: mini-swe-agent, tinker-atropos)
- Creates a virtual environment with Python 3.11
- Installs all dependencies and submodule packages
- Sets up the `hermes` command globally (no venv activation needed)
- Runs the interactive setup wizard

### After Installation

Reload your shell and start chatting:

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes setup       # Configure API keys (if you skipped during install)
hermes             # Start chatting!
```

---

## Prerequisites

| Requirement | Minimum Version | Check Command | Notes |
|-------------|----------------|---------------|-------|
| **Git** | Any recent | `git --version` | Required |
| **Node.js** | 18+ | `node --version` | Optional — needed for browser automation and WhatsApp bridge |
| **ripgrep** | Any | `rg --version` | Optional — faster file search (falls back to grep) |

:::info
Python and pip are **not** prerequisites. The installer uses [uv](https://docs.astral.sh/uv/) to provision Python 3.11 automatically (no sudo needed). If you already have Python 3.11+ installed, uv will use it.
:::

<details>
<summary><strong>Installing prerequisites by platform</strong></summary>

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install git
# Optional:
sudo apt install ripgrep nodejs npm
```

**macOS (Homebrew):**
```bash
brew install git
# Optional:
brew install ripgrep node
```

**Windows:**
Native Windows is not supported. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the Ubuntu/Debian instructions above.

</details>

---

## Manual Installation

If you prefer full control over the installation process, follow these steps.

### Step 1: Clone the Repository

Clone with `--recurse-submodules` to pull the required submodules:

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### Step 2: Install uv & Create Virtual Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with Python 3.11 (uv downloads it if not present — no sudo needed)
uv venv venv --python 3.11
```

:::tip
You do **not** need to activate the venv to use `hermes`. The entry point has a hardcoded shebang pointing to the venv Python, so it works globally once symlinked.
:::

### Step 3: Install Python Dependencies

```bash
# Tell uv which venv to install into
export VIRTUAL_ENV="$(pwd)/venv"

# Install with all extras
uv pip install -e ".[all]"
```

If you only want the core agent (no Telegram/Discord/cron support):
```bash
uv pip install -e "."
```

<details>
<summary><strong>Optional extras breakdown</strong></summary>

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `all` | Everything below | `uv pip install -e ".[all]"` |
| `messaging` | Telegram & Discord gateway | `uv pip install -e ".[messaging]"` |
| `cron` | Cron expression parsing for scheduled tasks | `uv pip install -e ".[cron]"` |
| `cli` | Terminal menu UI for setup wizard | `uv pip install -e ".[cli]"` |
| `modal` | Modal cloud execution backend | `uv pip install -e ".[modal]"` |
| `dev` | pytest & test utilities | `uv pip install -e ".[dev]"` |

You can combine extras: `uv pip install -e ".[messaging,cron]"`

</details>

### Step 4: Install Submodule Packages

```bash
# Terminal tool backend (required for terminal/command-execution)
uv pip install -e "./mini-swe-agent"

# RL training backend
uv pip install -e "./tinker-atropos"
```

Both are optional — if you skip them, the corresponding toolsets simply won't be available.

### Step 5: Install Node.js Dependencies (Optional)

Only needed for **browser automation** (Browserbase-powered):

```bash
npm install
```

### Step 6: Create the Configuration Directory

```bash
# Create the directory structure
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}

# Copy the example config file
cp cli-config.yaml.example ~/.hermes/config.yaml

# Create an empty .env file for API keys
touch ~/.hermes/.env
```

### Step 7: Add Your API Keys

Open `~/.hermes/.env` and add at minimum an LLM provider key:

```bash
# Required — at least one LLM provider:
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional — enable additional tools:
FIRECRAWL_API_KEY=fc-your-key          # Web search & scraping
FAL_KEY=your-fal-key                   # Image generation (FLUX)
```

Or set them via the CLI:
```bash
hermes config set OPENROUTER_API_KEY sk-or-v1-your-key-here
```

### Step 8: Add `hermes` to Your PATH

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
```

If `~/.local/bin` isn't on your PATH, add it to your shell config:

```bash
# Bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Fish
fish_add_path $HOME/.local/bin
```

### Step 9: Run the Setup Wizard (Optional)

```bash
hermes setup
```

### Step 10: Verify the Installation

```bash
hermes version    # Check that the command is available
hermes doctor     # Run diagnostics to verify everything is working
hermes status     # Check your configuration
hermes chat -q "Hello! What tools do you have available?"
```

---

## Quick-Reference: Manual Install (Condensed)

For those who just want the commands:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & enter
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# Install everything
uv pip install -e ".[all]"
uv pip install -e "./mini-swe-agent"
uv pip install -e "./tinker-atropos"
npm install  # optional, for browser tools

# Configure
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' >> ~/.hermes/.env

# Make hermes available globally
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# Verify
hermes doctor
hermes
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `hermes: command not found` | Reload your shell (`source ~/.bashrc`) or check PATH |
| `API key not set` | Run `hermes setup` or `hermes config set OPENROUTER_API_KEY your_key` |
| Missing config after update | Run `hermes config check` then `hermes config migrate` |

For more diagnostics, run `hermes doctor` — it will tell you exactly what's missing and how to fix it.
