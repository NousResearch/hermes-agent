# Hermes Operator Post-Training

This folder contains templates for turning redacted Hermes operator sessions,
Codex rollout logs, tool-call traces, and Harness-style outcomes into SFT/DPO
training data. Do not commit generated corpora, adapters, checkpoints, or
merged models.

Recommended flow:

```powershell
python scripts/export_training_corpus.py `
  --state-db "$env:USERPROFILE\.hermes\state.db" `
  --output training/corpora/hermes_operator_corpus.redacted.jsonl `
  --limit-sessions 200 `
  --include-logs `
  --harness-result "$env:USERPROFILE\.hermes\hypura-daemon.json" `
  --codex-sessions-dir "$env:USERPROFILE\.codex\sessions" `
  --limit-codex-rollouts 50

python scripts/build_sft_jsonl.py `
  training/corpora/hermes_operator_corpus.redacted.jsonl `
  training/corpora/hermes_operator_sft.jsonl
```

SFT generation keeps long evidence in the redacted corpus but bounds trainable
examples by default (`--max-messages 120`, `--max-message-chars 8000`,
`--max-tool-chars 2000`). Increase these only after a tokenizer smoke check
shows the resulting rows fit your chosen `sequence_len`.

Optional DPO seed data can be built from already redacted preference records:

```powershell
python scripts/build_dpo_jsonl.py `
  training/dpo_seed.jsonl `
  training/corpora/hermes_operator_dpo.seed.jsonl
```

Before training, scan the generated files for obvious secrets and local-only
identifiers:

```powershell
rg --pcre2 -n "(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}|hf_[A-Za-z0-9]{10,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9]{30,}|C:\\\\Users\\\\downl|C:/Users/downl|ngrok-free\.app|ngrok\.io|BWS_ACCESS_TOKEN=\S+|OPENAI_API_KEY=\S+)" `
  training/corpora/hermes_operator_corpus.redacted.jsonl `
  training/corpora/hermes_operator_sft.jsonl `
  training/corpora/hermes_operator_dpo.seed.jsonl
```

`rg` exit code 1 means no matches were found.

Run the operator eval manifest before and after training:

```powershell
python scripts/run_hermes_operator_eval.py `
  --evals evals/hermes_operator_eval.jsonl `
  --sft training/corpora/hermes_operator_sft.jsonl
```

Check that the local files and `base_model` are trainable before launching
Axolotl. `base_model` must be the matching HF checkpoint or merged HF model,
not the served GGUF:

```powershell
python scripts/check_training_ready.py `
  --sft training/corpora/hermes_operator_sft.jsonl `
  --dpo training/corpora/hermes_operator_dpo.seed.jsonl `
  --qlora-config training/qlora_config.yaml
```

For machine-local values, copy `training/local-env.example` to
`training/local.env` and fill in the current GGUF path, the matching HF
checkpoint, and `LLAMA_CPP_ROOT`. Keep `training/local.env` uncommitted.
Render a local config from the template:

```powershell
python scripts/render_training_config.py `
  --env-file training/local.env `
  --template training/qlora_config.yaml `
  --output training/local_qlora_config.yaml `
  --sft training/corpora/hermes_operator_sft.jsonl
```

Train with Axolotl from `training/local_qlora_config.yaml`. If the preference
set is large enough, use `training/dpo_config.yaml` after the SFT adapter is
available. Axolotl's current CLI shape is:

```bash
axolotl preprocess training/local_qlora_config.yaml --debug --debug-num-examples 3
axolotl train training/local_qlora_config.yaml
axolotl merge-lora training/local_qlora_config.yaml --lora-model-dir=training/runs/hermes-operator-qlora
```

On Linux/WSL/CUDA, the scripted path is:

```bash
export HERMES_OPERATOR_CONFIG=training/local_qlora_config.yaml
export LLAMA_CPP_ROOT=/path/to/llama.cpp
scripts/run_hermes_operator_posttrain.sh all
```

If Axolotl is unavailable but Transformers/PEFT/bitsandbytes are installed,
use the fallback trainer. It consumes the same redacted SFT file and writes a
LoRA adapter plus an optional merged HF model:

```powershell
python scripts/train_hermes_operator_peft.py check-deps
python scripts/train_hermes_operator_peft.py env-report
$env:HERMES_OPERATOR_CAUSAL_LM_BASE = "$env:HERMES_OPERATOR_BASE_MODEL-causal-lm"
python scripts/prepare_hf_causal_lm_checkpoint.py `
  "$env:HERMES_OPERATOR_BASE_MODEL" `
  "$env:HERMES_OPERATOR_CAUSAL_LM_BASE" `
  --link-files
python scripts/train_hermes_operator_peft.py config-smoke `
  --base-model "$env:HERMES_OPERATOR_CAUSAL_LM_BASE"
python scripts/train_hermes_operator_peft.py tokenize-smoke `
  --base-model "$env:HERMES_OPERATOR_CAUSAL_LM_BASE" `
  --sft training/corpora/hermes_operator_sft.jsonl `
  --limit 3 `
  --sequence-len 8192
python scripts/train_hermes_operator_peft.py train-smoke `
  --base-model "$env:HERMES_OPERATOR_CAUSAL_LM_BASE" `
  --sft training/corpora/hermes_operator_sft.jsonl
python scripts/train_hermes_operator_peft.py train `
  --base-model "$env:HERMES_OPERATOR_CAUSAL_LM_BASE" `
  --sft training/corpora/hermes_operator_sft.jsonl `
  --max-steps -1
python scripts/train_hermes_operator_peft.py merge `
  --base-model "$env:HERMES_OPERATOR_CAUSAL_LM_BASE" `
  --adapter-dir training/runs/hermes-operator-peft `
  --merged-dir training/runs/hermes-operator-peft-merged
```

`train-smoke` writes to `training/runs/hermes-operator-peft-smoke` and runs one
training step on one SFT row. Use it to verify model loading, 4-bit setup,
Trainer wiring, and adapter saving before a full run.

On this Windows Python, `nvidia-smi` may show the GPU while `torch.cuda` is
still unavailable. In that case `train-smoke` will fall back to CPU behavior
and may not finish promptly. Treat `env-report` as the authoritative check for
the active Python training environment; run the real LoRA job only where it
reports CUDA availability.

After merge, export a new GGUF on Windows:

```powershell
scripts/export_gguf.ps1 `
  -MergedModelDir training/runs/hermes-operator-qlora/merged `
  -OutputGguf H:\elt_data\releases\hermes-operator-q8_0.gguf `
  -LlamaCppRoot C:\src\llama.cpp `
  -PythonExe C:\Users\downl\AppData\Local\Programs\Python\Python312\python.exe `
  -Quantization Q8_0
```

If the converter and quantizer are installed separately, pass them explicitly:

```powershell
scripts/export_gguf.ps1 `
  -MergedModelDir H:\elt_data\posttrain\hermes_operator_peft_smoke_merged `
  -OutputGguf H:\elt_data\posttrain\hermes_operator_peft_smoke-q8_0.gguf `
  -ConvertScript H:\elt_data\tools\llama.cpp\convert_hf_to_gguf.py `
  -QuantizeExe C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-quantize.exe `
  -PythonExe C:\Users\downl\AppData\Local\Programs\Python\Python312\python.exe `
  -NoMtp `
  -Quantization Q8_0
```

Use `-NoMtp` for Qwen3.5-family checkpoints that do not contain MTP tensors.

Keep durable facts, local secrets, exact machine paths, and mutable runtime
state in memory/RAG. The LoRA should learn operator procedure and tool-call
discipline, not private machine state.

Use `--harness-log` only when you need bounded raw log context in the redacted
corpus. Prefer `--harness-result` for JSON/JSONL health checks, smoke results,
or daemon status snapshots that should become trainable SFT examples.

On Windows, prefer running Axolotl in WSL2/Linux or on a separate CUDA Linux
host. The repo-side checks and corpus generation work on Windows, but Axolotl's
training stack is Linux/macOS oriented. If only the GGUF is available, stop and
recover the corresponding HF checkpoint first; do not point `base_model` at a
GGUF.
