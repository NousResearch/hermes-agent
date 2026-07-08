---
sidebar_position: 2
title: "ç¯å¢ƒå˜é‡"
description: "Hermes Agent ä½¿ç”¨çš„æ‰€æœ‰ç¯å¢ƒå˜é‡å®Œæ•´å‚è€ƒ"
---

# ç¯å¢ƒå˜é‡å‚è€ƒ

æ‰€æœ‰å˜é‡å‡å†™å…¥ `~/.hermes/.env`ã€‚ä¹Ÿå¯ä»¥ä½¿ç”¨ `hermes config set VAR value` è¿›è¡Œè®¾ç½®ã€‚

## LLM æä¾›å•†

| å˜é‡ | æè¿° |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API å¯†é’¥ï¼ˆæ¨èï¼Œçµæ´»æ€§å¼ºï¼‰ |
| `OPENROUTER_BASE_URL` | è¦†ç›– OpenRouter å…¼å®¹çš„ base URL |
| `HERMES_OPENROUTER_CACHE` | å¯ç”¨ OpenRouter å“åº”ç¼“å­˜ï¼ˆ`1`/`true`/`yes`/`on`ï¼‰ã€‚è¦†ç›– config.yaml ä¸­çš„ `openrouter.response_cache`ã€‚å‚è§ [Response Caching](https://openrouter.ai/docs/guides/features/response-caching)ã€‚ |
| `HERMES_OPENROUTER_CACHE_TTL` | ç¼“å­˜ TTLï¼ˆç§’ï¼Œ1-86400ï¼‰ã€‚è¦†ç›– config.yaml ä¸­çš„ `openrouter.response_cache_ttl`ã€‚ |
| `NOUS_BASE_URL` | è¦†ç›– Nous Portal base URLï¼ˆæå°‘ä½¿ç”¨ï¼›ä»…ç”¨äºå¼€å‘/æµ‹è¯•ï¼‰ |
| `NOUS_INFERENCE_BASE_URL` | ç›´æ¥è¦†ç›– Nous æ¨ç†ç«¯ç‚¹ |
| `OPENAI_API_KEY` | è‡ªå®šä¹‰ OpenAI å…¼å®¹ç«¯ç‚¹çš„ API å¯†é’¥ï¼ˆä¸ `OPENAI_BASE_URL` é…åˆä½¿ç”¨ï¼‰ |
| `OPENAI_BASE_URL` | è‡ªå®šä¹‰ç«¯ç‚¹çš„ base URLï¼ˆVLLMã€SGLang ç­‰ï¼‰ |
| `COPILOT_GITHUB_TOKEN` | ç”¨äº Copilot API çš„ GitHub tokenâ€”â€”æœ€é«˜ä¼˜å…ˆçº§ï¼ˆOAuth `gho_*` æˆ–ç»†ç²’åº¦ PAT `github_pat_*`ï¼›ç»å…¸ PAT `ghp_*` **ä¸æ”¯æŒ**ï¼‰ |
| `GH_TOKEN` | GitHub tokenâ€”â€”Copilot ç¬¬äºŒä¼˜å…ˆçº§ï¼ˆä¹Ÿä¾› `gh` CLI ä½¿ç”¨ï¼‰ |
| `GITHUB_TOKEN` | GitHub tokenâ€”â€”Copilot ç¬¬ä¸‰ä¼˜å…ˆçº§ |
| `HERMES_COPILOT_ACP_COMMAND` | è¦†ç›– Copilot ACP CLI äºŒè¿›åˆ¶è·¯å¾„ï¼ˆé»˜è®¤ï¼š`copilot`ï¼‰ |
| `COPILOT_CLI_PATH` | `HERMES_COPILOT_ACP_COMMAND` çš„åˆ«å |
| `HERMES_COPILOT_ACP_ARGS` | è¦†ç›– Copilot ACP å‚æ•°ï¼ˆé»˜è®¤ï¼š`--acp --stdio`ï¼‰ |
| `COPILOT_ACP_BASE_URL` | è¦†ç›– Copilot ACP base URL |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM API å¯†é’¥ï¼ˆ[z.ai](https://z.ai)ï¼‰ |
| `ZAI_API_KEY` | `GLM_API_KEY` çš„åˆ«å |
| `Z_AI_API_KEY` | `GLM_API_KEY` çš„åˆ«å |
| `GLM_BASE_URL` | è¦†ç›– z.ai base URLï¼ˆé»˜è®¤ï¼š`https://api.z.ai/api/paas/v4`ï¼‰ |
| `KIMI_API_KEY` | Kimi / Moonshot AI API å¯†é’¥ï¼ˆ[moonshot.ai](https://platform.moonshot.ai)ï¼‰ |
| `KIMI_BASE_URL` | è¦†ç›– Kimi base URLï¼ˆé»˜è®¤ï¼š`https://api.moonshot.ai/v1`ï¼‰ |
| `KIMI_CN_API_KEY` | Kimi / Moonshot ä¸­å›½åŒº API å¯†é’¥ï¼ˆ[moonshot.cn](https://platform.moonshot.cn)ï¼‰ |
| `ARCEEAI_API_KEY` | Arcee AI API å¯†é’¥ï¼ˆ[chat.arcee.ai](https://chat.arcee.ai/)ï¼‰ |
| `ARCEE_BASE_URL` | è¦†ç›– Arcee base URLï¼ˆé»˜è®¤ï¼š`https://api.arcee.ai/api/v1`ï¼‰ |
| `GMI_API_KEY` | GMI Cloud API å¯†é’¥ï¼ˆ[gmicloud.ai](https://www.gmicloud.ai/)ï¼‰ |
| `GMI_BASE_URL` | è¦†ç›– GMI Cloud base URLï¼ˆé»˜è®¤ï¼š`https://api.gmi-serving.com/v1`ï¼‰ |
| `MINIMAX_API_KEY` | MiniMax API å¯†é’¥â€”â€”å…¨çƒç«¯ç‚¹ï¼ˆ[minimax.io](https://www.minimax.io)ï¼‰ã€‚**`minimax-oauth` ä¸ä½¿ç”¨æ­¤å˜é‡**ï¼ˆOAuth è·¯å¾„é€šè¿‡æµè§ˆå™¨ç™»å½•ï¼‰ã€‚ |
| `MINIMAX_BASE_URL` | è¦†ç›– MiniMax base URLï¼ˆé»˜è®¤ï¼š`https://api.minimax.io/anthropic`â€”â€”Hermes ä½¿ç”¨ MiniMax çš„ Anthropic Messages å…¼å®¹ç«¯ç‚¹ï¼‰ã€‚**`minimax-oauth` ä¸ä½¿ç”¨æ­¤å˜é‡**ã€‚ |
| `MINIMAX_CN_API_KEY` | MiniMax API å¯†é’¥â€”â€”ä¸­å›½åŒºç«¯ç‚¹ï¼ˆ[minimaxi.com](https://www.minimaxi.com)ï¼‰ã€‚**`minimax-oauth` ä¸ä½¿ç”¨æ­¤å˜é‡**ï¼ˆOAuth è·¯å¾„é€šè¿‡æµè§ˆå™¨ç™»å½•ï¼‰ã€‚ |
| `MINIMAX_CN_BASE_URL` | è¦†ç›– MiniMax ä¸­å›½åŒº base URLï¼ˆé»˜è®¤ï¼š`https://api.minimaxi.com/anthropic`ï¼‰ã€‚**`minimax-oauth` ä¸ä½¿ç”¨æ­¤å˜é‡**ã€‚ |
| `KILOCODE_API_KEY` | Kilo Code API å¯†é’¥ï¼ˆ[kilo.ai](https://kilo.ai)ï¼‰ |
| `KILOCODE_BASE_URL` | è¦†ç›– Kilo Code base URLï¼ˆé»˜è®¤ï¼š`https://api.kilo.ai/api/gateway`ï¼‰ |
| `XIAOMI_API_KEY` | å°ç±³ MiMo API å¯†é’¥ï¼ˆ[platform.xiaomimimo.com](https://platform.xiaomimimo.com)ï¼‰ |
| `XIAOMI_BASE_URL` | è¦†ç›–å°ç±³ MiMo base URLï¼ˆé»˜è®¤ï¼š`https://api.xiaomimimo.com/v1`ï¼‰ |
| `TOKENHUB_API_KEY` | è…¾è®¯ TokenHub API å¯†é’¥ï¼ˆ[tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)ï¼‰ |
| `TOKENHUB_BASE_URL` | è¦†ç›–è…¾è®¯ TokenHub base URLï¼ˆé»˜è®¤ï¼š`https://tokenhub.tencentmaas.com/v1`ï¼‰ |
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI API å¯†é’¥ï¼ˆ[ai.azure.com](https://ai.azure.com/)ï¼‰ã€‚å½“ `model.auth_mode: entra_id` æ—¶ä¸éœ€è¦ |
| `AZURE_FOUNDRY_BASE_URL` | Microsoft Foundry ç«¯ç‚¹ URLï¼ˆä¾‹å¦‚ OpenAI é£æ ¼ï¼š`https://<resource>.openai.azure.com/openai/v1`ï¼ŒAnthropic é£æ ¼ï¼š`https://<resource>.services.ai.azure.com/anthropic`ï¼‰ |
| `AZURE_ANTHROPIC_KEY` | ç”¨äº `provider: anthropic` + `base_url` æŒ‡å‘ Microsoft Foundry Claude éƒ¨ç½²çš„ Azure Anthropic API å¯†é’¥ï¼ˆå½“åŒæ—¶é…ç½®äº† Anthropic å’Œ Azure Anthropic æ—¶ï¼Œä½œä¸º `ANTHROPIC_API_KEY` çš„æ›¿ä»£ï¼‰ |
| `AZURE_TENANT_ID` | Entra ID ç§Ÿæˆ· IDï¼ˆæœåŠ¡ä¸»ä½“æµç¨‹ï¼›å½“ `model.auth_mode: entra_id` æ—¶ç”± `azure-identity` è¯»å–ï¼‰ |
| `AZURE_CLIENT_ID` | Entra ID å®¢æˆ·ç«¯ IDï¼ˆæœåŠ¡ä¸»ä½“ã€å·¥ä½œè´Ÿè½½æ ‡è¯†æˆ–ç”¨æˆ·åˆ†é…çš„æ‰˜ç®¡æ ‡è¯†ï¼‰ |
| `AZURE_CLIENT_SECRET` | `EnvironmentCredential` ä½¿ç”¨çš„æœåŠ¡ä¸»ä½“å¯†é’¥ |
| `AZURE_CLIENT_CERTIFICATE_PATH` | æœåŠ¡ä¸»ä½“è¯ä¹¦ï¼ˆ`AZURE_CLIENT_SECRET` çš„æ›¿ä»£æ–¹æ¡ˆï¼‰ |
| `AZURE_FEDERATED_TOKEN_FILE` | AKS Workload Identity / OIDC æµç¨‹çš„è”åˆ token æ–‡ä»¶è·¯å¾„ |
| `AZURE_AUTHORITY_HOST` | ä¸»æƒäº‘ authority è¦†ç›–ï¼ˆä¾‹å¦‚ Azure Government ä½¿ç”¨ `https://login.microsoftonline.us`ï¼‰ã€‚å‚è§ [Azure Foundry æŒ‡å—](/guides/azure-foundry#sovereign-clouds-government-china) |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | App Serviceã€Functions å’Œ Container Apps çš„æ‰˜ç®¡æ ‡è¯†ç«¯ç‚¹ï¼›VM é€šå¸¸ä½¿ç”¨ IMDS è€Œä¸è®¾ç½®è¿™äº›å˜é‡ |
| `HF_TOKEN` | Hugging Face Inference Providers tokenï¼ˆ[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)ï¼‰ |
| `HF_BASE_URL` | è¦†ç›– Hugging Face base URLï¼ˆé»˜è®¤ï¼š`https://router.huggingface.co/v1`ï¼‰ |
| `GOOGLE_API_KEY` | Google AI Studio API å¯†é’¥ï¼ˆ[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)ï¼‰ |
| `GEMINI_API_KEY` | `GOOGLE_API_KEY` çš„åˆ«å |
| `GEMINI_BASE_URL` | è¦†ç›– Google AI Studio base URL |
| `ANTHROPIC_API_KEY` | Anthropic Console API å¯†é’¥ï¼ˆ[console.anthropic.com](https://console.anthropic.com/)ï¼‰ |
| `ANTHROPIC_TOKEN` | æ‰‹åŠ¨æˆ–æ—§ç‰ˆ Anthropic OAuth/setup-token è¦†ç›– |
| `DASHSCOPE_API_KEY` | Qwen Cloudï¼ˆé˜¿é‡Œå·´å·´ DashScopeï¼‰Qwen æ¨¡å‹ API å¯†é’¥ï¼ˆ[modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)ï¼‰ |
| `DASHSCOPE_BASE_URL` | è‡ªå®šä¹‰ DashScope base URLï¼ˆé»˜è®¤ï¼š`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`ï¼›ä¸­å›½å¤§é™†åŒºåŸŸä½¿ç”¨ `https://dashscope.aliyuncs.com/compatible-mode/v1`ï¼‰ |
| `DEEPSEEK_API_KEY` | ç›´æ¥è®¿é—® DeepSeek çš„ API å¯†é’¥ï¼ˆ[platform.deepseek.com](https://platform.deepseek.com/api_keys)ï¼‰ |
| `DEEPSEEK_BASE_URL` | è‡ªå®šä¹‰ DeepSeek API base URL |
| `NOVITA_API_KEY` | NovitaAI API å¯†é’¥â€”â€”é¢å‘ Model APIã€Agent Sandbox å’Œ GPU Cloud çš„ AI åŸç”Ÿäº‘ï¼ˆ[novita.ai/settings/key-management](https://novita.ai/settings/key-management)ï¼‰ |
| `NOVITA_BASE_URL` | è¦†ç›– NovitaAI base URLï¼ˆé»˜è®¤ï¼š`https://api.novita.ai/openai/v1`ï¼‰ |
| `NVIDIA_API_KEY` | NVIDIA NIM API å¯†é’¥â€”â€”Nemotron åŠå¼€æºæ¨¡å‹ï¼ˆ[build.nvidia.com](https://build.nvidia.com)ï¼‰ |
| `NVIDIA_BASE_URL` | è¦†ç›– NVIDIA base URLï¼ˆé»˜è®¤ï¼š`https://integrate.api.nvidia.com/v1`ï¼›æœ¬åœ° NIM ç«¯ç‚¹è®¾ä¸º `http://localhost:8000/v1`ï¼‰ |
| `STEPFUN_API_KEY` | StepFun API å¯†é’¥â€”â€”Step ç³»åˆ—æ¨¡å‹ï¼ˆ[platform.stepfun.com](https://platform.stepfun.com)ï¼‰ |
| `STEPFUN_BASE_URL` | è¦†ç›– StepFun base URLï¼ˆé»˜è®¤ï¼š`https://api.stepfun.com/v1`ï¼‰ |
| `OLLAMA_API_KEY` | Ollama Cloud API å¯†é’¥â€”â€”æ— éœ€æœ¬åœ° GPU çš„æ‰˜ç®¡ Ollama ç›®å½•ï¼ˆ[ollama.com/settings/keys](https://ollama.com/settings/keys)ï¼‰ |
| `OLLAMA_BASE_URL` | è¦†ç›– Ollama Cloud base URLï¼ˆé»˜è®¤ï¼š`https://ollama.com/v1`ï¼‰ |
| `XAI_API_KEY` | xAIï¼ˆGrokï¼‰API å¯†é’¥ï¼Œæ”¯æŒèŠå¤©ã€TTS å’Œç½‘ç»œæœç´¢ï¼ˆ[console.x.ai](https://console.x.ai/)ï¼‰ |
| `XAI_BASE_URL` | è¦†ç›– xAI base URLï¼ˆé»˜è®¤ï¼š`https://api.x.ai/v1`ï¼‰ |
| `MISTRAL_API_KEY` | Mistral API å¯†é’¥ï¼Œç”¨äº Voxtral TTS å’Œ Voxtral STTï¼ˆ[console.mistral.ai](https://console.mistral.ai)ï¼‰ |
| `AWS_REGION` | Bedrock æ¨ç†çš„ AWS åŒºåŸŸï¼ˆä¾‹å¦‚ `us-east-1`ã€`eu-central-1`ï¼‰ã€‚ç”± boto3 è¯»å–ã€‚ |
| `AWS_PROFILE` | Bedrock è®¤è¯çš„ AWS å‘½åé…ç½®æ–‡ä»¶ï¼ˆè¯»å– `~/.aws/credentials`ï¼‰ã€‚ä¸è®¾ç½®åˆ™ä½¿ç”¨é»˜è®¤ boto3 å‡­è¯é“¾ã€‚ |
| `BEDROCK_BASE_URL` | è¦†ç›– Bedrock runtime base URLï¼ˆé»˜è®¤ï¼š`https://bedrock-runtime.us-east-1.amazonaws.com`ï¼›é€šå¸¸ä¸è®¾ç½®ï¼Œæ”¹ç”¨ `AWS_REGION`ï¼‰ |
| `HERMES_QWEN_BASE_URL` | Qwen Portal base URL è¦†ç›–ï¼ˆé»˜è®¤ï¼š`https://portal.qwen.ai/v1`ï¼‰ |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen API å¯†é’¥â€”â€”æŒ‰éœ€ä»˜è´¹è®¿é—®ç²¾é€‰æ¨¡å‹ï¼ˆ[opencode.ai](https://opencode.ai/auth)ï¼‰ |
| `OPENCODE_ZEN_BASE_URL` | è¦†ç›– OpenCode Zen base URL |
| `OPENCODE_GO_API_KEY` | OpenCode Go API å¯†é’¥â€”â€”$10/æœˆè®¢é˜…å¼€æºæ¨¡å‹ï¼ˆ[opencode.ai](https://opencode.ai/auth)ï¼‰ |
| `OPENCODE_GO_BASE_URL` | è¦†ç›– OpenCode Go base URL |
| `CLAUDE_CODE_OAUTH_TOKEN` | æ‰‹åŠ¨å¯¼å‡ºæ—¶çš„æ˜¾å¼ Claude Code token è¦†ç›– |
| `HERMES_MODEL` | åœ¨è¿›ç¨‹çº§åˆ«è¦†ç›–æ¨¡å‹åç§°ï¼ˆä¾› cron è°ƒåº¦å™¨ä½¿ç”¨ï¼›æ­£å¸¸ä½¿ç”¨è¯·ä¼˜å…ˆåœ¨ `config.yaml` ä¸­é…ç½®ï¼‰ |
| `VOICE_TOOLS_OPENAI_KEY` | OpenAI è¯­éŸ³è½¬æ–‡å­—å’Œæ–‡å­—è½¬è¯­éŸ³æä¾›å•†çš„é¦–é€‰ OpenAI å¯†é’¥ |
| `HERMES_LOCAL_STT_COMMAND` | å¯é€‰çš„æœ¬åœ°è¯­éŸ³è½¬æ–‡å­—å‘½ä»¤æ¨¡æ¿ã€‚æ”¯æŒ `{input_path}`ã€`{output_dir}`ã€`{language}` å’Œ `{model}` å ä½ç¬¦ |
| `HERMES_LOCAL_STT_LANGUAGE` | ä¼ é€’ç»™ `HERMES_LOCAL_STT_COMMAND` æˆ–è‡ªåŠ¨æ£€æµ‹çš„æœ¬åœ° `whisper` CLI å›é€€çš„é»˜è®¤è¯­è¨€ï¼ˆé»˜è®¤ï¼š`en`ï¼‰ |
| `HERMES_HOME` | è¦†ç›– Hermes é…ç½®ç›®å½•ï¼ˆé»˜è®¤ï¼š`~/.hermes`ï¼‰ã€‚åŒæ—¶é™å®š gateway PID æ–‡ä»¶å’Œ systemd æœåŠ¡åç§°ï¼Œå…è®¸å¤šä¸ªå®‰è£…å¹¶å‘è¿è¡Œ |
| `HERMES_GIT_BASH_PATH` | **ä»… Windowsã€‚** è¦†ç›–ç»ˆç«¯å·¥å…·çš„ `bash.exe` å‘ç°è·¯å¾„ã€‚å¯æŒ‡å‘ä»»æ„ bashâ€”â€”å®Œæ•´ Git-for-Windows å®‰è£…ã€é€šè¿‡ç¬¦å·é“¾æ¥çš„ WSL bashã€MSYS2ã€Cygwinã€‚å®‰è£…ç¨‹åºä¼šè‡ªåŠ¨å°†å…¶è®¾ç½®ä¸ºæ‰€é…ç½®çš„ PortableGitã€‚å‚è§ [Windowsï¼ˆåŸç”Ÿï¼‰æŒ‡å—](../user-guide/windows-native.md#how-hermes-runs-shell-commands-on-windows) |
| `HERMES_DISABLE_WINDOWS_UTF8` | **ä»… Windowsã€‚** è®¾ä¸º `1` å¯ç¦ç”¨ UTF-8 stdio shimï¼ˆ`configure_windows_stdio()`ï¼‰ï¼Œå›é€€åˆ°æ§åˆ¶å°çš„æœ¬åœ°ä»£ç é¡µã€‚ç”¨äºæ’æŸ¥ç¼–ç é—®é¢˜ï¼›æ­£å¸¸æ“ä½œä¸­æå°‘éœ€è¦ |
| `HERMES_KANBAN_HOME` | è¦†ç›–é”šå®š kanban çœ‹æ¿ï¼ˆæ•°æ®åº“ + å·¥ä½œåŒº + å·¥ä½œæ—¥å¿—ï¼‰çš„å…±äº« Hermes æ ¹ç›®å½•ã€‚å›é€€åˆ° `get_default_hermes_root()`ï¼ˆä»»æ„æ´»åŠ¨ profile çš„çˆ¶ç›®å½•ï¼‰ã€‚é€‚ç”¨äºæµ‹è¯•å’Œéå¸¸è§„éƒ¨ç½² |
| `HERMES_KANBAN_BOARD` | ä¸ºå½“å‰è¿›ç¨‹å›ºå®šæ´»åŠ¨ kanban çœ‹æ¿ã€‚ä¼˜å…ˆäº `~/.hermes/kanban/current`ï¼›è°ƒåº¦å™¨å°†å…¶æ³¨å…¥å·¥ä½œè¿›ç¨‹å­è¿›ç¨‹ç¯å¢ƒï¼Œä½¿å·¥ä½œè¿›ç¨‹æ— æ³•çœ‹åˆ°å…¶ä»–çœ‹æ¿ä¸Šçš„ä»»åŠ¡ã€‚é»˜è®¤ä¸º `default`ã€‚slug éªŒè¯ï¼šå°å†™å­—æ¯æ•°å­— + è¿å­—ç¬¦ + ä¸‹åˆ’çº¿ï¼Œ1-64 å­—ç¬¦ |
| `HERMES_KANBAN_DB` | ç›´æ¥å›ºå®š kanban æ•°æ®åº“æ–‡ä»¶è·¯å¾„ï¼ˆæœ€é«˜ä¼˜å…ˆçº§ï¼›ä¼˜å…ˆäº `HERMES_KANBAN_BOARD` å’Œ `HERMES_KANBAN_HOME`ï¼‰ã€‚è°ƒåº¦å™¨å°†å…¶æ³¨å…¥å·¥ä½œè¿›ç¨‹å­è¿›ç¨‹ç¯å¢ƒï¼Œä½¿ profile å·¥ä½œè¿›ç¨‹æ”¶æ•›åˆ°è°ƒåº¦å™¨çš„çœ‹æ¿ |
| `HERMES_KANBAN_WORKSPACES_ROOT` | ç›´æ¥å›ºå®š kanban å·¥ä½œåŒºæ ¹ç›®å½•ï¼ˆå·¥ä½œåŒºæœ€é«˜ä¼˜å…ˆçº§ï¼›ä¼˜å…ˆäº `HERMES_KANBAN_HOME`ï¼‰ã€‚è°ƒåº¦å™¨å°†å…¶æ³¨å…¥å·¥ä½œè¿›ç¨‹å­è¿›ç¨‹ç¯å¢ƒ |
| `HERMES_KANBAN_DISPATCH_IN_GATEWAY` | `kanban.dispatch_in_gateway` çš„è¿è¡Œæ—¶è¦†ç›–ã€‚è®¾ä¸º `0`ã€`false`ã€`no` æˆ– `off` å¯é˜»æ­¢ gateway å¯åŠ¨å†…åµŒ Kanban è°ƒåº¦å™¨ï¼›ä»»ä½•å…¶ä»–éç©ºå€¼åˆ™å¯ç”¨ã€‚é€‚ç”¨äºç‹¬ç«‹è°ƒåº¦å™¨è¿›ç¨‹æ‹¥æœ‰çœ‹æ¿çš„åœºæ™¯ã€‚ |

## æä¾›å•†è®¤è¯ï¼ˆOAuthï¼‰

å¯¹äºåŸç”Ÿ Anthropic è®¤è¯ï¼ŒHermes åœ¨ Claude Code è‡ªèº«å‡­è¯æ–‡ä»¶å­˜åœ¨æ—¶ä¼˜å…ˆä½¿ç”¨ï¼Œå› ä¸ºè¿™äº›å‡­è¯å¯ä»¥è‡ªåŠ¨åˆ·æ–°ã€‚**é’ˆå¯¹ Anthropic çš„ OAuth éœ€è¦è´­ä¹°äº†é¢å¤–ä½¿ç”¨é¢åº¦çš„ Claude Max è®¡åˆ’**â€”â€”Hermes ä»¥ Claude Code èº«ä»½è·¯ç”±ï¼Œä»…æ¶ˆè€— Max è®¡åˆ’çš„é¢å¤–/è¶…é¢é¢åº¦ï¼Œä¸æ¶ˆè€—åŸºç¡€ Max é…é¢ï¼Œä¸”ä¸é€‚ç”¨äº Claude Proã€‚æ²¡æœ‰ Max + é¢å¤–é¢åº¦æ—¶ï¼Œè¯·æ”¹ç”¨ API å¯†é’¥ã€‚`ANTHROPIC_TOKEN` ç­‰ç¯å¢ƒå˜é‡ä½œä¸ºæ‰‹åŠ¨è¦†ç›–ä»ç„¶æœ‰ç”¨ï¼Œä½†ä¸å†æ˜¯ Claude Max ç™»å½•çš„é¦–é€‰è·¯å¾„ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `HERMES_PORTAL_BASE_URL` | è¦†ç›– Nous Portal URLï¼ˆç”¨äºå¼€å‘/æµ‹è¯•ï¼‰ |
| `NOUS_INFERENCE_BASE_URL` | è¦†ç›– Nous æ¨ç† API URL |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | é‡æ–°é“¸é€ å‰çš„æœ€å° agent å¯†é’¥ TTLï¼ˆé»˜è®¤ï¼š1800 = 30 åˆ†é’Ÿï¼‰ |
| `HERMES_NOUS_TIMEOUT_SECONDS` | Nous å‡­è¯/token æµç¨‹çš„ HTTP è¶…æ—¶ |
| `HERMES_DUMP_REQUESTS` | å°† API è¯·æ±‚è½½è·è½¬å‚¨åˆ°æ—¥å¿—æ–‡ä»¶ï¼ˆ`true`/`false`ï¼‰ |
| `HERMES_PREFILL_MESSAGES_FILE` | åŒ…å«åœ¨ API è°ƒç”¨æ—¶æ³¨å…¥çš„ä¸´æ—¶é¢„å¡«æ¶ˆæ¯çš„ JSON æ–‡ä»¶è·¯å¾„ |
| `HERMES_TIMEZONE` | IANA æ—¶åŒºè¦†ç›–ï¼ˆä¾‹å¦‚ `America/New_York`ï¼‰ |

## å·¥å…· API

| å˜é‡ | æè¿° |
|----------|-------------|
| `PARALLEL_API_KEY` | AI åŸç”Ÿç½‘ç»œæœç´¢ï¼ˆ[parallel.ai](https://parallel.ai/)ï¼‰ |
| `FIRECRAWL_API_KEY` | ç½‘é¡µæŠ“å–å’Œäº‘æµè§ˆå™¨ï¼ˆ[firecrawl.dev](https://firecrawl.dev/)ï¼‰ |
| `FIRECRAWL_API_URL` | è‡ªæ‰˜ç®¡å®ä¾‹çš„è‡ªå®šä¹‰ Firecrawl API ç«¯ç‚¹ï¼ˆå¯é€‰ï¼‰ |
| `TAVILY_API_KEY` | Tavily API å¯†é’¥ï¼Œç”¨äº AI åŸç”Ÿç½‘ç»œæœç´¢ã€æå–å’Œçˆ¬å–ï¼ˆ[app.tavily.com](https://app.tavily.com/home)ï¼‰ |
| `SEARXNG_URL` | å…è´¹è‡ªæ‰˜ç®¡ç½‘ç»œæœç´¢çš„ SearXNG å®ä¾‹ URLâ€”â€”æ— éœ€ API å¯†é’¥ï¼ˆ[searxng.github.io](https://searxng.github.io/searxng/)ï¼‰ |
| `TAVILY_BASE_URL` | è¦†ç›– Tavily API ç«¯ç‚¹ã€‚é€‚ç”¨äºä¼ä¸šä»£ç†å’Œè‡ªæ‰˜ç®¡ Tavily å…¼å®¹æœç´¢åç«¯ã€‚ä¸ `GROQ_BASE_URL` æ¨¡å¼ç›¸åŒã€‚ |
| `EXA_API_KEY` | Exa API å¯†é’¥ï¼Œç”¨äº AI åŸç”Ÿç½‘ç»œæœç´¢å’Œå†…å®¹è·å–ï¼ˆ[exa.ai](https://exa.ai/)ï¼‰ |
| `BROWSERBASE_API_KEY` | æµè§ˆå™¨è‡ªåŠ¨åŒ–ï¼ˆ[browserbase.com](https://browserbase.com/)ï¼‰ |
| `BROWSERBASE_PROJECT_ID` | Browserbase é¡¹ç›® ID |
| `BROWSER_USE_API_KEY` | Browser Use äº‘æµè§ˆå™¨ API å¯†é’¥ï¼ˆ[browser-use.com](https://browser-use.com/)ï¼‰ |
| `FIRECRAWL_BROWSER_TTL` | Firecrawl æµè§ˆå™¨ä¼šè¯ TTLï¼ˆç§’ï¼Œé»˜è®¤ï¼š300ï¼‰ |
| `BROWSER_CDP_URL` | æœ¬åœ°æµè§ˆå™¨çš„ Chrome DevTools Protocolï¼ˆCDPï¼‰URLï¼ˆé€šè¿‡ `/browser connect` è®¾ç½®ï¼Œä¾‹å¦‚ `ws://localhost:9222`ï¼‰ |
| `CAMOFOX_URL` | Camofox æœ¬åœ°åæ£€æµ‹æµè§ˆå™¨ URLï¼ˆé»˜è®¤ï¼š`http://localhost:9377`ï¼‰ |
| `CAMOFOX_USER_ID` | å¯é€‰çš„å¤–éƒ¨ç®¡ç† Camofox ç”¨æˆ· IDï¼Œç”¨äºå…±äº«å¯è§ä¼šè¯ |
| `CAMOFOX_SESSION_KEY` | ä¸º `CAMOFOX_USER_ID` åˆ›å»ºæ ‡ç­¾é¡µæ—¶ä½¿ç”¨çš„å¯é€‰ Camofox ä¼šè¯å¯†é’¥ |
| `CAMOFOX_ADOPT_EXISTING_TAB` | è®¾ä¸º `true` å¯åœ¨åˆ›å»ºæ–°æ ‡ç­¾é¡µå‰å¤ç”¨ç°æœ‰ Camofox æ ‡ç­¾é¡µ |
| `BROWSER_INACTIVITY_TIMEOUT` | æµè§ˆå™¨ä¼šè¯ä¸æ´»åŠ¨è¶…æ—¶ï¼ˆç§’ï¼‰ |
| `AGENT_BROWSER_ARGS` | é¢å¤–çš„ Chromium å¯åŠ¨æ ‡å¿—ï¼ˆé€—å·æˆ–æ¢è¡Œåˆ†éš”ï¼‰ã€‚ä»¥ root èº«ä»½è¿è¡Œæˆ–åœ¨ AppArmor é™åˆ¶çš„éç‰¹æƒç”¨æˆ·å‘½åç©ºé—´ï¼ˆUbuntu 23.10+ã€DGX Sparkã€è®¸å¤šå®¹å™¨é•œåƒï¼‰ä¸­è¿è¡Œæ—¶ï¼ŒHermes è‡ªåŠ¨æ³¨å…¥ `--no-sandbox,--disable-dev-shm-usage`ï¼›ä»…åœ¨éœ€è¦è¦†ç›–æˆ–æ·»åŠ å…¶ä»–æ ‡å¿—æ—¶æ‰‹åŠ¨è®¾ç½®ã€‚ |
| `FAL_KEY` | å›¾åƒç”Ÿæˆï¼ˆ[fal.ai](https://fal.ai/)ï¼‰ |
| `GROQ_API_KEY` | Groq Whisper STT API å¯†é’¥ï¼ˆ[groq.com](https://groq.com/)ï¼‰ |
| `ELEVENLABS_API_KEY` | ElevenLabs é«˜çº§ TTS è¯­éŸ³ï¼ˆ[elevenlabs.io](https://elevenlabs.io/)ï¼‰ |
| `STT_GROQ_MODEL` | è¦†ç›– Groq STT æ¨¡å‹ï¼ˆé»˜è®¤ï¼š`whisper-large-v3-turbo`ï¼‰ |
| `GROQ_BASE_URL` | è¦†ç›– Groq OpenAI å…¼å®¹ STT ç«¯ç‚¹ |
| `STT_OPENAI_MODEL` | è¦†ç›– OpenAI STT æ¨¡å‹ï¼ˆé»˜è®¤ï¼š`whisper-1`ï¼‰ |
| `STT_OPENAI_BASE_URL` | è¦†ç›– OpenAI å…¼å®¹ STT ç«¯ç‚¹ |
| `GITHUB_TOKEN` | Skills Hub çš„ GitHub tokenï¼ˆæ›´é«˜ API é€Ÿç‡é™åˆ¶ï¼ŒæŠ€èƒ½å‘å¸ƒï¼‰ |
| `HONCHO_API_KEY` | è·¨ä¼šè¯ç”¨æˆ·å»ºæ¨¡ï¼ˆ[honcho.dev](https://honcho.dev/)ï¼‰ |
| `HONCHO_BASE_URL` | è‡ªæ‰˜ç®¡ Honcho å®ä¾‹çš„ base URLï¼ˆé»˜è®¤ï¼šHoncho äº‘ï¼‰ã€‚æœ¬åœ°å®ä¾‹æ— éœ€ API å¯†é’¥ |
| `HINDSIGHT_TIMEOUT` | Hindsight å†…å­˜æä¾›å•† API è°ƒç”¨è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`60`ï¼‰ã€‚å¦‚æœ Hindsight å®ä¾‹åœ¨ `/sync` æˆ– `on_session_switch` æœŸé—´å“åº”ç¼“æ…¢å¹¶å‡ºç°è¶…æ—¶ï¼Œè¯·å¢å¤§æ­¤å€¼ï¼Œå¹¶æ£€æŸ¥ `errors.log`ã€‚ |
| `SUPERMEMORY_API_KEY` | æ”¯æŒ profile å¬å›å’Œä¼šè¯æ‘„å–çš„è¯­ä¹‰é•¿æœŸè®°å¿†ï¼ˆ[supermemory.ai](https://supermemory.ai)ï¼‰ |
| `DAYTONA_API_KEY` | Daytona äº‘æ²™ç®±ï¼ˆ[daytona.io](https://daytona.io/)ï¼‰ |

### Langfuse å¯è§‚æµ‹æ€§

å†…ç½® [`observability/langfuse`](/user-guide/features/built-in-plugins#observabilitylangfuse) æ’ä»¶çš„ç¯å¢ƒå˜é‡ã€‚åœ¨ `~/.hermes/.env` ä¸­è®¾ç½®ã€‚åœ¨è¿™äº›å˜é‡ç”Ÿæ•ˆä¹‹å‰ï¼Œè¿˜å¿…é¡»å¯ç”¨è¯¥æ’ä»¶ï¼ˆ`hermes plugins enable observability/langfuse`ï¼Œæˆ–åœ¨ `hermes plugins` ä¸­å‹¾é€‰ï¼‰ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Langfuse é¡¹ç›®å…¬é’¥ï¼ˆ`pk-lf-...`ï¼‰ã€‚å¿…å¡«ã€‚ |
| `HERMES_LANGFUSE_SECRET_KEY` | Langfuse é¡¹ç›®å¯†é’¥ï¼ˆ`sk-lf-...`ï¼‰ã€‚å¿…å¡«ã€‚ |
| `HERMES_LANGFUSE_BASE_URL` | Langfuse æœåŠ¡å™¨ URLï¼ˆé»˜è®¤ï¼š`https://cloud.langfuse.com`ï¼‰ã€‚è‡ªæ‰˜ç®¡æ—¶è®¾ç½®ã€‚ |
| `HERMES_LANGFUSE_ENV` | trace ä¸Šçš„ç¯å¢ƒæ ‡ç­¾ï¼ˆ`production`ã€`staging` ç­‰ï¼‰ |
| `HERMES_LANGFUSE_RELEASE` | trace ä¸Šçš„å‘å¸ƒ/ç‰ˆæœ¬æ ‡ç­¾ |
| `HERMES_LANGFUSE_SAMPLE_RATE` | SDK é‡‡æ ·ç‡ 0.0â€“1.0ï¼ˆé»˜è®¤ï¼š`1.0`ï¼‰ |
| `HERMES_LANGFUSE_MAX_CHARS` | åºåˆ—åŒ–è½½è·çš„æ¯å­—æ®µæˆªæ–­é•¿åº¦ï¼ˆé»˜è®¤ï¼š`12000`ï¼‰ |
| `HERMES_LANGFUSE_DEBUG` | `true` å¯å°†è¯¦ç»†æ’ä»¶æ—¥å¿—è¾“å‡ºåˆ° `agent.log` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | æ ‡å‡† Langfuse SDK å˜é‡åã€‚å½“å¯¹åº”çš„ `HERMES_LANGFUSE_*` æœªè®¾ç½®æ—¶ä½œä¸ºå›é€€ã€‚ |

### Nous Tool Gateway

è¿™äº›å˜é‡ä¸ºä»˜è´¹ Nous è®¢é˜…è€…æˆ–è‡ªæ‰˜ç®¡ gateway éƒ¨ç½²é…ç½® [Tool Gateway](/user-guide/features/tool-gateway)ã€‚å¤§å¤šæ•°ç”¨æˆ·æ— éœ€è®¾ç½®â€”â€”gateway é€šè¿‡ `hermes model` æˆ– `hermes tools` è‡ªåŠ¨é…ç½®ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | Tool Gateway è·¯ç”±çš„åŸºç¡€åŸŸåï¼ˆé»˜è®¤ï¼š`nousresearch.com`ï¼‰ |
| `TOOL_GATEWAY_SCHEME` | gateway URL çš„ HTTP æˆ– HTTPS åè®®ï¼ˆé»˜è®¤ï¼š`https`ï¼‰ |
| `TOOL_GATEWAY_USER_TOKEN` | Tool Gateway çš„è®¤è¯ tokenï¼ˆé€šå¸¸ç”± Nous è®¤è¯è‡ªåŠ¨å¡«å……ï¼‰ |
| `FIRECRAWL_GATEWAY_URL` | ä¸“é—¨è¦†ç›– Firecrawl gateway ç«¯ç‚¹çš„ URL |

## ç»ˆç«¯åç«¯

| å˜é‡ | æè¿° |
|----------|-------------|
| `TERMINAL_ENV` | åç«¯ï¼š`local`ã€`docker`ã€`ssh`ã€`singularity`ã€`modal`ã€`daytona` |
| `HERMES_DOCKER_BINARY` | è¦†ç›– Hermes è°ƒç”¨çš„å®¹å™¨äºŒè¿›åˆ¶ï¼ˆä¾‹å¦‚ `podman`ã€`/usr/local/bin/docker`ï¼‰ã€‚æœªè®¾ç½®æ—¶ï¼ŒHermes è‡ªåŠ¨åœ¨ `PATH` ä¸Šå‘ç° `docker` æˆ– `podman`ã€‚å½“ä¸¤è€…éƒ½å·²å®‰è£…ä¸”éœ€è¦éé»˜è®¤é€‰é¡¹ï¼Œæˆ–äºŒè¿›åˆ¶ä¸åœ¨ `PATH` ä¸­æ—¶ä½¿ç”¨ã€‚ |
| `TERMINAL_DOCKER_IMAGE` | Docker é•œåƒï¼ˆé»˜è®¤ï¼š`nikolaik/python-nodejs:python3.11-nodejs20`ï¼‰ |
| `TERMINAL_DOCKER_FORWARD_ENV` | æ˜¾å¼è½¬å‘åˆ° Docker ç»ˆç«¯ä¼šè¯çš„ç¯å¢ƒå˜é‡å JSON æ•°ç»„ã€‚æ³¨æ„ï¼šæŠ€èƒ½å£°æ˜çš„ `required_environment_variables` ä¼šè‡ªåŠ¨è½¬å‘â€”â€”ä»…å¯¹æœªè¢«ä»»ä½•æŠ€èƒ½å£°æ˜çš„å˜é‡ä½¿ç”¨æ­¤é¡¹ã€‚ |
| `TERMINAL_DOCKER_VOLUMES` | é¢å¤–çš„ Docker å·æŒ‚è½½ï¼ˆé€—å·åˆ†éš”çš„ `host:container` å¯¹ï¼‰ |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | é«˜çº§é€‰é¡¹ï¼šå°†å¯åŠ¨æ—¶çš„ cwd æŒ‚è½½åˆ° Docker `/workspace`ï¼ˆ`true`/`false`ï¼Œé»˜è®¤ï¼š`false`ï¼‰ |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity é•œåƒæˆ– `.sif` è·¯å¾„ |
| `TERMINAL_MODAL_IMAGE` | Modal å®¹å™¨é•œåƒ |
| `TERMINAL_DAYTONA_IMAGE` | Daytona æ²™ç®±é•œåƒ |
| `TERMINAL_TIMEOUT` | å‘½ä»¤è¶…æ—¶ï¼ˆç§’ï¼‰ |
| `TERMINAL_LIFETIME_SECONDS` | ç»ˆç«¯ä¼šè¯æœ€å¤§ç”Ÿå‘½å‘¨æœŸï¼ˆç§’ï¼‰ |
| `TERMINAL_CWD` | ç»ˆç«¯ä¼šè¯çš„å·¥ä½œç›®å½•ï¼ˆä»… gateway/cronï¼›CLI ä½¿ç”¨å¯åŠ¨ç›®å½•ï¼‰ |
| `SUDO_PASSWORD` | æ— éœ€äº¤äº’æç¤ºå³å¯ä½¿ç”¨ sudo |

å¯¹äºäº‘æ²™ç®±åç«¯ï¼ŒæŒä¹…åŒ–ä»¥æ–‡ä»¶ç³»ç»Ÿä¸ºå¯¼å‘ã€‚`TERMINAL_LIFETIME_SECONDS` æ§åˆ¶ Hermes ä½•æ—¶æ¸…ç†ç©ºé—²ç»ˆç«¯ä¼šè¯ï¼Œåç»­æ¢å¤å¯èƒ½ä¼šé‡æ–°åˆ›å»ºæ²™ç®±è€Œéä¿æŒç›¸åŒçš„æ´»è·ƒè¿›ç¨‹ã€‚

## SSH åç«¯

| å˜é‡ | æè¿° |
|----------|-------------|
| `TERMINAL_SSH_HOST` | è¿œç¨‹æœåŠ¡å™¨ä¸»æœºå |
| `TERMINAL_SSH_USER` | SSH ç”¨æˆ·å |
| `TERMINAL_SSH_PORT` | SSH ç«¯å£ï¼ˆé»˜è®¤ï¼š22ï¼‰ |
| `TERMINAL_SSH_KEY` | ç§é’¥è·¯å¾„ |
| `TERMINAL_SSH_PERSISTENT` | è¦†ç›– SSH çš„æŒä¹… shellï¼ˆé»˜è®¤ï¼šè·Ÿéš `TERMINAL_PERSISTENT_SHELL`ï¼‰ |

## å®¹å™¨èµ„æºï¼ˆDockerã€Singularityã€Modalã€Daytonaï¼‰

| å˜é‡ | æè¿° |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPU æ ¸å¿ƒæ•°ï¼ˆé»˜è®¤ï¼š1ï¼‰ |
| `TERMINAL_CONTAINER_MEMORY` | å†…å­˜ï¼ˆMBï¼Œé»˜è®¤ï¼š5120ï¼‰ |
| `TERMINAL_CONTAINER_DISK` | ç£ç›˜ï¼ˆMBï¼Œé»˜è®¤ï¼š51200ï¼‰ |
| `TERMINAL_CONTAINER_PERSISTENT` | è·¨ä¼šè¯æŒä¹…åŒ–å®¹å™¨æ–‡ä»¶ç³»ç»Ÿï¼ˆé»˜è®¤ï¼š`true`ï¼‰ |
| `TERMINAL_SANDBOX_DIR` | å·¥ä½œåŒºå’Œ overlay çš„å®¿ä¸»æœºç›®å½•ï¼ˆé»˜è®¤ï¼š`~/.hermes/sandboxes/`ï¼‰ |

## æŒä¹… Shell

| å˜é‡ | æè¿° |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | ä¸ºéæœ¬åœ°åç«¯å¯ç”¨æŒä¹… shellï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚ä¹Ÿå¯é€šè¿‡ config.yaml ä¸­çš„ `terminal.persistent_shell` è®¾ç½® |
| `TERMINAL_LOCAL_PERSISTENT` | ä¸ºæœ¬åœ°åç«¯å¯ç”¨æŒä¹… shellï¼ˆé»˜è®¤ï¼š`false`ï¼‰ |
| `TERMINAL_SSH_PERSISTENT` | è¦†ç›– SSH åç«¯çš„æŒä¹… shellï¼ˆé»˜è®¤ï¼šè·Ÿéš `TERMINAL_PERSISTENT_SHELL`ï¼‰ |

## æ¶ˆæ¯å¹³å°

| å˜é‡ | æè¿° |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot tokenï¼ˆæ¥è‡ª @BotFatherï¼‰ |
| `TELEGRAM_ALLOWED_USERS` | å…è®¸ä½¿ç”¨ bot çš„é€—å·åˆ†éš”ç”¨æˆ· IDï¼ˆé€‚ç”¨äºç§èŠã€ç¾¤ç»„å’Œè®ºå›ï¼‰ |
| `TELEGRAM_GROUP_ALLOWED_USERS` | ä»…åœ¨ç¾¤ç»„/è®ºå›ä¸­æˆæƒçš„é€—å·åˆ†éš”å‘é€è€…ç”¨æˆ· IDï¼ˆ**ä¸**æˆäºˆç§èŠæƒé™ï¼‰ã€‚ä»¥ `-` å¼€å¤´çš„èŠå¤© ID å½¢å¼å€¼ä»ä½œä¸ºèŠå¤© ID å¤„ç†ï¼Œä»¥å‘åå…¼å®¹ #17686 ä¹‹å‰çš„é…ç½®ï¼Œå¹¶æ˜¾ç¤ºå¼ƒç”¨è­¦å‘Šã€‚ |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | é€—å·åˆ†éš”çš„ç¾¤ç»„/è®ºå›èŠå¤© IDï¼›ä»»æ„æˆå‘˜å‡å¯æˆæƒ |
| `TELEGRAM_HOME_CHANNEL` | cron æŠ•é€’çš„é»˜è®¤ Telegram èŠå¤©/é¢‘é“ |
| `TELEGRAM_HOME_CHANNEL_NAME` | Telegram ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `TELEGRAM_CRON_THREAD_ID` | æ¥æ”¶ cron æŠ•é€’çš„è®ºå›è¯é¢˜ IDï¼›ä»…å¯¹ cron è¦†ç›– `TELEGRAM_HOME_CHANNEL_THREAD_ID`ã€‚åœ¨è¯é¢˜æ¨¡å¼ä¸‹ä½¿ç”¨ï¼Œä½¿ cron æ¶ˆæ¯çš„å›å¤å¼€å¯æ–°ä¼šè¯è€Œéè¿›å…¥ç³»ç»Ÿå¤§å…ï¼ˆ#24409ï¼‰ã€‚ |
| `TELEGRAM_WEBHOOK_URL` | webhook æ¨¡å¼çš„å…¬å…± HTTPS URLï¼ˆå¯ç”¨ webhook è€Œéè½®è¯¢ï¼‰ |
| `TELEGRAM_WEBHOOK_PORT` | webhook æœåŠ¡å™¨æœ¬åœ°ç›‘å¬ç«¯å£ï¼ˆé»˜è®¤ï¼š`8443`ï¼‰ |
| `TELEGRAM_WEBHOOK_SECRET` | Telegram åœ¨æ¯æ¬¡æ›´æ–°ä¸­å›ä¼ çš„å¯†é’¥ tokenï¼Œç”¨äºéªŒè¯ã€‚**è®¾ç½® `TELEGRAM_WEBHOOK_URL` æ—¶å¿…å¡«**â€”â€”æœªè®¾ç½®æ—¶ gateway æ‹’ç»å¯åŠ¨ï¼ˆGHSA-3vpc-7q5r-276hï¼‰ã€‚ä½¿ç”¨ `openssl rand -hex 32` ç”Ÿæˆã€‚ |
| `TELEGRAM_REACTIONS` | å¤„ç†æœŸé—´åœ¨æ¶ˆæ¯ä¸Šå¯ç”¨ emoji ååº”ï¼ˆé»˜è®¤ï¼š`false`ï¼‰ |
| `TELEGRAM_REQUIRE_MENTION` | åœ¨ Telegram ç¾¤ç»„ä¸­å“åº”å‰è¦æ±‚æ˜¾å¼è§¦å‘ã€‚ç­‰åŒäº `config.yaml` ä¸­çš„ `telegram.require_mention`ã€‚ |
| `TELEGRAM_MENTION_PATTERNS` | å¯ç”¨ Telegram ç¾¤ç»„ mention é—¨æ§æ—¶æ¥å—çš„æ­£åˆ™å”¤é†’è¯æ¨¡å¼ï¼ŒJSON æ•°ç»„ã€æ¢è¡Œåˆ†éš”åˆ—è¡¨æˆ–é€—å·åˆ†éš”åˆ—è¡¨ã€‚ç­‰åŒäº `telegram.mention_patterns`ã€‚ |
| `TELEGRAM_EXCLUSIVE_BOT_MENTIONS` | å¯ç”¨åï¼ŒTelegram ç¾¤ç»„ä¸­çš„æ˜¾å¼ `@...bot` mention ä»…è·¯ç”±åˆ°è¢« mention çš„ bot ç”¨æˆ·åï¼Œç„¶åå†æ‰§è¡Œå›å¤æˆ–å”¤é†’è¯å›é€€ã€‚é»˜è®¤ï¼š`true`ã€‚ç­‰åŒäº `telegram.exclusive_bot_mentions`ã€‚ |
| `TELEGRAM_REPLY_TO_MODE` | å›å¤å¼•ç”¨è¡Œä¸ºï¼š`off`ã€`first`ï¼ˆé»˜è®¤ï¼‰æˆ– `all`ã€‚ä¸ Discord æ¨¡å¼ä¸€è‡´ã€‚ |
| `TELEGRAM_IGNORED_THREADS` | bot æ°¸ä¸å“åº”çš„é€—å·åˆ†éš” Telegram è®ºå›è¯é¢˜/çº¿ç¨‹ ID |
| `TELEGRAM_PROXY` | Telegram è¿æ¥çš„ä»£ç† URLâ€”â€”è¦†ç›– `HTTPS_PROXY`ã€‚æ”¯æŒ `http://`ã€`https://`ã€`socks5://` |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USERS` | å…è®¸ä½¿ç”¨ bot çš„é€—å·åˆ†éš” Discord ç”¨æˆ· ID |
| `DISCORD_ALLOWED_ROLES` | å…è®¸ä½¿ç”¨ bot çš„é€—å·åˆ†éš” Discord è§’è‰² IDï¼ˆä¸ `DISCORD_ALLOWED_USERS` å– ORï¼‰ã€‚è‡ªåŠ¨å¯ç”¨ Members intentã€‚é€‚ç”¨äºç®¡ç†å›¢é˜Ÿé¢‘ç¹å˜åŠ¨çš„åœºæ™¯â€”â€”è§’è‰²æˆæƒè‡ªåŠ¨ä¼ æ’­ã€‚ |
| `DISCORD_ALLOWED_CHANNELS` | é€—å·åˆ†éš”çš„ Discord é¢‘é“ IDã€‚è®¾ç½®åï¼Œbot ä»…åœ¨è¿™äº›é¢‘é“ï¼ˆä»¥åŠå…è®¸çš„ç§èŠï¼‰ä¸­å“åº”ã€‚è¦†ç›– `config.yaml` ä¸­çš„ `discord.allowed_channels`ã€‚ |
| `DISCORD_PROXY` | Discord è¿æ¥çš„ä»£ç† URLâ€”â€”è¦†ç›– `HTTPS_PROXY`ã€‚æ”¯æŒ `http://`ã€`https://`ã€`socks5://` |
| `DISCORD_HOME_CHANNEL` | cron æŠ•é€’çš„é»˜è®¤ Discord é¢‘é“ |
| `DISCORD_HOME_CHANNEL_NAME` | Discord ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `DISCORD_COMMAND_SYNC_POLICY` | Discord æ–œæ å‘½ä»¤å¯åŠ¨åŒæ­¥ç­–ç•¥ï¼š`safe`ï¼ˆå·®å¼‚å¯¹æ¯”å¹¶åè°ƒï¼‰ã€`bulk`ï¼ˆæ—§ç‰ˆ `tree.sync()`ï¼‰æˆ– `off` |
| `DISCORD_REQUIRE_MENTION` | åœ¨æœåŠ¡å™¨é¢‘é“ä¸­å“åº”å‰è¦æ±‚ @mention |
| `DISCORD_FREE_RESPONSE_CHANNELS` | ä¸éœ€è¦ mention çš„é€—å·åˆ†éš”é¢‘é“ ID |
| `DISCORD_AUTO_THREAD` | æ”¯æŒæ—¶è‡ªåŠ¨å°†é•¿å›å¤è½¬ä¸ºçº¿ç¨‹ |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | è®¾ä¸º `true` æ—¶æ¥å—ä»»æ„æ–‡ä»¶ç±»å‹çš„é™„ä»¶ï¼ˆä¸ä»…é™äºå†…ç½®çš„ PDF/æ–‡æœ¬/zip/office ç™½åå•ï¼‰ã€‚æœªçŸ¥ç±»å‹è¢«ç¼“å­˜å¹¶ä»¥æœ¬åœ°è·¯å¾„å½¢å¼æä¾›ç»™ agentï¼Œä¾›å…¶é€šè¿‡ `terminal`/`read_file`/`ffprobe` æ£€æŸ¥ã€‚é»˜è®¤ `false`ã€‚ |
| `DISCORD_MAX_ATTACHMENT_BYTES` | gateway ç¼“å­˜çš„æ¯ä¸ªé™„ä»¶æœ€å¤§å­—èŠ‚æ•°ã€‚é»˜è®¤ `33554432`ï¼ˆ32 MiBï¼‰ã€‚è®¾ä¸º `0` è¡¨ç¤ºæ— ä¸Šé™ï¼ˆé™„ä»¶åœ¨å†™å…¥æ—¶ä¿å­˜åœ¨å†…å­˜ä¸­ï¼‰ã€‚ |
| `DISCORD_REACTIONS` | å¤„ç†æœŸé—´åœ¨æ¶ˆæ¯ä¸Šå¯ç”¨ emoji ååº”ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ |
| `DISCORD_IGNORED_CHANNELS` | bot æ°¸ä¸å“åº”çš„é€—å·åˆ†éš”é¢‘é“ ID |
| `DISCORD_NO_THREAD_CHANNELS` | bot ä¸è‡ªåŠ¨åˆ›å»ºçº¿ç¨‹çš„é€—å·åˆ†éš”é¢‘é“ ID |
| `DISCORD_REPLY_TO_MODE` | å›å¤å¼•ç”¨è¡Œä¸ºï¼š`off`ã€`first`ï¼ˆé»˜è®¤ï¼‰æˆ– `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | å…è®¸ bot ping `@everyone`/`@here`ï¼ˆé»˜è®¤ï¼š`false`ï¼‰ã€‚å‚è§ [Mention æ§åˆ¶](../user-guide/messaging/discord.md#mention-control)ã€‚ |
| `DISCORD_ALLOW_MENTION_ROLES` | å…è®¸ bot ping `@role` mentionï¼ˆé»˜è®¤ï¼š`false`ï¼‰ã€‚ |
| `DISCORD_ALLOW_MENTION_USERS` | å…è®¸ bot ping å•ä¸ª `@user` mentionï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚ |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | å›å¤æ¶ˆæ¯æ—¶ ping åŸä½œè€…ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚ |
| `SLACK_BOT_TOKEN` | Slack bot tokenï¼ˆ`xoxb-...`ï¼‰ |
| `SLACK_APP_TOKEN` | Slack åº”ç”¨çº§ tokenï¼ˆ`xapp-...`ï¼ŒSocket Mode å¿…éœ€ï¼‰ |
| `SLACK_ALLOWED_USERS` | é€—å·åˆ†éš”çš„ Slack ç”¨æˆ· ID |
| `SLACK_HOME_CHANNEL` | cron æŠ•é€’çš„é»˜è®¤ Slack é¢‘é“ |
| `SLACK_HOME_CHANNEL_NAME` | Slack ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `GOOGLE_CHAT_PROJECT_ID` | æ‰˜ç®¡ Pub/Sub è¯é¢˜çš„ GCP é¡¹ç›®ï¼ˆå›é€€åˆ° `GOOGLE_CLOUD_PROJECT`ï¼‰ |
| `GOOGLE_CHAT_SUBSCRIPTION_NAME` | å®Œæ•´ Pub/Sub è®¢é˜…è·¯å¾„ï¼Œ`projects/{proj}/subscriptions/{sub}`ï¼ˆæ—§ç‰ˆåˆ«åï¼š`GOOGLE_CHAT_SUBSCRIPTION`ï¼‰ |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` | Service Account JSON æ–‡ä»¶è·¯å¾„ï¼Œæˆ–å†…è” JSONï¼ˆå›é€€åˆ° `GOOGLE_APPLICATION_CREDENTIALS`ï¼‰ |
| `GOOGLE_CHAT_ALLOWED_USERS` | å…è®¸ä¸ bot èŠå¤©çš„é€—å·åˆ†éš”ç”¨æˆ·é‚®ç®± |
| `GOOGLE_CHAT_ALLOW_ALL_USERS` | å…è®¸ä»»æ„ Google Chat ç”¨æˆ·è§¦å‘ botï¼ˆä»…ç”¨äºå¼€å‘ï¼‰ |
| `GOOGLE_CHAT_HOME_CHANNEL` | cron æŠ•é€’çš„é»˜è®¤ç©ºé—´ï¼ˆä¾‹å¦‚ `spaces/AAAA...`ï¼‰ |
| `GOOGLE_CHAT_HOME_CHANNEL_NAME` | Google Chat ä¸»ç©ºé—´çš„æ˜¾ç¤ºåç§° |
| `GOOGLE_CHAT_MAX_MESSAGES` | Pub/Sub FlowControl æœ€å¤§åœ¨é€”æ¶ˆæ¯æ•°ï¼ˆé»˜è®¤ï¼š`1`ï¼‰ |
| `GOOGLE_CHAT_MAX_BYTES` | Pub/Sub FlowControl æœ€å¤§åœ¨é€”å­—èŠ‚æ•°ï¼ˆé»˜è®¤ï¼š`16777216`ï¼Œ16 MiBï¼‰ |
| `GOOGLE_CHAT_BOOTSTRAP_SPACES` | å¯åŠ¨æ—¶æ¢æµ‹ä»¥è§£æ bot è‡ªèº« `users/{id}` çš„é€—å·åˆ†éš”é¢å¤–ç©ºé—´ ID |
| `GOOGLE_CHAT_DEBUG_RAW` | è®¾ç½®ä»»æ„å€¼å¯åœ¨ DEBUG çº§åˆ«è®°å½•è„±æ•çš„ Pub/Sub ä¿¡å°ï¼ˆä»…ç”¨äºè°ƒè¯•ï¼‰ |
| `WHATSAPP_ENABLED` | å¯ç”¨ WhatsApp æ¡¥æ¥ï¼ˆ`true`/`false`ï¼‰ |
| `WHATSAPP_MODE` | `bot`ï¼ˆç‹¬ç«‹å·ç ï¼‰æˆ– `self-chat`ï¼ˆç»™è‡ªå·±å‘æ¶ˆæ¯ï¼‰ |
| `WHATSAPP_ALLOWED_USERS` | é€—å·åˆ†éš”çš„æ‰‹æœºå·ç ï¼ˆå«å›½å®¶ä»£ç ï¼Œä¸å« `+`ï¼‰ï¼Œæˆ– `*` å…è®¸æ‰€æœ‰å‘é€è€… |
| `WHATSAPP_ALLOW_ALL_USERS` | æ— éœ€ç™½åå•å…è®¸æ‰€æœ‰ WhatsApp å‘é€è€…ï¼ˆ`true`/`false`ï¼‰ |
| `WHATSAPP_DEBUG` | åœ¨æ¡¥æ¥ä¸­è®°å½•åŸå§‹æ¶ˆæ¯äº‹ä»¶ä»¥ä¾›æ’æŸ¥ï¼ˆ`true`/`false`ï¼‰ |
| `SIGNAL_HTTP_URL` | signal-cli å®ˆæŠ¤è¿›ç¨‹ HTTP ç«¯ç‚¹ï¼ˆä¾‹å¦‚ `http://127.0.0.1:8080`ï¼‰ |
| `SIGNAL_ACCOUNT` | E.164 æ ¼å¼çš„ bot æ‰‹æœºå·ç  |
| `SIGNAL_ALLOWED_USERS` | é€—å·åˆ†éš”çš„ E.164 æ‰‹æœºå·ç æˆ– UUID |
| `SIGNAL_GROUP_ALLOWED_USERS` | é€—å·åˆ†éš”çš„ç¾¤ç»„ IDï¼Œæˆ– `*` è¡¨ç¤ºæ‰€æœ‰ç¾¤ç»„ |
| `SIGNAL_HOME_CHANNEL_NAME` | Signal ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `SIGNAL_IGNORE_STORIES` | å¿½ç•¥ Signal æ•…äº‹/çŠ¶æ€æ›´æ–° |
| `SIGNAL_ALLOW_ALL_USERS` | æ— éœ€ç™½åå•å…è®¸æ‰€æœ‰ Signal ç”¨æˆ· |
| `TWILIO_ACCOUNT_SID` | Twilio Account SIDï¼ˆä¸ç”µè¯æŠ€èƒ½å…±äº«ï¼‰ |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Tokenï¼ˆä¸ç”µè¯æŠ€èƒ½å…±äº«ï¼›ä¹Ÿç”¨äº webhook ç­¾åéªŒè¯ï¼‰ |
| `TWILIO_PHONE_NUMBER` | E.164 æ ¼å¼çš„ Twilio æ‰‹æœºå·ç ï¼ˆä¸ç”µè¯æŠ€èƒ½å…±äº«ï¼‰ |
| `SMS_WEBHOOK_URL` | Twilio ç­¾åéªŒè¯çš„å…¬å…± URLâ€”â€”å¿…é¡»ä¸ Twilio Console ä¸­çš„ webhook URL ä¸€è‡´ï¼ˆå¿…å¡«ï¼‰ |
| `SMS_WEBHOOK_PORT` | å…¥ç«™ SMS çš„ webhook ç›‘å¬ç«¯å£ï¼ˆé»˜è®¤ï¼š`8080`ï¼‰ |
| `SMS_WEBHOOK_HOST` | webhook ç»‘å®šåœ°å€ï¼ˆé»˜è®¤ï¼š`0.0.0.0`ï¼‰ |
| `SMS_INSECURE_NO_SIGNATURE` | è®¾ä¸º `true` å¯ç¦ç”¨ Twilio ç­¾åéªŒè¯ï¼ˆä»…ç”¨äºæœ¬åœ°å¼€å‘â€”â€”ä¸é€‚ç”¨äºç”Ÿäº§ç¯å¢ƒï¼‰ |
| `SMS_ALLOWED_USERS` | å…è®¸èŠå¤©çš„é€—å·åˆ†éš” E.164 æ‰‹æœºå·ç  |
| `SMS_ALLOW_ALL_USERS` | æ— éœ€ç™½åå•å…è®¸æ‰€æœ‰ SMS å‘é€è€… |
| `SMS_HOME_CHANNEL` | cron ä»»åŠ¡/é€šçŸ¥æŠ•é€’çš„æ‰‹æœºå·ç  |
| `SMS_HOME_CHANNEL_NAME` | SMS ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `EMAIL_ADDRESS` | Email gateway é€‚é…å™¨çš„é‚®ç®±åœ°å€ |
| `EMAIL_PASSWORD` | é‚®ç®±è´¦æˆ·çš„å¯†ç æˆ–åº”ç”¨å¯†ç  |
| `EMAIL_IMAP_HOST` | é‚®ä»¶é€‚é…å™¨çš„ IMAP ä¸»æœºå |
| `EMAIL_IMAP_PORT` | IMAP ç«¯å£ |
| `EMAIL_SMTP_HOST` | é‚®ä»¶é€‚é…å™¨çš„ SMTP ä¸»æœºå |
| `EMAIL_SMTP_PORT` | SMTP ç«¯å£ |
| `EMAIL_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš”é‚®ç®±åœ°å€ |
| `EMAIL_HOME_ADDRESS` | ä¸»åŠ¨é‚®ä»¶æŠ•é€’çš„é»˜è®¤æ”¶ä»¶äºº |
| `EMAIL_HOME_ADDRESS_NAME` | é‚®ä»¶ä¸»ç›®æ ‡çš„æ˜¾ç¤ºåç§° |
| `EMAIL_POLL_INTERVAL` | é‚®ä»¶è½®è¯¢é—´éš”ï¼ˆç§’ï¼‰ |
| `EMAIL_ALLOW_ALL_USERS` | å…è®¸æ‰€æœ‰å…¥ç«™é‚®ä»¶å‘é€è€… |
| `DINGTALK_CLIENT_ID` | æ¥è‡ªå¼€å‘è€…é—¨æˆ·çš„é’‰é’‰ bot AppKeyï¼ˆ[open.dingtalk.com](https://open.dingtalk.com)ï¼‰ |
| `DINGTALK_CLIENT_SECRET` | æ¥è‡ªå¼€å‘è€…é—¨æˆ·çš„é’‰é’‰ bot AppSecret |
| `DINGTALK_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš”é’‰é’‰ç”¨æˆ· ID |
| `FEISHU_APP_ID` | æ¥è‡ª [open.feishu.cn](https://open.feishu.cn/) çš„é£ä¹¦/Lark bot App ID |
| `FEISHU_APP_SECRET` | é£ä¹¦/Lark bot App Secret |
| `FEISHU_DOMAIN` | `feishu`ï¼ˆä¸­å›½ï¼‰æˆ– `lark`ï¼ˆå›½é™…ï¼‰ã€‚é»˜è®¤ï¼š`feishu` |
| `FEISHU_CONNECTION_MODE` | `websocket`ï¼ˆæ¨èï¼‰æˆ– `webhook`ã€‚é»˜è®¤ï¼š`websocket` |
| `FEISHU_ENCRYPT_KEY` | webhook æ¨¡å¼çš„å¯é€‰åŠ å¯†å¯†é’¥ |
| `FEISHU_VERIFICATION_TOKEN` | webhook æ¨¡å¼çš„å¯é€‰éªŒè¯ token |
| `FEISHU_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš”é£ä¹¦ç”¨æˆ· ID |
| `FEISHU_ALLOW_BOTS` | `none`ï¼ˆé»˜è®¤ï¼‰/`mentions`/`all`â€”â€”æ¥å—æ¥è‡ªå…¶ä»– bot çš„å…¥ç«™æ¶ˆæ¯ã€‚å‚è§ [bot é—´æ¶ˆæ¯ä¼ é€’](../user-guide/messaging/feishu.md#bot-to-bot-messaging) |
| `FEISHU_REQUIRE_MENTION` | `true`ï¼ˆé»˜è®¤ï¼‰/`false`â€”â€”ç¾¤ç»„æ¶ˆæ¯æ˜¯å¦å¿…é¡» @mention botã€‚å¯é€šè¿‡ `group_rules.<chat_id>.require_mention` æŒ‰èŠå¤©è¦†ç›–ã€‚ |
| `FEISHU_HOME_CHANNEL` | cron æŠ•é€’å’Œé€šçŸ¥çš„é£ä¹¦èŠå¤© ID |
| `WECOM_BOT_ID` | æ¥è‡ªç®¡ç†æ§åˆ¶å°çš„ä¼ä¸šå¾®ä¿¡ AI Bot ID |
| `WECOM_SECRET` | ä¼ä¸šå¾®ä¿¡ AI Bot å¯†é’¥ |
| `WECOM_WEBSOCKET_URL` | è‡ªå®šä¹‰ WebSocket URLï¼ˆé»˜è®¤ï¼š`wss://openws.work.weixin.qq.com`ï¼‰ |
| `WECOM_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš”ä¼ä¸šå¾®ä¿¡ç”¨æˆ· ID |
| `WECOM_HOME_CHANNEL` | cron æŠ•é€’å’Œé€šçŸ¥çš„ä¼ä¸šå¾®ä¿¡èŠå¤© ID |
| `WECOM_CALLBACK_CORP_ID` | ä¼ä¸šå¾®ä¿¡å›è°ƒè‡ªå»ºåº”ç”¨çš„ä¼ä¸š Corp ID |
| `WECOM_CALLBACK_CORP_SECRET` | è‡ªå»ºåº”ç”¨çš„ä¼ä¸šå¯†é’¥ |
| `WECOM_CALLBACK_AGENT_ID` | è‡ªå»ºåº”ç”¨çš„ Agent ID |
| `WECOM_CALLBACK_TOKEN` | å›è°ƒéªŒè¯ token |
| `WECOM_CALLBACK_ENCODING_AES_KEY` | å›è°ƒåŠ å¯†çš„ AES å¯†é’¥ |
| `WECOM_CALLBACK_HOST` | å›è°ƒæœåŠ¡å™¨ç»‘å®šåœ°å€ï¼ˆé»˜è®¤ï¼š`0.0.0.0`ï¼‰ |
| `WECOM_CALLBACK_PORT` | å›è°ƒæœåŠ¡å™¨ç«¯å£ï¼ˆé»˜è®¤ï¼š`8645`ï¼‰ |
| `WECOM_CALLBACK_ALLOWED_USERS` | ç™½åå•çš„é€—å·åˆ†éš”ç”¨æˆ· ID |
| `WECOM_CALLBACK_ALLOW_ALL_USERS` | è®¾ä¸º `true` å¯æ— éœ€ç™½åå•å…è®¸æ‰€æœ‰ç”¨æˆ· |
| `WEIXIN_ACCOUNT_ID` | é€šè¿‡ iLink Bot API æ‰«ç ç™»å½•è·å–çš„å¾®ä¿¡è´¦å· ID |
| `WEIXIN_TOKEN` | é€šè¿‡ iLink Bot API æ‰«ç ç™»å½•è·å–çš„å¾®ä¿¡è®¤è¯ token |
| `WEIXIN_BASE_URL` | è¦†ç›–å¾®ä¿¡ iLink Bot API base URLï¼ˆé»˜è®¤ï¼š`https://ilinkai.weixin.qq.com`ï¼‰ |
| `WEIXIN_CDN_BASE_URL` | è¦†ç›–åª’ä½“çš„å¾®ä¿¡ CDN base URLï¼ˆé»˜è®¤ï¼š`https://novac2c.cdn.weixin.qq.com/c2c`ï¼‰ |
| `WEIXIN_DM_POLICY` | ç§ä¿¡ç­–ç•¥ï¼š`open`ã€`allowlist`ã€`pairing`ã€`disabled`ï¼ˆé»˜è®¤ï¼š`open`ï¼‰ |
| `WEIXIN_GROUP_POLICY` | ç¾¤æ¶ˆæ¯ç­–ç•¥ï¼š`open`ã€`allowlist`ã€`disabled`ï¼ˆé»˜è®¤ï¼š`disabled`ï¼‰ |
| `WEIXIN_ALLOWED_USERS` | å…è®¸ç§ä¿¡ bot çš„é€—å·åˆ†éš”å¾®ä¿¡ç”¨æˆ· ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | å…è®¸ä¸ bot äº’åŠ¨çš„é€—å·åˆ†éš”å¾®ä¿¡**ç¾¤èŠ ID**ï¼ˆéæˆå‘˜ç”¨æˆ· IDï¼‰ã€‚å˜é‡åä¸ºå†å²é—ç•™â€”â€”æœŸæœ›ä¼ å…¥ç¾¤ IDã€‚ä»…å½“ iLink å®é™…æŠ•é€’ç¾¤äº‹ä»¶æ—¶ç”Ÿæ•ˆï¼›æ‰«ç ç™»å½•çš„ iLink bot èº«ä»½ï¼ˆ`...@im.bot`ï¼‰é€šå¸¸ä¸æ¥æ”¶æ™®é€šå¾®ä¿¡ç¾¤æ¶ˆæ¯ã€‚ |
| `WEIXIN_HOME_CHANNEL` | cron æŠ•é€’å’Œé€šçŸ¥çš„å¾®ä¿¡èŠå¤© ID |
| `WEIXIN_HOME_CHANNEL_NAME` | å¾®ä¿¡ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `WEIXIN_ALLOW_ALL_USERS` | æ— éœ€ç™½åå•å…è®¸æ‰€æœ‰å¾®ä¿¡ç”¨æˆ·ï¼ˆ`true`/`false`ï¼‰ |
| `BLUEBUBBLES_SERVER_URL` | BlueBubbles æœåŠ¡å™¨ URLï¼ˆä¾‹å¦‚ `http://192.168.1.10:1234`ï¼‰ |
| `BLUEBUBBLES_PASSWORD` | BlueBubbles æœåŠ¡å™¨å¯†ç  |
| `BLUEBUBBLES_WEBHOOK_HOST` | webhook ç›‘å¬ç»‘å®šåœ°å€ï¼ˆé»˜è®¤ï¼š`127.0.0.1`ï¼‰ |
| `BLUEBUBBLES_WEBHOOK_PORT` | webhook ç›‘å¬ç«¯å£ï¼ˆé»˜è®¤ï¼š`8645`ï¼‰ |
| `BLUEBUBBLES_HOME_CHANNEL` | cron/é€šçŸ¥æŠ•é€’çš„æ‰‹æœº/é‚®ç®± |
| `BLUEBUBBLES_ALLOWED_USERS` | é€—å·åˆ†éš”çš„æˆæƒç”¨æˆ· |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | å…è®¸æ‰€æœ‰ç”¨æˆ·ï¼ˆ`true`/`false`ï¼‰ |
| `QQ_APP_ID` | æ¥è‡ª [q.qq.com](https://q.qq.com) çš„ QQ Bot App ID |
| `QQ_CLIENT_SECRET` | æ¥è‡ª [q.qq.com](https://q.qq.com) çš„ QQ Bot App Secret |
| `QQ_STT_API_KEY` | å¤–éƒ¨ STT å›é€€æä¾›å•†çš„ API å¯†é’¥ï¼ˆå¯é€‰ï¼Œå½“ QQ å†…ç½® ASR æœªè¿”å›æ–‡æœ¬æ—¶ä½¿ç”¨ï¼‰ |
| `QQ_STT_BASE_URL` | å¤–éƒ¨ STT æä¾›å•†çš„ base URLï¼ˆå¯é€‰ï¼‰ |
| `QQ_STT_MODEL` | å¤–éƒ¨ STT æä¾›å•†çš„æ¨¡å‹åç§°ï¼ˆå¯é€‰ï¼‰ |
| `QQ_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš” QQ ç”¨æˆ· openID |
| `QQ_GROUP_ALLOWED_USERS` | ç¾¤ @æ¶ˆæ¯è®¿é—®çš„é€—å·åˆ†éš” QQ ç¾¤ ID |
| `QQ_ALLOW_ALL_USERS` | å…è®¸æ‰€æœ‰ç”¨æˆ·ï¼ˆ`true`/`false`ï¼Œè¦†ç›– `QQ_ALLOWED_USERS`ï¼‰ |
| `QQBOT_HOME_CHANNEL` | cron æŠ•é€’å’Œé€šçŸ¥çš„ QQ ç”¨æˆ·/ç¾¤ openID |
| `QQBOT_HOME_CHANNEL_NAME` | QQ ä¸»é¢‘é“çš„æ˜¾ç¤ºåç§° |
| `QQ_PORTAL_HOST` | è¦†ç›– QQ portal ä¸»æœºï¼ˆè®¾ä¸º `sandbox.q.qq.com` å¯é€šè¿‡æ²™ç®± gateway è·¯ç”±ï¼›é»˜è®¤ï¼š`q.qq.com`ï¼‰ã€‚ |
| `MATTERMOST_URL` | Mattermost æœåŠ¡å™¨ URLï¼ˆä¾‹å¦‚ `https://mm.example.com`ï¼‰ |
| `MATTERMOST_TOKEN` | Mattermost çš„ bot token æˆ–ä¸ªäººè®¿é—® token |
| `MATTERMOST_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš” Mattermost ç”¨æˆ· ID |
| `MATTERMOST_HOME_CHANNEL` | ä¸»åŠ¨æ¶ˆæ¯æŠ•é€’ï¼ˆcronã€é€šçŸ¥ï¼‰çš„é¢‘é“ ID |
| `MATTERMOST_REQUIRE_MENTION` | åœ¨é¢‘é“ä¸­è¦æ±‚ `@mention`ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚è®¾ä¸º `false` å¯å“åº”æ‰€æœ‰æ¶ˆæ¯ã€‚ |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | bot æ— éœ€ `@mention` å³å¯å“åº”çš„é€—å·åˆ†éš”é¢‘é“ ID |
| `MATTERMOST_REPLY_MODE` | å›å¤é£æ ¼ï¼š`thread`ï¼ˆçº¿ç¨‹å›å¤ï¼‰æˆ– `off`ï¼ˆå¹³é“ºæ¶ˆæ¯ï¼Œé»˜è®¤ï¼‰ |
| `MATRIX_HOMESERVER` | Matrix homeserver URLï¼ˆä¾‹å¦‚ `https://matrix.org`ï¼‰ |
| `MATRIX_ACCESS_TOKEN` | bot è®¤è¯çš„ Matrix è®¿é—® token |
| `MATRIX_USER_ID` | Matrix ç”¨æˆ· IDï¼ˆä¾‹å¦‚ `@hermes:matrix.org`ï¼‰â€”â€”å¯†ç ç™»å½•æ—¶å¿…å¡«ï¼Œä½¿ç”¨è®¿é—® token æ—¶å¯é€‰ |
| `MATRIX_PASSWORD` | Matrix å¯†ç ï¼ˆè®¿é—® token çš„æ›¿ä»£æ–¹æ¡ˆï¼‰ |
| `MATRIX_ALLOWED_USERS` | å…è®¸å‘ bot å‘é€æ¶ˆæ¯çš„é€—å·åˆ†éš” Matrix ç”¨æˆ· IDï¼ˆä¾‹å¦‚ `@alice:matrix.org`ï¼‰ |
| `MATRIX_HOME_ROOM` | ä¸»åŠ¨æ¶ˆæ¯æŠ•é€’çš„æˆ¿é—´ IDï¼ˆä¾‹å¦‚ `!abc123:matrix.org`ï¼‰ |
| `MATRIX_ENCRYPTION` | å¯ç”¨ç«¯åˆ°ç«¯åŠ å¯†ï¼ˆ`true`/`false`ï¼Œé»˜è®¤ï¼š`false`ï¼‰ |
| `MATRIX_DEVICE_ID` | ç”¨äº E2EE è·¨é‡å¯æŒä¹…åŒ–çš„ç¨³å®š Matrix è®¾å¤‡ IDï¼ˆä¾‹å¦‚ `HERMES_BOT`ï¼‰ã€‚ä¸è®¾ç½®æ—¶ï¼ŒE2EE å¯†é’¥æ¯æ¬¡å¯åŠ¨éƒ½ä¼šè½®æ¢ï¼Œå†å²æˆ¿é—´è§£å¯†å°†å¤±è´¥ã€‚ |
| `MATRIX_REACTIONS` | å¯¹å…¥ç«™æ¶ˆæ¯å¯ç”¨å¤„ç†ç”Ÿå‘½å‘¨æœŸ emoji ååº”ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚è®¾ä¸º `false` å¯ç¦ç”¨ã€‚ |
| `MATRIX_REQUIRE_MENTION` | åœ¨æˆ¿é—´ä¸­è¦æ±‚ `@mention`ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚è®¾ä¸º `false` å¯å“åº”æ‰€æœ‰æ¶ˆæ¯ã€‚ |
| `MATRIX_FREE_RESPONSE_ROOMS` | bot æ— éœ€ `@mention` å³å¯å“åº”çš„é€—å·åˆ†éš”æˆ¿é—´ ID |
| `MATRIX_AUTO_THREAD` | ä¸ºæˆ¿é—´æ¶ˆæ¯è‡ªåŠ¨åˆ›å»ºçº¿ç¨‹ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ |
| `MATRIX_DM_MENTION_THREADS` | åœ¨ç§èŠä¸­è¢« `@mention` æ—¶åˆ›å»ºçº¿ç¨‹ï¼ˆé»˜è®¤ï¼š`false`ï¼‰ |
| `MATRIX_RECOVERY_KEY` | è®¾å¤‡å¯†é’¥è½®æ¢åäº¤å‰ç­¾åéªŒè¯çš„æ¢å¤å¯†é’¥ã€‚æ¨èç”¨äºå¯ç”¨äº†äº¤å‰ç­¾åçš„ E2EE è®¾ç½®ã€‚ |
| `HASS_TOKEN` | Home Assistant é•¿æœŸè®¿é—® tokenï¼ˆå¯ç”¨ HA å¹³å° + å·¥å…·ï¼‰ |
| `HASS_URL` | Home Assistant URLï¼ˆé»˜è®¤ï¼š`http://homeassistant.local:8123`ï¼‰ |
| `WEBHOOK_ENABLED` | å¯ç”¨ webhook å¹³å°é€‚é…å™¨ï¼ˆ`true`/`false`ï¼‰ |
| `WEBHOOK_PORT` | æ¥æ”¶ webhook çš„ HTTP æœåŠ¡å™¨ç«¯å£ï¼ˆé»˜è®¤ï¼š`8644`ï¼‰ |
| `WEBHOOK_SECRET` | webhook ç­¾åéªŒè¯çš„å…¨å±€ HMAC å¯†é’¥ï¼ˆå½“è·¯ç”±æœªæŒ‡å®šè‡ªå·±çš„å¯†é’¥æ—¶ä½œä¸ºå›é€€ï¼‰ |
| `API_SERVER_ENABLED` | å¯ç”¨ OpenAI å…¼å®¹ API æœåŠ¡å™¨ï¼ˆ`true`/`false`ï¼‰ã€‚ä¸å…¶ä»–å¹³å°å¹¶è¡Œè¿è¡Œã€‚ |
| `API_SERVER_KEY` | API æœåŠ¡å™¨è®¤è¯çš„ Bearer tokenã€‚éå›ç¯ç»‘å®šæ—¶å¼ºåˆ¶æ‰§è¡Œã€‚ |
| `API_SERVER_CORS_ORIGINS` | å…è®¸ç›´æ¥è°ƒç”¨ API æœåŠ¡å™¨çš„é€—å·åˆ†éš”æµè§ˆå™¨æ¥æºï¼ˆä¾‹å¦‚ `http://localhost:3000,http://127.0.0.1:3000`ï¼‰ã€‚é»˜è®¤ï¼šç¦ç”¨ã€‚ |
| `API_SERVER_PORT` | API æœåŠ¡å™¨ç«¯å£ï¼ˆé»˜è®¤ï¼š`8642`ï¼‰ |
| `API_SERVER_HOST` | API æœåŠ¡å™¨ä¸»æœº/ç»‘å®šåœ°å€ï¼ˆé»˜è®¤ï¼š`127.0.0.1`ï¼‰ã€‚ä½¿ç”¨ `0.0.0.0` å¼€æ”¾ç½‘ç»œè®¿é—®â€”â€”éœ€è¦ `API_SERVER_KEY` å’Œä¸¥æ ¼çš„ `API_SERVER_CORS_ORIGINS` ç™½åå•ã€‚ |
| `API_SERVER_MODEL_NAME` | `/v1/models` ä¸Šå…¬å‘Šçš„æ¨¡å‹åç§°ã€‚é»˜è®¤ä¸º profile åç§°ï¼ˆé»˜è®¤ profile ä¸º `hermes-agent`ï¼‰ã€‚é€‚ç”¨äº Open WebUI ç­‰å‰ç«¯éœ€è¦æ¯ä¸ªè¿æ¥ä½¿ç”¨ä¸åŒæ¨¡å‹åç§°çš„å¤šç”¨æˆ·åœºæ™¯ã€‚ |
| `GATEWAY_PROXY_URL` | å°†æ¶ˆæ¯è½¬å‘åˆ°çš„è¿œç¨‹ Hermes API æœåŠ¡å™¨ URLï¼ˆ[ä»£ç†æ¨¡å¼](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)ï¼‰ã€‚è®¾ç½®åï¼Œgateway ä»…å¤„ç†å¹³å° I/Oâ€”â€”æ‰€æœ‰ agent å·¥ä½œå§”æ‰˜ç»™è¿œç¨‹æœåŠ¡å™¨ã€‚ä¹Ÿå¯é€šè¿‡ `config.yaml` ä¸­çš„ `gateway.proxy_url` é…ç½®ã€‚ |
| `GATEWAY_PROXY_KEY` | ä»£ç†æ¨¡å¼ä¸‹ä¸è¿œç¨‹ API æœåŠ¡å™¨è®¤è¯çš„ Bearer tokenã€‚å¿…é¡»ä¸è¿œç¨‹ä¸»æœºä¸Šçš„ `API_SERVER_KEY` ä¸€è‡´ã€‚ |
| `MESSAGING_CWD` | æ¶ˆæ¯æ¨¡å¼ä¸‹ç»ˆç«¯å‘½ä»¤çš„å·¥ä½œç›®å½•ï¼ˆé»˜è®¤ï¼š`~`ï¼‰ |
| `GATEWAY_ALLOWED_USERS` | è·¨æ‰€æœ‰å¹³å°å…è®¸çš„é€—å·åˆ†éš”ç”¨æˆ· ID |
| `GATEWAY_ALLOW_ALL_USERS` | æ— éœ€ç™½åå•å…è®¸æ‰€æœ‰ç”¨æˆ·ï¼ˆ`true`/`false`ï¼Œé»˜è®¤ï¼š`false`ï¼‰ |

### Microsoft Graphï¼ˆTeams ä¼šè®®ï¼‰

ç”¨äºå³å°†æ¨å‡ºçš„ Teams ä¼šè®®æ‘˜è¦æµæ°´çº¿çš„ Microsoft Graph REST å®¢æˆ·ç«¯çš„ä»…åº”ç”¨å‡­è¯ã€‚Azure é—¨æˆ·æ“ä½œæ­¥éª¤å’Œæ‰€éœ€ API æƒé™è¯¦è§[æ³¨å†Œ Microsoft Graph åº”ç”¨ç¨‹åº](/guides/microsoft-graph-app-registration)ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `MSGRAPH_TENANT_ID` | Graph åº”ç”¨æ³¨å†Œçš„ Azure AD ç§Ÿæˆ· IDï¼ˆç›®å½• GUIDï¼‰ã€‚ |
| `MSGRAPH_CLIENT_ID` | Azure åº”ç”¨æ³¨å†Œçš„åº”ç”¨ç¨‹åºï¼ˆå®¢æˆ·ç«¯ï¼‰IDã€‚ |
| `MSGRAPH_CLIENT_SECRET` | åº”ç”¨æ³¨å†Œçš„å®¢æˆ·ç«¯å¯†é’¥å€¼ã€‚å­˜å‚¨åœ¨ `~/.hermes/.env` ä¸­å¹¶è®¾ç½® `chmod 600`ï¼›å®šæœŸé€šè¿‡ Azure é—¨æˆ·è½®æ¢ã€‚ |
| `MSGRAPH_SCOPE` | å®¢æˆ·ç«¯å‡­è¯ token è¯·æ±‚çš„ OAuth2 èŒƒå›´ï¼ˆé»˜è®¤ï¼š`https://graph.microsoft.com/.default`ï¼‰ã€‚ |
| `MSGRAPH_AUTHORITY_URL` | Microsoft èº«ä»½å¹³å° authorityï¼ˆé»˜è®¤ï¼š`https://login.microsoftonline.com`ï¼‰ã€‚ä»…å¯¹å›½å®¶/ä¸»æƒäº‘è¦†ç›–ï¼ˆä¾‹å¦‚ GCC High ä½¿ç”¨ `https://login.microsoftonline.us`ï¼‰ã€‚ |

### Microsoft Graph Webhook ç›‘å¬å™¨

Graph äº‹ä»¶ï¼ˆTeams ä¼šè®®ã€æ—¥å†ã€èŠå¤©ç­‰ï¼‰çš„å…¥ç«™å˜æ›´é€šçŸ¥ç›‘å¬å™¨ã€‚è®¾ç½®å’Œå®‰å…¨åŠ å›ºè¯¦è§ [Microsoft Graph Webhook ç›‘å¬å™¨](/user-guide/messaging/msgraph-webhook)ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `MSGRAPH_WEBHOOK_ENABLED` | å¯ç”¨ `msgraph_webhook` gateway å¹³å°ï¼ˆ`true`/`1`/`yes`ï¼‰ã€‚ |
| `MSGRAPH_WEBHOOK_PORT` | ç›‘å¬å™¨ç»‘å®šç«¯å£ï¼ˆé»˜è®¤ï¼š`8646`ï¼‰ã€‚ |
| `MSGRAPH_WEBHOOK_CLIENT_STATE` | Graph åœ¨æ¯æ¬¡é€šçŸ¥ä¸­å›ä¼ çš„å…±äº«å¯†é’¥ï¼›ä¸ `hmac.compare_digest` æ¯”è¾ƒã€‚ä½¿ç”¨ `openssl rand -hex 32` ç”Ÿæˆã€‚ |
| `MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES` | é€—å·åˆ†éš”çš„ Graph èµ„æºè·¯å¾„/æ¨¡å¼ç™½åå•ï¼ˆä¾‹å¦‚ `communications/onlineMeetings,chats/*/messages`ï¼‰ã€‚æœ«å°¾ `*` ä¸ºå‰ç¼€åŒ¹é…ã€‚ä¸ºç©ºåˆ™æ¥å—æ‰€æœ‰ã€‚ |
| `MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS` | å…è®¸ POST åˆ°ç›‘å¬å™¨çš„é€—å·åˆ†éš” CIDR èŒƒå›´ï¼ˆä¾‹å¦‚ `52.96.0.0/14,52.104.0.0/14`ï¼‰ã€‚ä¸ºç©ºåˆ™å…è®¸æ‰€æœ‰ï¼ˆé»˜è®¤ï¼‰ã€‚ç”Ÿäº§ç¯å¢ƒä¸­åº”é™åˆ¶ä¸º Microsoft Graph å…¬å¸ƒçš„å‡ºå£èŒƒå›´ã€‚ |

### Teams ä¼šè®®æ‘˜è¦æŠ•é€’

ä»…åœ¨å¯ç”¨ [`teams_pipeline` æ’ä»¶](/user-guide/messaging/msgraph-webhook)æ—¶ä½¿ç”¨ã€‚è®¾ç½®ä¹Ÿå¯åœ¨ `config.yaml` çš„ `platforms.teams.extra` ä¸‹é…ç½®â€”â€”ä¸¤è€…éƒ½è®¾ç½®æ—¶ç¯å¢ƒå˜é‡ä¼˜å…ˆã€‚å‚è§ [Microsoft Teams â†’ ä¼šè®®æ‘˜è¦æŠ•é€’](/user-guide/messaging/teams#meeting-summary-delivery-teams-meeting-pipeline)ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `TEAMS_DELIVERY_MODE` | `graph` æˆ– `incoming_webhook`ã€‚ |
| `TEAMS_INCOMING_WEBHOOK_URL` | Teams ç”Ÿæˆçš„ webhook URLï¼›`TEAMS_DELIVERY_MODE=incoming_webhook` æ—¶å¿…å¡«ã€‚ |
| `TEAMS_GRAPH_ACCESS_TOKEN` | Graph æŠ•é€’çš„é¢„è·å–å§”æ‰˜è®¿é—® tokenã€‚æå°‘éœ€è¦â€”â€”æœªè®¾ç½®æ—¶ writer å›é€€åˆ° `MSGRAPH_*` åº”ç”¨å‡­è¯ã€‚ |
| `TEAMS_TEAM_ID` | é¢‘é“æŠ•é€’çš„ç›®æ ‡ Team IDï¼ˆ`graph` æ¨¡å¼ï¼‰ã€‚ |
| `TEAMS_CHANNEL_ID` | ç›®æ ‡é¢‘é“ IDï¼ˆä¸ `TEAMS_TEAM_ID` é…å¯¹ï¼‰ã€‚ |
| `TEAMS_CHAT_ID` | ç›®æ ‡ 1:1 æˆ–ç¾¤èŠ IDï¼ˆ`graph` æ¨¡å¼ä¸‹ team+channel çš„æ›¿ä»£æ–¹æ¡ˆï¼‰ã€‚ |

### LINE Messaging API

ç”±å†…ç½® LINE å¹³å°æ’ä»¶ï¼ˆ`plugins/platforms/line/`ï¼‰ä½¿ç”¨ã€‚å®Œæ•´è®¾ç½®è¯¦è§ [æ¶ˆæ¯ Gateway â†’ LINE](/user-guide/messaging/line)ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `LINE_CHANNEL_ACCESS_TOKEN` | æ¥è‡ª LINE Developers Consoleï¼ˆMessaging API æ ‡ç­¾ï¼‰çš„é•¿æœŸé¢‘é“è®¿é—® tokenã€‚å¿…å¡«ã€‚ |
| `LINE_CHANNEL_SECRET` | é¢‘é“å¯†é’¥ï¼ˆBasic settings æ ‡ç­¾ï¼‰ï¼›ç”¨äº HMAC-SHA256 webhook ç­¾åéªŒè¯ã€‚å¿…å¡«ã€‚ |
| `LINE_HOST` | webhook ç»‘å®šä¸»æœºï¼ˆé»˜è®¤ï¼š`0.0.0.0`ï¼‰ã€‚ |
| `LINE_PORT` | webhook ç»‘å®šç«¯å£ï¼ˆé»˜è®¤ï¼š`8646`ï¼‰ã€‚ |
| `LINE_PUBLIC_URL` | å…¬å…± HTTPS base URLï¼ˆä¾‹å¦‚ `https://my-tunnel.example.com`ï¼‰ã€‚å‘é€å›¾ç‰‡/éŸ³é¢‘/è§†é¢‘æ—¶å¿…å¡«â€”â€”LINE ä»…æ¥å— HTTPS å¯è®¿é—®çš„ URLã€‚ |
| `LINE_ALLOWED_USERS` | å…è®¸ç§ä¿¡ bot çš„é€—å·åˆ†éš”ç”¨æˆ· IDï¼ˆ`U` å‰ç¼€ï¼‰ã€‚ |
| `LINE_ALLOWED_GROUPS` | bot å°†åœ¨å…¶ä¸­å“åº”çš„é€—å·åˆ†éš”ç¾¤ç»„ IDï¼ˆ`C` å‰ç¼€ï¼‰ã€‚ |
| `LINE_ALLOWED_ROOMS` | bot å°†åœ¨å…¶ä¸­å“åº”çš„é€—å·åˆ†éš”æˆ¿é—´ IDï¼ˆ`R` å‰ç¼€ï¼‰ã€‚ |
| `LINE_ALLOW_ALL_USERS` | ä»…ç”¨äºå¼€å‘çš„é€ƒç”Ÿèˆ±â€”â€”æ¥å—ä»»æ„æ¥æºã€‚é»˜è®¤ï¼š`false`ã€‚ |
| `LINE_HOME_CHANNEL` | `deliver: line` çš„ cron ä»»åŠ¡çš„é»˜è®¤æŠ•é€’ç›®æ ‡ã€‚ |
| `LINE_SLOW_RESPONSE_THRESHOLD` | æ…¢é€Ÿ LLM Template Buttons postback è§¦å‘å‰çš„ç­‰å¾…ç§’æ•°ï¼ˆé»˜è®¤ï¼š`45`ï¼‰ã€‚è®¾ä¸º `0` å¯ç¦ç”¨å¹¶å§‹ç»ˆä½¿ç”¨ Push å›é€€ã€‚ |
| `LINE_PENDING_TEXT` | ä¸ postback æŒ‰é’®ä¸€èµ·æ˜¾ç¤ºçš„æ°”æ³¡æ–‡æœ¬ã€‚ |
| `LINE_BUTTON_LABEL` | Postback æŒ‰é’®æ ‡ç­¾ï¼ˆé»˜è®¤ï¼š`Get answer`ï¼‰ã€‚ |
| `LINE_DELIVERED_TEXT` | å†æ¬¡ç‚¹å‡»å·²æŠ•é€’ postback æ—¶çš„å›å¤ï¼ˆé»˜è®¤ï¼š`Already replied âœ…`ï¼‰ã€‚ |
| `LINE_INTERRUPTED_TEXT` | ç‚¹å‡» `/stop` å­¤ç«‹ postback æŒ‰é’®æ—¶çš„å›å¤ï¼ˆé»˜è®¤ï¼š`Run was interrupted before completion.`ï¼‰ã€‚ |

### ntfyï¼ˆæ¨é€é€šçŸ¥ï¼‰

[ntfy](https://ntfy.sh/) æ˜¯ä¸€ä¸ªè½»é‡çº§åŸºäº HTTP çš„æ¨é€é€šçŸ¥æœåŠ¡ã€‚é€šè¿‡ [ntfy ç§»åŠ¨åº”ç”¨](https://ntfy.sh/docs/subscribe/phone/)è®¢é˜…è¯é¢˜ï¼Œå‘è¯¥è¯é¢˜å‘å¸ƒæ¶ˆæ¯å³å¯ä¸ agent äº¤äº’ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `NTFY_TOPIC` | è®¢é˜…çš„è¯é¢˜ï¼ˆå…¥ç«™æ¶ˆæ¯ï¼‰ã€‚å¿…å¡«ã€‚ |
| `NTFY_SERVER_URL` | æœåŠ¡å™¨ URLï¼ˆé»˜è®¤ï¼š`https://ntfy.sh`ï¼‰ã€‚æŒ‡å‘è‡ªæ‰˜ç®¡ ntfy ä»¥ä¿æŠ¤éšç§ã€‚ |
| `NTFY_TOKEN` | å¯é€‰è®¤è¯ tokenã€‚Bearer tokenï¼ˆä¾‹å¦‚ `tk_xyz`ï¼‰æˆ– `user:pass` ç”¨äº Basic è®¤è¯ã€‚ |
| `NTFY_PUBLISH_TOPIC` | å‡ºç«™å›å¤çš„è¯é¢˜ï¼ˆé»˜è®¤ä¸º `NTFY_TOPIC`ï¼‰ã€‚ |
| `NTFY_MARKDOWN` | è®¾ä¸º `true` å¯ä½¿ç”¨ `X-Markdown: true` å¤´å‘é€å›å¤ã€‚é»˜è®¤ï¼š`false`ã€‚ |
| `NTFY_ALLOWED_USERS` | ç™½åå•ï¼ˆè§†ä¸ºç”¨æˆ· IDï¼›åœ¨ ntfy ä¸­å³è¯é¢˜åç§°ï¼‰ã€‚é€šå¸¸è®¾ä¸ºä¸ `NTFY_TOPIC` ç›¸åŒçš„å€¼ã€‚ |
| `NTFY_ALLOW_ALL_USERS` | ä»…ç”¨äºå¼€å‘çš„é€ƒç”Ÿèˆ±â€”â€”ä»…åœ¨è®¿é—®æ§åˆ¶çš„ç§æœ‰è¯é¢˜ä¸Šå®‰å…¨ã€‚é»˜è®¤ï¼š`false`ã€‚ |
| `NTFY_HOME_CHANNEL` | `deliver: ntfy` çš„ cron ä»»åŠ¡çš„é»˜è®¤æŠ•é€’ç›®æ ‡ã€‚ |
| `NTFY_HOME_CHANNEL_NAME` | ä¸»é¢‘é“çš„äººç±»å¯è¯»æ ‡ç­¾ï¼ˆé»˜è®¤ä¸ºè¯é¢˜åç§°ï¼‰ã€‚ |

åœ¨ä½¿ç”¨ä¸å—ä¿¡ä»»çš„è¯é¢˜éƒ¨ç½²å‰ï¼Œè¯·å‚é˜… [ntfy æ¶ˆæ¯æŒ‡å—](/user-guide/messaging/ntfy)â€”â€”ç‰¹åˆ«æ˜¯**èº«ä»½æ¨¡å‹**éƒ¨åˆ†ã€‚

### é«˜çº§æ¶ˆæ¯è°ƒä¼˜

ç”¨äºé™åˆ¶å‡ºç«™æ¶ˆæ¯æ‰¹å¤„ç†å™¨çš„é«˜çº§æ¯å¹³å°æ—‹é’®ã€‚å¤§å¤šæ•°ç”¨æˆ·æ— éœ€è°ƒæ•´ï¼›é»˜è®¤å€¼å·²è®¾ç½®ä¸ºåœ¨éµå®ˆå„å¹³å°é€Ÿç‡é™åˆ¶çš„åŒæ—¶ä¸æ˜¾å¾—è¿Ÿç¼“ã€‚

| å˜é‡ | æè¿° |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | åˆ·æ–°æ’é˜Ÿ Telegram æ–‡æœ¬å—å‰çš„å®½é™çª—å£ï¼ˆé»˜è®¤ï¼š`0.6`ï¼‰ã€‚ |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | å•æ¡ Telegram æ¶ˆæ¯è¶…è¿‡é•¿åº¦é™åˆ¶æ—¶åˆ†å—ä¹‹é—´çš„å»¶è¿Ÿï¼ˆé»˜è®¤ï¼š`2.0`ï¼‰ã€‚ |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | åˆ·æ–°æ’é˜Ÿ Telegram åª’ä½“å‰çš„å®½é™çª—å£ï¼ˆé»˜è®¤ï¼š`0.6`ï¼‰ã€‚ |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | agent å®Œæˆåå‘é€åç»­æ¶ˆæ¯å‰çš„å»¶è¿Ÿï¼Œä»¥é¿å…ä¸æœ€åä¸€ä¸ªæµå—ç«äº‰ã€‚ |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | è¦†ç›–åº•å±‚ `python-telegram-bot` HTTP è¶…æ—¶ï¼ˆç§’ï¼‰ã€‚ |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | åˆ° Telegram API çš„æœ€å¤§å¹¶å‘ HTTP è¿æ¥æ•°ã€‚ |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | ç¦ç”¨ DNS å¤±è´¥æ—¶ä½¿ç”¨çš„ç¡¬ç¼–ç  Cloudflare å›é€€ IPï¼ˆ`true`/`false`ï¼‰ã€‚ |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | åˆ·æ–°æ’é˜Ÿ Discord æ–‡æœ¬å—å‰çš„å®½é™çª—å£ï¼ˆé»˜è®¤ï¼š`0.6`ï¼‰ã€‚ |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Discord æ¶ˆæ¯è¶…è¿‡é•¿åº¦é™åˆ¶æ—¶åˆ†å—ä¹‹é—´çš„å»¶è¿Ÿï¼ˆé»˜è®¤ï¼š`2.0`ï¼‰ã€‚ |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Matrix ç­‰åŒäº Telegram æ‰¹å¤„ç†æ—‹é’®ã€‚ |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | é£ä¹¦æ‰¹å¤„ç†å™¨è°ƒä¼˜â€”â€”å»¶è¿Ÿã€åˆ†å—å»¶è¿Ÿã€æ¯æ¡æ¶ˆæ¯æœ€å¤§å­—ç¬¦æ•°ã€æ¯æ‰¹æœ€å¤§æ¶ˆæ¯æ•°ã€‚ |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | é£ä¹¦åª’ä½“åˆ·æ–°å»¶è¿Ÿã€‚ |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | é£ä¹¦ webhook å»é‡ç¼“å­˜å¤§å°ï¼ˆé»˜è®¤ï¼š`1024`ï¼‰ã€‚ |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | ä¼ä¸šå¾®ä¿¡æ‰¹å¤„ç†å™¨è°ƒä¼˜ã€‚ |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | å°†å›¾ç‰‡äº¤ç»™è§†è§‰æ¨¡å‹å‰ä¸‹è½½çš„è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`30`ï¼‰ã€‚ |
| `HERMES_RESTART_DRAIN_TIMEOUT` | Gatewayï¼š`/restart` æ—¶ç­‰å¾…æ´»è·ƒè¿è¡Œæ’ç©ºçš„ç§’æ•°ï¼Œè¶…æ—¶åå¼ºåˆ¶é‡å¯ï¼ˆé»˜è®¤ï¼š`900`ï¼‰ã€‚ |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | gateway å¯åŠ¨æœŸé—´æ¯ä¸ªå¹³å°çš„è¿æ¥è¶…æ—¶ï¼ˆç§’ï¼‰ã€‚ |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | é»˜è®¤ gateway ç¹å¿™è¾“å…¥è¡Œä¸ºï¼š`queue`ã€`steer` æˆ– `interrupt`ã€‚å¯é€šè¿‡ `/busy` æŒ‰èŠå¤©è¦†ç›–ã€‚ |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | gateway æ˜¯å¦åœ¨ç”¨æˆ· agent ç¹å¿™æ—¶å‘é€ç¡®è®¤æ¶ˆæ¯ï¼ˆâš¡/â³/â©ï¼‰ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚è®¾ä¸º `false` å¯å®Œå…¨æŠ‘åˆ¶è¿™äº›æ¶ˆæ¯â€”â€”è¾“å…¥ä»ä¼šæ­£å¸¸æ’é˜Ÿ/å¼•å¯¼/ä¸­æ–­ï¼Œåªæ˜¯èŠå¤©å›å¤è¢«é™é»˜ã€‚ä» `config.yaml` ä¸­çš„ `display.busy_ack_enabled` æ¡¥æ¥ã€‚ |
| `HERMES_GATEWAY_NO_SUPERVISE` | åœ¨ s6-overlay Docker é•œåƒå†…éƒ¨è¿è¡Œ `hermes gateway run` æ—¶è·³è¿‡ s6 è‡ªåŠ¨ç›‘ç®¡ï¼Œé€€å›åˆ° pre-s6 å‰å°è¯­ä¹‰ï¼ˆæ— è‡ªåŠ¨é‡å¯ï¼Œgateway ä½œä¸ºå®¹å™¨ä¸»è¿›ç¨‹ï¼‰ã€‚çœŸå€¼ï¼š`1`ã€`true`ã€`yes`ã€‚ç­‰åŒäº `--no-supervise` CLI æ ‡å¿—ã€‚åœ¨ s6 é•œåƒä¹‹å¤–ä¸ºç©ºæ“ä½œã€‚ |
| `HERMES_GATEWAY_BOOTSTRAP_STATE` | åœ¨ s6-overlay Docker é•œåƒå†…éƒ¨ï¼Œä¸º**å…¨æ–°å·**å£°æ˜ gateway çš„åˆå§‹å—ç›‘ç®¡çŠ¶æ€ã€‚ç©ºç™½å·ä¸Šä¸å­˜åœ¨æŒä¹…åŒ–çš„ `gateway_state.json`ï¼Œå› æ­¤å¯åŠ¨åè°ƒå™¨ä¼šæ³¨å†Œ `gateway-default` æ§½ä½ä½†ä¿æŒå…¶**å…³é—­**ï¼ˆåªæœ‰ä¸Šæ¬¡è®°å½•çŠ¶æ€ä¸º `running` æ—¶æ‰ä¼šè‡ªåŠ¨å¯åŠ¨ï¼‰ã€‚å°†æ­¤å˜é‡è®¾ä¸º `running` åï¼Œé¦–æ¬¡å¯åŠ¨ hook ä¼šåœ¨åè°ƒå™¨è¿è¡Œå‰é¢„å†™å…¥ `gateway_state.json`ï¼Œä»è€Œè®© gateway åœ¨ç¬¬ä¸€æ¬¡å¯åŠ¨æ—¶å°±è‡ªåŠ¨æ‹‰èµ·ã€‚ä»…å­—é¢å€¼ `running` ç”Ÿæ•ˆã€‚ä»…å½±å“é¦–æ¬¡å¯åŠ¨ï¼šè‹¥å·²æœ‰ `gateway_state.json`ï¼Œç»ä¸ä¼šè¢«è¦†ç›–ï¼Œå› æ­¤è¢«åˆ»æ„åœæ­¢çš„ gateway åœ¨é‡å¯åä»ä¿æŒåœæ­¢ã€‚åœ¨ s6 é•œåƒä¹‹å¤–ä¸ºç©ºæ“ä½œã€‚ |
| `HERMES_FILE_MUTATION_VERIFIER` | å¯ç”¨æ¯è½®æ–‡ä»¶å˜æ›´éªŒè¯å™¨é¡µè„šï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚å¯ç”¨åï¼ŒHermes é™„åŠ ä¸€ä¸ªå»ºè®®åˆ—è¡¨ï¼Œåˆ—å‡ºæœ¬è½®ä¸­å¤±è´¥ä¸”æœªè¢«æˆåŠŸå†™å…¥è¦†ç›–çš„ `write_file`/`patch` è°ƒç”¨ã€‚è®¾ä¸º `0`ã€`false`ã€`no` æˆ– `off` å¯æŠ‘åˆ¶ã€‚é•œåƒ `config.yaml` ä¸­çš„ `display.file_mutation_verifier`ï¼›è®¾ç½®æ—¶ç¯å¢ƒå˜é‡ä¼˜å…ˆã€‚ |
| `HERMES_CRON_TIMEOUT` | cron ä»»åŠ¡ agent è¿è¡Œçš„ä¸æ´»åŠ¨è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`600`ï¼‰ã€‚agent åœ¨ä¸»åŠ¨è°ƒç”¨å·¥å…·æˆ–æ¥æ”¶æµ token æ—¶å¯æ— é™è¿è¡Œâ€”â€”ä»…åœ¨ç©ºé—²æ—¶è§¦å‘ã€‚è®¾ä¸º `0` è¡¨ç¤ºæ— é™åˆ¶ã€‚ |
| `HERMES_CRON_SCRIPT_TIMEOUT` | cron ä»»åŠ¡é™„åŠ çš„é¢„è¿è¡Œè„šæœ¬è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`3600`ï¼‰ã€‚å¯¹éœ€è¦æ›´é•¿æ‰§è¡Œæ—¶é—´çš„è„šæœ¬ï¼ˆä¾‹å¦‚éšæœºå»¶è¿Ÿçš„åæœºå™¨äººè®¡æ—¶ï¼‰å¯å¢å¤§æ­¤å€¼ã€‚ä¹Ÿå¯é€šè¿‡ `config.yaml` ä¸­çš„ `cron.script_timeout_seconds` é…ç½®ã€‚ |
| `HERMES_CRON_MAX_PARALLEL` | æ¯æ¬¡ tick å¹¶è¡Œè¿è¡Œçš„æœ€å¤§ cron ä»»åŠ¡æ•°ï¼ˆé»˜è®¤ï¼š`4`ï¼‰ã€‚ |

## Agent è¡Œä¸º

| å˜é‡ | æè¿° |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | æ¯æ¬¡å¯¹è¯çš„æœ€å¤§å·¥å…·è°ƒç”¨è¿­ä»£æ¬¡æ•°ï¼ˆé»˜è®¤ï¼š90ï¼‰ |
| `HERMES_INFERENCE_MODEL` | åœ¨è¿›ç¨‹çº§åˆ«è¦†ç›–æ¨¡å‹åç§°ï¼ˆä¼˜å…ˆäºæœ¬æ¬¡ä¼šè¯çš„ `config.yaml`ï¼‰ã€‚ä¹Ÿå¯é€šè¿‡ `-m`/`--model` æ ‡å¿—è®¾ç½®ã€‚ |
| `HERMES_YOLO_MODE` | è®¾ä¸º `1` å¯ç»•è¿‡å±é™©å‘½ä»¤å®¡æ‰¹æç¤ºã€‚ç­‰åŒäº `--yolo`ã€‚ |
| `HERMES_ACCEPT_HOOKS` | æ— éœ€ TTY æç¤ºè‡ªåŠ¨æ‰¹å‡† `config.yaml` ä¸­å£°æ˜çš„ä»»ä½•æœªè§è¿‡çš„ shell hookã€‚ç­‰åŒäº `--accept-hooks` æˆ– `hooks_auto_accept: true`ã€‚ |
| `HERMES_IGNORE_USER_CONFIG` | è·³è¿‡ `~/.hermes/config.yaml` å¹¶ä½¿ç”¨å†…ç½®é»˜è®¤å€¼ï¼ˆ`.env` ä¸­çš„å‡­è¯ä»ä¼šåŠ è½½ï¼‰ã€‚ç­‰åŒäº `--ignore-user-config`ã€‚ |
| `HERMES_IGNORE_RULES` | è·³è¿‡ `AGENTS.md`ã€`SOUL.md`ã€`.cursorrules`ã€è®°å¿†å’Œé¢„åŠ è½½æŠ€èƒ½çš„è‡ªåŠ¨æ³¨å…¥ã€‚ç­‰åŒäº `--ignore-rules`ã€‚ |
| `HERMES_SAFE_MODE` | æ•…éšœæ’æŸ¥æ¨¡å¼ï¼šç¦ç”¨**æ‰€æœ‰**è‡ªå®šä¹‰é¡¹â€”â€”è·³è¿‡æ’ä»¶å‘ç°ã€MCP æœåŠ¡å™¨åŠ è½½å’Œ shell hook æ³¨å†Œã€‚ç”± `--safe-mode` è‡ªåŠ¨è®¾ç½®ï¼ˆåŒæ—¶ä¹Ÿä¼šè®¾ç½®ä¸Šé¢ä¸¤ä¸ª flagï¼‰ã€‚ |
| `HERMES_MD_NAMES` | è‡ªåŠ¨æ³¨å…¥çš„è§„åˆ™æ–‡ä»¶åé€—å·åˆ†éš”åˆ—è¡¨ï¼ˆé»˜è®¤ï¼š`AGENTS.md,CLAUDE.md,.cursorrules,SOUL.md`ï¼‰ã€‚ |
| `HERMES_TOOL_PROGRESS` | å·¥å…·è¿›åº¦æ˜¾ç¤ºçš„å·²å¼ƒç”¨å…¼å®¹å˜é‡ã€‚ä¼˜å…ˆä½¿ç”¨ `config.yaml` ä¸­çš„ `display.tool_progress`ã€‚ |
| `HERMES_TOOL_PROGRESS_MODE` | å·¥å…·è¿›åº¦æ¨¡å¼çš„å·²å¼ƒç”¨å…¼å®¹å˜é‡ã€‚ä¼˜å…ˆä½¿ç”¨ `config.yaml` ä¸­çš„ `display.tool_progress`ã€‚ |
| `HERMES_HUMAN_DELAY_MODE` | å“åº”èŠ‚å¥ï¼š`off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | è‡ªå®šä¹‰å»¶è¿ŸèŒƒå›´æœ€å°å€¼ï¼ˆæ¯«ç§’ï¼‰ |
| `HERMES_HUMAN_DELAY_MAX_MS` | è‡ªå®šä¹‰å»¶è¿ŸèŒƒå›´æœ€å¤§å€¼ï¼ˆæ¯«ç§’ï¼‰ |
| `HERMES_QUIET` | æŠ‘åˆ¶éå¿…è¦è¾“å‡ºï¼ˆ`true`/`false`ï¼‰ |
| `CODEX_HOME` | å¯ç”¨ [Codex åº”ç”¨æœåŠ¡å™¨è¿è¡Œæ—¶](../user-guide/features/codex-app-server-runtime)æ—¶ï¼Œè¦†ç›– Codex CLI è¯»å–å…¶é…ç½® + è®¤è¯çš„ç›®å½•ï¼ˆé»˜è®¤ï¼š`~/.codex`ï¼‰ã€‚Hermes çš„è¿ç§»å°†æ‰˜ç®¡å—å†™å…¥ `<CODEX_HOME>/config.toml`ã€‚ |
| `HERMES_KANBAN_TASK` | kanban è°ƒåº¦å™¨ç”Ÿæˆå·¥ä½œè¿›ç¨‹æ—¶è®¾ç½®ï¼ˆä»»åŠ¡ UUIDï¼‰ã€‚å·¥ä½œè¿›ç¨‹å’Œç”Ÿæˆçš„ `hermes-tools` MCP å­è¿›ç¨‹ç»§æ‰¿å®ƒï¼Œä»¥ä¾¿ kanban å·¥å…·æ­£ç¡®é—¨æ§ã€‚è¯·å‹¿æ‰‹åŠ¨è®¾ç½®ã€‚ |
| `HERMES_API_TIMEOUT` | LLM API è°ƒç”¨è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`1800`ï¼‰ |
| `HERMES_API_CALL_STALE_TIMEOUT` | éæµå¼è¿‡æœŸè°ƒç”¨è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`90`ï¼‰ã€‚æœªè®¾ç½®æ—¶å¯¹æœ¬åœ°æä¾›å•†è‡ªåŠ¨ç¦ç”¨ã€‚ä¹Ÿå¯é€šè¿‡ `config.yaml` ä¸­çš„ `providers.<id>.stale_timeout_seconds` æˆ– `providers.<id>.models.<model>.stale_timeout_seconds` é…ç½®ã€‚ |
| `HERMES_STREAM_READ_TIMEOUT` | æµå¼ socket è¯»å–è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`120`ï¼‰ã€‚å¯¹æœ¬åœ°æä¾›å•†è‡ªåŠ¨å¢å¤§åˆ° `HERMES_API_TIMEOUT`ã€‚å¦‚æœæœ¬åœ° LLM åœ¨é•¿ä»£ç ç”ŸæˆæœŸé—´è¶…æ—¶ï¼Œè¯·å¢å¤§æ­¤å€¼ã€‚ |
| `HERMES_STREAM_STALE_TIMEOUT` | è¿‡æœŸæµæ£€æµ‹è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`180`ï¼‰ã€‚å¯¹æœ¬åœ°æä¾›å•†è‡ªåŠ¨ç¦ç”¨ã€‚åœ¨æ­¤çª—å£å†…æ— å—åˆ°è¾¾æ—¶è§¦å‘è¿æ¥ç»ˆæ­¢ã€‚ |
| `HERMES_STREAM_RETRIES` | ç¬æ—¶ç½‘ç»œé”™è¯¯æ—¶çš„æµä¸­é‡è¿å°è¯•æ¬¡æ•°ï¼ˆé»˜è®¤ï¼š`3`ï¼‰ã€‚ |
| `HERMES_AGENT_TIMEOUT` | gateway ä¸­è¿è¡Œ agent çš„ä¸æ´»åŠ¨è¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`1800`ï¼‰ã€‚æ¯æ¬¡å·¥å…·è°ƒç”¨å’Œæµ token æ—¶é‡ç½®ã€‚è®¾ä¸º `0` å¯ç¦ç”¨ã€‚ |
| `HERMES_AGENT_TIMEOUT_WARNING` | Gatewayï¼šä¸æ´»åŠ¨è¶…è¿‡æ­¤ç§’æ•°åå‘é€è­¦å‘Šæ¶ˆæ¯ï¼ˆé»˜è®¤ï¼š`HERMES_AGENT_TIMEOUT` çš„ 75%ï¼‰ã€‚ |
| `HERMES_AGENT_NOTIFY_INTERVAL` | Gatewayï¼šé•¿æ—¶é—´è¿è¡Œçš„ agent è½®æ¬¡ä¸­è¿›åº¦é€šçŸ¥çš„é—´éš”ï¼ˆç§’ï¼‰ã€‚ |
| `HERMES_CHECKPOINT_TIMEOUT` | æ–‡ä»¶ç³»ç»Ÿæ£€æŸ¥ç‚¹åˆ›å»ºè¶…æ—¶ï¼ˆç§’ï¼Œé»˜è®¤ï¼š`30`ï¼‰ã€‚ |
| `HERMES_EXEC_ASK` | åœ¨ gateway æ¨¡å¼ä¸‹å¯ç”¨æ‰§è¡Œå®¡æ‰¹æç¤ºï¼ˆ`true`/`false`ï¼‰ |
| `HERMES_ENABLE_PROJECT_PLUGINS` | ä¸º agent åŠ è½½å™¨å’Œä»ªè¡¨æ¿ Web æœåŠ¡å™¨å¯ç”¨ä» `./.hermes/plugins/` è‡ªåŠ¨å‘ç°ä»“åº“æœ¬åœ°æ’ä»¶ã€‚æ¥å—æ ‡å‡†çœŸå€¼é›†ï¼š`1`/`true`/`yes`/`on`ï¼ˆä¸åŒºåˆ†å¤§å°å†™ï¼‰ã€‚å…¶ä»–æ‰€æœ‰å€¼â€”â€”åŒ…æ‹¬ `0`ã€`false`ã€`no`ã€`off` å’Œç©ºå­—ç¬¦ä¸²â€”â€”å‡è§†ä¸º**ç¦ç”¨**ï¼ˆé»˜è®¤ï¼‰ã€‚æ³¨æ„ï¼šè‡ª GHSA-5qr3-c538-wm9jï¼ˆ#29156ï¼‰èµ·ï¼Œå³ä½¿å¯ç”¨æ­¤å˜é‡ï¼Œä»ªè¡¨æ¿ Web æœåŠ¡å™¨ä¹Ÿæ‹’ç»è‡ªåŠ¨å¯¼å…¥é¡¹ç›®æ’ä»¶çš„ Python `api` æ–‡ä»¶â€”â€”é¡¹ç›®æ’ä»¶å¯é€šè¿‡é™æ€ JS/CSS æ‰©å±• UIï¼Œä½†å…¶åç«¯è·¯ç”±ä»…åœ¨ç§»è‡³ `~/.hermes/plugins/` åæ‰ä¼šåŠ è½½ã€‚ |
| `HERMES_PLUGINS_DEBUG` | `1`/`true` å¯åœ¨ stderr ä¸Šè¾“å‡ºè¯¦ç»†çš„æ’ä»¶å‘ç°æ—¥å¿—â€”â€”æ‰«æçš„ç›®å½•ã€è§£æçš„ manifestã€è·³è¿‡åŸå› ä»¥åŠè§£ææˆ– `register()` å¤±è´¥æ—¶çš„å®Œæ•´å›æº¯ã€‚é¢å‘æ’ä»¶ä½œè€…ã€‚ |
| `HERMES_BACKGROUND_NOTIFICATIONS` | gateway ä¸­åå°è¿›ç¨‹é€šçŸ¥æ¨¡å¼ï¼š`all`ï¼ˆé»˜è®¤ï¼‰ã€`result`ã€`error`ã€`off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | åœ¨ API è°ƒç”¨æ—¶æ³¨å…¥çš„ä¸´æ—¶ç³»ç»Ÿ promptï¼ˆæ°¸ä¸æŒä¹…åŒ–åˆ°ä¼šè¯ï¼‰ |
| `HERMES_PREFILL_MESSAGES_FILE` | åŒ…å«åœ¨ API è°ƒç”¨æ—¶æ³¨å…¥çš„ä¸´æ—¶é¢„å¡«æ¶ˆæ¯çš„ JSON æ–‡ä»¶è·¯å¾„ã€‚ |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false`â€”â€”å…è®¸å·¥å…·è·å– localhost/ç§æœ‰ç½‘ç»œ URLã€‚gateway æ¨¡å¼ä¸‹é»˜è®¤å…³é—­ã€‚ |
| `HERMES_REDACT_SECRETS` | `true`/`false`â€”â€”æ§åˆ¶å·¥å…·è¾“å‡ºã€æ—¥å¿—å’ŒèŠå¤©å“åº”ä¸­çš„å¯†é’¥è„±æ•ï¼ˆé»˜è®¤ï¼š`true`ï¼‰ã€‚ |
| `HERMES_WRITE_SAFE_ROOT` | å¯é€‰ç›®å½•å‰ç¼€ï¼Œé™åˆ¶ `write_file`/`patch` å†™å…¥ï¼›è¶…å‡ºèŒƒå›´çš„è·¯å¾„éœ€è¦å®¡æ‰¹ã€‚æ”¯æŒå¤šä¸ªç›®å½•ï¼Œä½¿ç”¨ `os.pathsep` åˆ†éš”ï¼ˆUnix ä¸º `:`ï¼ŒWindows ä¸º `;`ï¼‰ã€‚ |
| `HERMES_DISABLE_LAZY_INSTALLS` | å®˜æ–¹ Docker é•œåƒä¸­è‡ªåŠ¨è®¾ç½®çš„å†…éƒ¨æ¡¥æ¥å˜é‡ï¼Œç”¨äºé˜»æ­¢è¿è¡Œæ—¶å°†ä¾èµ–å®‰è£…åˆ°ä¸å¯å˜çš„ `/opt/hermes` æ ‘ã€‚é¢å‘ç”¨æˆ·çš„ç­‰ä»·é…ç½®æ˜¯ `config.yaml` ä¸­çš„ `security.allow_lazy_installs: false`ï¼›ä¸è¦åœ¨ `.env` ä¸­æ‰‹åŠ¨è®¾ç½®æ­¤å˜é‡ã€‚ |
| `HERMES_DISABLE_FILE_STATE_GUARD` | è®¾ä¸º `1` å¯å…³é—­ `patch`/`write_file` ä¸Šçš„"æ–‡ä»¶è‡ªä¸Šæ¬¡è¯»å–åå·²æ›´æ”¹"ä¿æŠ¤ã€‚ |
| `HERMES_CORE_TOOLS` | è§„èŒƒæ ¸å¿ƒå·¥å…·åˆ—è¡¨çš„é€—å·åˆ†éš”è¦†ç›–ï¼ˆé«˜çº§ï¼›æå°‘éœ€è¦ï¼‰ã€‚ |
| `HERMES_BUNDLED_SKILLS` | å¯åŠ¨æ—¶åŠ è½½çš„å†…ç½®æŠ€èƒ½åˆ—è¡¨çš„é€—å·åˆ†éš”è¦†ç›–ã€‚ |
| `HERMES_OPTIONAL_SKILLS` | é¦–æ¬¡è¿è¡Œæ—¶è‡ªåŠ¨å®‰è£…çš„å¯é€‰æŠ€èƒ½åç§°é€—å·åˆ†éš”åˆ—è¡¨ã€‚ |
| `HERMES_DEBUG_INTERRUPT` | è®¾ä¸º `1` å¯å°†è¯¦ç»†çš„ä¸­æ–­/å–æ¶ˆè¿½è¸ªè®°å½•åˆ° `agent.log`ã€‚ |
| `HERMES_DUMP_REQUESTS` | å°† API è¯·æ±‚è½½è·è½¬å‚¨åˆ°æ—¥å¿—æ–‡ä»¶ï¼ˆ`true`/`false`ï¼‰ |
| `HERMES_DUMP_REQUEST_STDOUT` | å°† API è¯·æ±‚è½½è·è½¬å‚¨åˆ° stdout è€Œéæ—¥å¿—æ–‡ä»¶ã€‚ |
| `HERMES_OAUTH_TRACE` | è®¾ä¸º `1` å¯è®°å½• OAuth token äº¤æ¢å’Œåˆ·æ–°å°è¯•ã€‚åŒ…å«è„±æ•çš„æ—¶åºä¿¡æ¯ã€‚ |
| `HERMES_OAUTH_FILE` | è¦†ç›– OAuth å‡­è¯å­˜å‚¨è·¯å¾„ï¼ˆé»˜è®¤ï¼š`~/.hermes/auth.json`ï¼‰ã€‚ |
| `HERMES_AGENT_HELP_GUIDANCE` | ä¸ºè‡ªå®šä¹‰éƒ¨ç½²åœ¨ç³»ç»Ÿ prompt ä¸­è¿½åŠ é¢å¤–æŒ‡å¯¼æ–‡æœ¬ã€‚ |
| `HERMES_AGENT_LOGO` | è¦†ç›– CLI å¯åŠ¨æ—¶çš„ ASCII æ¨ªå¹… logoã€‚ |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | æ¯ä¸ª `delegate_task` æ‰¹æ¬¡çš„æœ€å¤§å¹¶è¡Œå­ agent æ•°ï¼ˆé»˜è®¤ï¼š`3`ï¼Œä¸‹é™ä¸º 1ï¼Œæ— ä¸Šé™ï¼‰ã€‚ä¹Ÿå¯é€šè¿‡ `config.yaml` ä¸­çš„ `delegation.max_concurrent_children` é…ç½®â€”â€”config å€¼ä¼˜å…ˆã€‚ |

## ç•Œé¢

| å˜é‡ | æè¿° |
|----------|-------------|
| `HERMES_TUI` | è®¾ä¸º `1` æ—¶å¯åŠ¨ [TUI](../user-guide/tui.md) è€Œéç»å…¸ CLIã€‚ç­‰åŒäºä¼ å…¥ `--tui`ã€‚ |
| `HERMES_TUI_DIR` | é¢„æ„å»º `ui-tui/` ç›®å½•çš„è·¯å¾„ï¼ˆå¿…é¡»åŒ…å« `dist/entry.js` å’Œå·²å¡«å……çš„ `node_modules`ï¼‰ã€‚ä¾›å‘è¡Œç‰ˆå’Œ Nix ä½¿ç”¨ä»¥è·³è¿‡é¦–æ¬¡å¯åŠ¨æ—¶çš„ `npm install`ã€‚ |
| `HERMES_TUI_RESUME` | å¯åŠ¨æ—¶æŒ‰ ID æ¢å¤ç‰¹å®š TUI ä¼šè¯ã€‚è®¾ç½®åï¼Œ`hermes --tui` è·³è¿‡åˆ›å»ºæ–°ä¼šè¯å¹¶æ¥ç»­æŒ‡å®šä¼šè¯â€”â€”é€‚ç”¨äºæ–­å¼€è¿æ¥æˆ–ç»ˆç«¯å´©æºƒåé‡æ–°è¿æ¥ã€‚ |
| `HERMES_TUI_THEME` | å¼ºåˆ¶ TUI é¢œè‰²ä¸»é¢˜ï¼š`light`ã€`dark` æˆ–åŸå§‹ 6 å­—ç¬¦èƒŒæ™¯åå…­è¿›åˆ¶ï¼ˆä¾‹å¦‚ `ffffff` æˆ– `1a1a2e`ï¼‰ã€‚æœªè®¾ç½®æ—¶ï¼ŒHermes ä½¿ç”¨ `COLORFGBG` å’Œç»ˆç«¯èƒŒæ™¯æŸ¥è¯¢è‡ªåŠ¨æ£€æµ‹ï¼›æ­¤å˜é‡è¦†ç›–ä¸è®¾ç½® `COLORFGBG` çš„ç»ˆç«¯ï¼ˆGhosttyã€Warpã€iTerm2 ç­‰ï¼‰ä¸Šçš„æ£€æµ‹ã€‚ |
| `HERMES_INFERENCE_MODEL` | ä¸º `hermes -z`/`hermes chat` å¼ºåˆ¶æŒ‡å®šæ¨¡å‹è€Œä¸ä¿®æ”¹ `config.yaml`ã€‚ä¸ `--provider` æ ‡å¿—é…åˆä½¿ç”¨ã€‚é€‚ç”¨äºéœ€è¦æ¯æ¬¡è¿è¡Œè¦†ç›–é»˜è®¤æ¨¡å‹çš„è„šæœ¬è°ƒç”¨è€…ï¼ˆsweeperã€CIã€æ‰¹é‡è¿è¡Œå™¨ï¼‰ã€‚ |

## ä¼šè¯è®¾ç½®

| å˜é‡ | æè¿° |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | ä¸æ´»åŠ¨ N åˆ†é’Ÿåé‡ç½®ä¼šè¯ï¼ˆé»˜è®¤ï¼š1440ï¼‰ |
| `SESSION_RESET_HOUR` | 24 å°æ—¶åˆ¶æ¯æ—¥é‡ç½®æ—¶é—´ï¼ˆé»˜è®¤ï¼š4 = å‡Œæ™¨ 4 ç‚¹ï¼‰ |
| `HERMES_SESSION_ID` | **è‡ªåŠ¨å¯¼å‡ºåˆ° Hermes ç”Ÿæˆçš„æ¯ä¸ªå·¥å…·å­è¿›ç¨‹**ï¼ˆ`terminal`ã€`execute_code`ã€æŒä¹… shellã€Docker/Singularity åç«¯ã€å§”æ‰˜å­ agent è¿è¡Œï¼‰ã€‚ç”± agent è®¾ç½®ä¸ºå½“å‰ä¼šè¯ IDï¼›ä»å·¥å…·è°ƒç”¨çš„ç”¨æˆ·è„šæœ¬å¯è¯»å–å®ƒï¼Œä»¥å°†å…¶è¾“å‡ºã€é¥æµ‹æˆ–å‰¯ä½œç”¨ä¸åŸå§‹ Hermes ä¼šè¯å…³è”ã€‚**ä¸åº”æ‰‹åŠ¨è®¾ç½®**â€”â€”ä»çˆ¶ shell è¦†ç›–ä»…åœ¨ agent è¿è¡Œå¤–ç”Ÿæ•ˆï¼Œä¸” agent å¯åŠ¨ä¼šè¯æ—¶ä¼šè¢«è¦†ç›–ã€‚ |

## ä¸Šä¸‹æ–‡å‹ç¼©ï¼ˆä»… config.yamlï¼‰

ä¸Šä¸‹æ–‡å‹ç¼©å®Œå…¨é€šè¿‡ `config.yaml` é…ç½®â€”â€”æ²¡æœ‰å¯¹åº”çš„ç¯å¢ƒå˜é‡ã€‚é˜ˆå€¼è®¾ç½®ä½äº `compression:` å—ï¼Œæ‘˜è¦æ¨¡å‹/æä¾›å•†ä½äº `auxiliary.compression:` ä¸‹ã€‚

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # fraction of threshold to preserve as recent tail
  protect_last_n: 20         # minimum recent messages to keep uncompressed
```

:::info æ—§ç‰ˆè¿ç§»
åŒ…å« `compression.summary_model`ã€`compression.summary_provider` å’Œ `compression.summary_base_url` çš„æ—§ç‰ˆé…ç½®åœ¨é¦–æ¬¡åŠ è½½æ—¶è‡ªåŠ¨è¿ç§»åˆ° `auxiliary.compression.*`ã€‚
:::

## è¾…åŠ©ä»»åŠ¡è¦†ç›–

| å˜é‡ | æè¿° |
|----------|-------------|
| `AUXILIARY_VISION_PROVIDER` | è¦†ç›–è§†è§‰ä»»åŠ¡çš„æä¾›å•† |
| `AUXILIARY_VISION_MODEL` | è¦†ç›–è§†è§‰ä»»åŠ¡çš„æ¨¡å‹ |
| `AUXILIARY_VISION_BASE_URL` | è§†è§‰ä»»åŠ¡çš„ç›´æ¥ OpenAI å…¼å®¹ç«¯ç‚¹ |
| `AUXILIARY_VISION_API_KEY` | ä¸ `AUXILIARY_VISION_BASE_URL` é…å¯¹çš„ API å¯†é’¥ |
| `AUXILIARY_WEB_EXTRACT_PROVIDER` | è¦†ç›–ç½‘é¡µæå–/æ‘˜è¦çš„æä¾›å•† |
| `AUXILIARY_WEB_EXTRACT_MODEL` | è¦†ç›–ç½‘é¡µæå–/æ‘˜è¦çš„æ¨¡å‹ |
| `AUXILIARY_WEB_EXTRACT_BASE_URL` | ç½‘é¡µæå–/æ‘˜è¦çš„ç›´æ¥ OpenAI å…¼å®¹ç«¯ç‚¹ |
| `AUXILIARY_WEB_EXTRACT_API_KEY` | ä¸ `AUXILIARY_WEB_EXTRACT_BASE_URL` é…å¯¹çš„ API å¯†é’¥ |

å¯¹äºç‰¹å®šä»»åŠ¡çš„ç›´æ¥ç«¯ç‚¹ï¼ŒHermes ä½¿ç”¨è¯¥ä»»åŠ¡é…ç½®çš„ API å¯†é’¥æˆ– `OPENAI_API_KEY`ã€‚ä¸ä¼šä¸ºè¿™äº›è‡ªå®šä¹‰ç«¯ç‚¹å¤ç”¨ `OPENROUTER_API_KEY`ã€‚

## å›é€€æä¾›å•†ï¼ˆä»… config.yamlï¼‰

ä¸»æ¨¡å‹å›é€€é“¾å®Œå…¨é€šè¿‡ `config.yaml` é…ç½®â€”â€”æ²¡æœ‰å¯¹åº”çš„ç¯å¢ƒå˜é‡ã€‚åœ¨é¡¶å±‚æ·»åŠ åŒ…å« `provider` å’Œ `model` é”®çš„ `fallback_providers` åˆ—è¡¨ï¼Œä»¥åœ¨ä¸»æ¨¡å‹é‡åˆ°é”™è¯¯æ—¶å¯ç”¨è‡ªåŠ¨æ•…éšœè½¬ç§»ã€‚

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

æ—§ç‰ˆé¡¶å±‚ `fallback_model` å•æä¾›å•†æ ¼å¼ä»å¯å‘åå…¼å®¹è¯»å–ï¼Œä½†æ–°é…ç½®åº”ä½¿ç”¨ `fallback_providers`ã€‚

è¯¦è§ [å›é€€æä¾›å•†](/user-guide/features/fallback-providers)ã€‚

## æä¾›å•†è·¯ç”±ï¼ˆä»… config.yamlï¼‰

è¿™äº›é…ç½®å†™å…¥ `~/.hermes/config.yaml` çš„ `provider_routing` éƒ¨åˆ†ï¼š

| é”® | æè¿° |
|-----|-------------|
| `sort` | æ’åºæä¾›å•†ï¼š`"price"`ï¼ˆé»˜è®¤ï¼‰ã€`"throughput"` æˆ– `"latency"` |
| `only` | å…è®¸çš„æä¾›å•† slug åˆ—è¡¨ï¼ˆä¾‹å¦‚ `["anthropic", "google"]`ï¼‰ |
| `ignore` | è·³è¿‡çš„æä¾›å•† slug åˆ—è¡¨ |
| `order` | æŒ‰é¡ºåºå°è¯•çš„æä¾›å•† slug åˆ—è¡¨ |
| `require_parameters` | ä»…ä½¿ç”¨æ”¯æŒæ‰€æœ‰è¯·æ±‚å‚æ•°çš„æä¾›å•†ï¼ˆ`true`/`false`ï¼‰ |
| `data_collection` | `"allow"`ï¼ˆé»˜è®¤ï¼‰æˆ– `"deny"` ä»¥æ’é™¤å­˜å‚¨æ•°æ®çš„æä¾›å•† |

:::tip
ä½¿ç”¨ `hermes config set` è®¾ç½®ç¯å¢ƒå˜é‡â€”â€”å®ƒä¼šè‡ªåŠ¨å°†å…¶ä¿å­˜åˆ°æ­£ç¡®çš„æ–‡ä»¶ï¼ˆå¯†é’¥ä¿å­˜åˆ° `.env`ï¼Œå…¶ä»–æ‰€æœ‰å†…å®¹ä¿å­˜åˆ° `config.yaml`ï¼‰ã€‚
:::