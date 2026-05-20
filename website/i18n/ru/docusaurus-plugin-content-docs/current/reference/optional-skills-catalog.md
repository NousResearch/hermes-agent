---
sidebar_position: 9
title: "Каталог опциональных навыков"
description: "Официальные опциональные навыки, поставляемые с hermes-agent, — установка через `hermes skills install official/<category>/<skill>`"
---

# Каталог опциональных навыков

Опциональные навыки поставляются вместе с `hermes-agent` в каталоге `optional-skills/`, но **не активны по умолчанию**. Устанавливайте их явно:

```bash
hermes skills install official/<category>/<skill>
```

Например:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

Каждый навык ниже ведёт на отдельную страницу с полным определением, настройкой и примерами использования.

Чтобы удалить навык:

```bash
hermes skills uninstall <skill-name>
```

## Автономные ИИ-агенты

| Навык | Описание |
|-------|-------------|
| [**blackbox**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | Делегирует задачи по коду CLI-агенту Blackbox AI. Много-модельный агент со встроенным judge, который прогоняет задачу через несколько LLM и выбирает лучший результат. Требует CLI `blackbox` и API-ключ Blackbox AI. |
| [**honcho**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | Настройка и использование памяти Honcho с Hermes: межсеансовое моделирование пользователя, изоляция участников по профилям, конфигурация наблюдения, диалектическое рассуждение, сводки сеансов и контроль бюджета контекста. Подходит для настройки Honcho, устранения неполадок с памятью, управления профилями через участников Honcho и настройки наблюдения, воспоминаний и диалектики. |

## Блокчейн

| Навык | Описание |
|-------|-------------|
| [**evm**](/docs/user-guide/skills/optional/blockchain/blockchain-evm) | Чтение EVM только на просмотр: кошельки, токены, газ на 8 сетях. |
| [**hyperliquid**](/docs/user-guide/skills/optional/blockchain/blockchain-hyperliquid) | Рыночные данные Hyperliquid, история аккаунта, просмотр сделок. |
| [**solana**](/docs/user-guide/skills/optional/blockchain/blockchain-solana) | Запрос данных блокчейна Solana с ценами в USD: балансы кошельков, портфели токенов с оценкой, детали транзакций, NFT, обнаружение китов и статистика сети. Использует Solana RPC + CoinGecko. API-ключ не нужен. |

## Коммуникация

| Навык | Описание |
|-------|-------------|
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | Структурированная схема принятия решений для технических предложений и анализа компромиссов. Когда у пользователя есть выбор между несколькими подходами (архитектура, инструмент, рефакторинг, миграция), этот навык выдаёт формат 1-3-1: одну формулировку проблемы, три варианта с плюсами и минусами и одну конкретную рекомендацию с определением готовности и планом внедрения. |

## Творчество

| Навык | Описание |
|-------|-------------|
| [**blender-mcp**](/docs/user-guide/skills/optional/creative/creative-blender-mcp) | Управление Blender напрямую из Hermes через подключение по сокету к аддону blender-mcp. Создавайте 3D-объекты, материалы, анимации и запускайте любой Python-код Blender (`bpy`). Используйте, когда пользователю нужно создать или изменить что-либо в Blender. |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | Генерирует плоские минималистичные SVG-диаграммы с поддержкой светлой и тёмной темы как отдельные HTML-файлы, используя единый образовательный визуальный язык с 9 семантическими цветовыми шкалами, типографикой в sentence case и автоматическим тёмным режимом. Лучше всего подходит для образовательных и нотационных схем. |
| [**hyperframes**](/docs/user-guide/skills/optional/creative/creative-hyperframes) | Создаёт HTML-композиции для видео, анимированные титры, социальные оверлеи, ролики talking head с подписями, аудио-реактивные визуализации и шейдерные переходы с помощью HyperFrames. HTML — источник истины для видео. Используйте, когда пользователю нужно сделать видео. |
| [**kanban-video-orchestrator**](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | Планирование, запуск и мониторинг многоагентного конвейера видеопроизводства на базе Hermes Kanban. Используйте, когда пользователю нужно сделать любое видео — художественный фильм, продуктовый или маркетинговый ролик, музыкальный клип, объясняющее видео, ASCII/terminal art, абстрактный генеративный ролик и т. д. |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | Генерация настоящих мемов: выбирает шаблон и накладывает текст через Pillow. На выходе получается реальный файл `.png`. |

## DevOps

| Навык | Описание |
|-------|-------------|
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-cli) | Запускает 150+ AI-приложений через inference.sh CLI (`infsh`) — генерация изображений, видео, LLM, поиск, 3D, автоматизация соцсетей. Использует терминальный инструмент. Триггеры: inference.sh, infsh, ai apps, flux, veo, image generation, video generation и т. д. |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | Управление контейнерами, образами, томами, сетями и стеком Compose — жизненный цикл, отладка, очистка и оптимизация Dockerfile. |
| [**pinggy-tunnel**](/docs/user-guide/skills/optional/devops/devops-pinggy-tunnel) | Туннели localhost по SSH через Pinggy без установки. |
| [**watchers**](/docs/user-guide/skills/optional/devops/devops-watchers) | Опрос RSS, JSON API и GitHub с дедупликацией по watermark. |

## Dogfood

| Навык | Описание |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | Ролевая модель самого сложного, технологически неуступчивого пользователя вашего продукта. Исследуйте приложение в образе этой персоны, найдите все UX-проблемы, затем пропустите жалобы через прагматический фильтр, чтобы отделить реальные проблемы от шума. Создаёт практичные задачи. |

## Почта

| Навык | Описание |
|-------|-------------|
| [**agentmail**](/docs/user-guide/skills/optional/email/email-agentmail) | Даёт агенту собственный почтовый ящик через AgentMail. Отправляйте, получайте и управляйте почтой автономно с помощью адресов, принадлежащих агенту (например, hermes-agent@agentmail.to). |

## Финансы

| Навык | Описание |
|-------|-------------|
| [**3-statement-model**](/docs/user-guide/skills/optional/finance/finance-3-statement-model) | Строит полноценные модели формата 3-statement (IS, BS, CF) в Excel с графиками оборотного капитала, сквозным расчётом износа и амортизации, графиком долга и связками, которые сводят денежный поток и нераспределённую прибыль. Пара к `excel-author`. |
| [**comps-analysis**](/docs/user-guide/skills/optional/finance/finance-comps-analysis) | Строит сравнительный анализ сопоставимых компаний в Excel: операционные метрики, мультипликаторы оценки и статистический бенчмарк по набору аналогов. Пара к `excel-author`. Подходит для публичной оценки компаний, ценообразования при IPO, отраслевого сравнения и поиска выбросов. |
| [**dcf-model**](/docs/user-guide/skills/optional/finance/finance-dcf-model) | Строит DCF-модели институционального уровня в Excel: прогнозы выручки, расчёт FCF, WACC, терминальная стоимость, медвежий/базовый/бычий сценарии, таблицы чувствительности 5x5. Пара к `excel-author`. Подходит для фундаментальной оценки акций по внутренней стоимости. |
| [**excel-author**](/docs/user-guide/skills/optional/finance/finance-excel-author) | Собирает аудируемые Excel-книги без GUI через `openpyxl` — цветовые соглашения blue/black/green, формулы вместо жёстко заданных значений, именованные диапазоны, проверка баланса и таблицы чувствительности. Подходит для финансовых моделей, проверочных отчётов и сверок. |
| [**lbo-model**](/docs/user-guide/skills/optional/finance/finance-lbo-model) | Строит LBO-модели в Excel: источники и использование средств, график долга, направление свободного денежного потока на погашение долга, мультипликатор выхода, чувствительность IRR/MOIC. Пара к `excel-author`. Подходит для отбора private equity-сделок, оценки спонсорских сценариев и иллюстративных LBO в презентациях. |
| [**merger-model**](/docs/user-guide/skills/optional/finance/finance-merger-model) | Строит модели аккреции и размывания (merger) в Excel: pro forma P&L, синергия, структура финансирования, влияние на EPS. Пара к `excel-author`. Подходит для M&A-презентаций, материалов для совета директоров и оценки сделки. |
| [**pptx-author**](/docs/user-guide/skills/optional/finance/finance-pptx-author) | Собирает презентации PowerPoint без GUI через `python-pptx`. Пара к `excel-author` для презентаций, где каждая цифра должна быть привязана к ячейке рабочей книги. Подходит для питч-презентаций, материалов для инвестиционного комитета и заметок по отчётности. |
| [**stocks**](/docs/user-guide/skills/optional/finance/finance-stocks) | Котировки акций, история, поиск, сравнение и криптовалюта через Yahoo. |

## Здоровье

| Навык | Описание |
|-------|-------------|
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | Планировщик тренировок и трекер питания. Ищите 690+ упражнений по мышцам, оборудованию или категории через wger. Смотрите макросы и калории для 380 000+ продуктов через USDA FoodData Central. Считайте BMI, TDEE, one-rep max, macro splits и body composition. |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | Подключается к запущенному экземпляру NeuroSkill и учитывает реальное когнитивное и эмоциональное состояние пользователя (фокус, расслабление, настроение, когнитивную нагрузку, сонливость, пульс, HRV, staging сна и 40+ производных EXG-score) в ответах. |

## MCP

| Навык | Описание |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | Создание, тестирование, инспекция, установка и деплой MCP-серверов с FastMCP на Python. Подходит для создания нового MCP-сервера, обёртки API или БД в MCP-инструменты, экспонирования ресурсов или промптов, а также подготовки FastMCP-сервера для Claude Code, Cursor и других клиентов. |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | Используйте CLI `mcporter`, чтобы перечислять, настраивать, авторизовывать и вызывать MCP-серверы/инструменты напрямую (HTTP или stdio), включая ad-hoc серверы, редактирование конфигурации и генерацию CLI/типов. |

## Миграция

| Навык | Описание |
|-------|-------------|
| [**openclaw-migration**](/docs/user-guide/skills/optional/migration/migration-openclaw-migration) | Мигрирует пользовательскую настройку OpenClaw в Hermes Agent. Импортирует совместимые с Hermes памяти, SOUL.md, списки разрешённых команд, пользовательские навыки и выбранные ассеты рабочей области из `~/.openclaw`, затем подробно сообщает, что именно не удалось перенести. |

## MLOps

| Навык | Описание |
|-------|-------------|
| [**huggingface-accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | Самый простой API для распределённого обучения. 4 строки, чтобы добавить distributed support в любой PyTorch-скрипт. Единый API для DeepSpeed/FSDP/Megatron/DDP. Автоматическое размещение на устройстве, mixed precision (FP16/BF16/FP8). Интерактивная настройка, один запуск командой. |
| [**axolotl**](/docs/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: YAML-LLM fine-tuning (LoRA, DPO, GRPO). |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | Открытая векторная БД для AI-приложений. Хранение embeddings и метаданных, векторный и полнотекстовый поиск, фильтрация по метаданным. Простой API из 4 функций. Масштабируется от ноутбука до production-кластера. Подходит для semantic search, RAG. |
| [**clip**](/docs/user-guide/skills/optional/mlops/mlops-clip) | Модель OpenAI, соединяющая зрение и язык. Поддерживает zero-shot классификацию изображений, сопоставление изображение-текст и cross-modal retrieval. Обучена на 400 млн пар изображение-текст. Подходит для image search, модерации контента и задач vision-language. |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | Библиотека Facebook для эффективного поиска похожих векторов и кластеризации плотных embeddings. Поддерживает миллиарды векторов, GPU-ускорение и разные типы индексов (Flat, IVF, HNSW). Подходит для быстрого k-NN search, крупномасштабного vector retrieval и т. п. |
| [**optimizing-attention-flash**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | Оптимизирует attention в трансформерах с помощью Flash Attention и даёт ускорение в 2–4 раза при снижении потребления памяти на 10–20 раз. Используйте, если обучаете или запускаете трансформеры с длинными последовательностями (>512 tokens), упираетесь в GPU memory или хотите более быстрый inference. |
| [**guidance**](/docs/user-guide/skills/optional/mlops/mlops-guidance) | Управляет выводом LLM через regex и грамматики, гарантирует корректный JSON/XML/код, навязывает структурированные форматы и строит многошаговые workflows с Guidance — framework constrained generation от Microsoft Research. |
| [**huggingface-tokenizers**](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | Быстрые токенизаторы, оптимизированные для исследований и продакшена. Rust-реализация токенизирует 1 ГБ менее чем за 20 секунд. Поддерживает алгоритмы BPE, WordPiece и Unigram. Обучайте собственные словари, отслеживайте выравнивания, обрабатывайте padding/truncation. |
| [**instructor**](/docs/user-guide/skills/optional/mlops/mlops-instructor) | Извлекает структурированные данные из ответов LLM с Pydantic-валидацией, автоматически повторяет неудачные извлечения, парсит сложный JSON с типовой безопасностью и стримит частичные результаты с Instructor — боевой библиотекой structured output. |
| [**lambda-labs-gpu-cloud**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | Выделенные и on-demand GPU-экземпляры в облаке для обучения и inference. Используйте, если нужны dedicated GPU с простым SSH-доступом, persistent filesystem или производительные multi-node-кластеры для крупномасштабного обучения. |
| [**llava**](/docs/user-guide/skills/optional/mlops/mlops-llava) | Large Language and Vision Assistant. Поддерживает визуальное instruction tuning и общение по изображениям. Сочетает vision encoder CLIP с языковыми моделями Vicuna/LLaMA. Поддерживает multi-turn image chat, visual QA и instruction-following. |
| [**modal-serverless-gpu**](/docs/user-guide/skills/optional/mlops/mlops-modal) | Serverless GPU cloud platform для запуска ML-нагрузок. Используйте, если нужен on-demand GPU без администрирования инфраструктуры, деплой моделей как API или batch jobs с автоматическим масштабированием. |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | GPU-ускоренная очистка данных для обучения LLM. Поддерживает text/image/video/audio. Есть fuzzy deduplication (в 16 раз быстрее), quality filtering (30+ heuristics), semantic deduplication, PII redaction, NSFW detection. Масштабируется по нескольким GPU. |
| [**outlines**](/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) | Outlines: структурированная генерация JSON/regex/Pydantic для LLM. |
| [**peft-fine-tuning**](/docs/user-guide/skills/optional/mlops/mlops-peft) | Parameter-efficient fine-tuning для LLM с LoRA, QLoRA и 25+ методами. Подходит, если вы дообучаете большие модели (7B–70B) при ограниченной GPU-памяти, если нужно обучать менее 1% параметров с минимальной потерей качества или если требуется multi-adapter setup. |
| [**pinecone**](/docs/user-guide/skills/optional/mlops/mlops-pinecone) | Управляемая векторная БД для production AI-приложений. Полностью managed, auto-scaling, поддержка hybrid search (dense + sparse), фильтрации по метаданным и namespaces. Низкая задержка (&lt;100ms p95). Подходит для production RAG, recommendation systems и semantic search. |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | Экспертные рекомендации по Fully Sharded Data Parallel training с PyTorch FSDP: шардирование параметров, mixed precision, CPU offloading, FSDP2. |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | Высокоуровневый PyTorch-фреймворк с классом Trainer, автоматическим распределённым обучением (DDP/FSDP/DeepSpeed), системой callback'ов и минимальным boilerplate. Масштабируется от ноутбука до суперкомпьютера с тем же кодом. |
| [**qdrant-vector-search**](/docs/user-guide/skills/optional/mlops/mlops-qdrant) | Высокопроизводительный engine векторного similarity search для RAG и semantic search. Подходит для production-систем, где нужен быстрый nearest neighbor search, hybrid search с фильтрацией и масштабируемое хранение векторов на базе Rust. |
| [**sparse-autoencoder-training**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | Рекомендации по обучению и анализу Sparse Autoencoders (SAEs) через SAELens для разложения активаций нейросети в интерпретируемые признаки. Подходит для поиска интерпретируемых признаков, анализа superposition и изучения представлений. |
| [**simpo-training**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | Simple Preference Optimization для выравнивания LLM. Альтернатива DPO без reference model с лучшей эффективностью (+6.4 points на AlpacaEval 2.0). Подходит для preference alignment, когда нужен более простой и экономичный путь, чем DPO. |
| [**slime-rl-training**](/docs/user-guide/skills/optional/mlops/mlops-slime) | Рекомендации по post-training LLM с RL через slime — фреймворк на базе Megatron+SGLang. Подходит для обучения моделей GLM, создания пользовательских workflows генерации данных и tight integration с Megatron-LM для масштабирования RL. |
| [**stable-diffusion-image-generation**](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) | Современная генерация изображений по тексту через Stable Diffusion и HuggingFace Diffusers. Подходит для text-to-image, image-to-image, inpainting и построения собственных diffusion pipeline. |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | Оптимизирует inference LLM с помощью NVIDIA TensorRT для максимального throughput и минимальной задержки. Подходит для production на NVIDIA GPU (A100/H100), когда нужен inference в 10–100 раз быстрее, чем у PyTorch, или для serving моделей с quantization. |
| [**distributed-llm-pretraining-torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | PyTorch-native distributed LLM pretraining через torchtitan с 4D-параллелизмом (FSDP2, TP, PP, CP). Подходит для pretraining Llama 3.1, DeepSeek V3 или собственных моделей в масштабе от 8 до 512+ GPU с Float8, `torch.compile` и распределённым планированием. |
| [**fine-tuning-with-trl**](/docs/user-guide/skills/optional/mlops/mlops-training-trl-fine-tuning) | TRL: SFT, DPO, PPO, GRPO, reward modeling для LLM RLHF. |
| [**unsloth**](/docs/user-guide/skills/optional/mlops/mlops-training-unsloth) | Unsloth: до 2–5 раз быстрее LoRA/QLoRA fine-tuning, меньше VRAM. |
| [**whisper**](/docs/user-guide/skills/optional/mlops/mlops-whisper) | Универсальная модель распознавания речи от OpenAI. Поддерживает 99 языков, транскрибацию, перевод на английский и определение языка. Шесть размеров модели — от tiny (39M params) до large (1550M params). Подходит для speech-to-text, подкастов, субтитров и других задач. |

## Продуктивность

| Навык | Описание |
|-------|-------------|
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | Интеграция с Canvas LMS: получает курсы и задания, на которые вы записаны, используя API token authentication. |
| [**here.now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | Публикует статические сайты на `{slug}.here.now` и хранит приватные файлы в cloud Drives для передачи между агентами. |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | Система флэш-карт с интервальным повторением. Создаёт карточки из фактов или текста, позволяет общаться с карточками свободным текстом, который агент сам оценивает, генерирует викторины из YouTube-транскриптов, показывает карточки по адаптивному расписанию и поддерживает экспорт/импорт. |
| [**shop-app**](/docs/user-guide/skills/optional/productivity/productivity-shop-app) | Shop.app: поиск товаров, отслеживание заказов, возвраты и повторные заказы. |
| [**shopify**](/docs/user-guide/skills/optional/productivity/productivity-shopify) | Shopify Admin и Storefront GraphQL API через curl. Товары, заказы, клиенты, остатки, metafields. |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | API SiYuan Note для поиска, чтения, создания и управления блоками и документами в самохостируемой базе знаний через curl. |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | Даёт Hermes телефонные возможности без изменений в core tools. Подготовьте и закрепите номер Twilio, отправляйте и принимайте SMS/MMS, совершайте прямые звонки и размещайте исходящие AI-звонки через Bland.ai или Vapi. |

## Исследования

| Навык | Описание |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | Вход в более чем 400 bioinformatics skills из bioSkills и ClawBio. Охватывает genomics, transcriptomics, single-cell, variant calling, pharmacogenomics, metagenomics, structural biology и многое другое. Подтягивает доменно-специфические справочные материалы. |
| [**darwinian-evolver**](/docs/user-guide/skills/optional/research/research-darwinian-evolver) | Эволюция prompts/regex/SQL/code через цикл эволюции Imbue. |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | Пассивная разведка домена на Python stdlib: поиск поддоменов, проверка SSL-сертификатов, WHOIS, DNS-записи, проверка доступности домена и массовый анализ нескольких доменов. API-ключи не нужны. |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | Помощник фармацевтических исследований для workflows drug discovery. Ищет биоактивные соединения в ChEMBL, вычисляет drug-likeness (Lipinski Ro5, QED, TPSA, synthetic accessibility), проверяет drug-drug interactions через OpenFDA, интерпретирует ADMET. |
| [**duckduckgo-search**](/docs/user-guide/skills/optional/research/research-duckduckgo-search) | Бесплатный веб-поиск через DuckDuckGo — текст, новости, изображения, видео. API-ключ не нужен. Если установлен `ddgs`, предпочитайте его; Python-библиотеку DDGS используйте только после проверки, что `ddgs` доступен в текущем runtime. |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | Индексирует кодовую базу через GitNexus и показывает интерактивный knowledge graph через web UI + Cloudflare tunnel. |
| [**osint-investigation**](/docs/user-guide/skills/optional/research/research-osint-investigation) | Фреймворк OSINT-расследований по открытым данным: SEC EDGAR, USAspending, Senate lobbying, OFAC sanctions, ICIJ offshore leaks, NYC ACRIS, OpenCorporates, CourtListener, Wayback и другие источники. |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | Опциональный vendor skill для Parallel CLI — agent-native веб-поиск, extraction, deep research, enrichment, FindAll и мониторинг. Предпочитайте JSON-вывод и неинтерактивные сценарии. |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | Поиск по личным базам знаний, заметкам, документам и стенограммам встреч локально через `qmd` — гибридный retrieval engine с BM25, vector search и LLM reranking. Поддерживает CLI и MCP. |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | Веб-скрейпинг через Scrapling — HTTP fetching, stealth browser automation, обход Cloudflare и spider crawling через CLI и Python. |
| [**searxng-search**](/docs/user-guide/skills/optional/research/research-searxng-search) | Бесплатный метапоиск через SearXNG — агрегирует результаты более чем из 70 search engines. Можно использовать свой self-hosted instance или публичный. API-ключ не нужен. Автоматически подхватывается, когда web search toolset недоступен. |

## Безопасность

| Навык | Описание |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | Настройка и использование `1Password CLI` (`op`). Подходит для установки CLI, интеграции с настольным приложением, входа в систему и чтения/подстановки секретов для команд. |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | Расследование supply chain, восстановление доказательств и форензика для GitHub-репозиториев. Охватывает восстановление удалённых коммитов, обнаружение force-push, извлечение IOC, сбор улик из нескольких источников, формирование и проверку гипотез и многое другое. |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | OSINT-поиск username в более чем 400 социальных сетях. Помогает искать аккаунты по имени пользователя. |

## Разработка ПО

| Навык | Описание |
|-------|-------------|
| [**rest-graphql-debug**](/docs/user-guide/skills/optional/software-development/software-development-rest-graphql-debug) | Отладка REST/GraphQL API: коды статуса, аутентификация, схемы, воспроизведение проблем. |

## Веб-разработка

| Навык | Описание |
|-------|-------------|
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | Встраивает `alibaba/page-agent` в собственное веб-приложение — встраиваемый JavaScript GUI-агент, который поставляется как один тег `<script>` или npm-пакет и позволяет конечным пользователям управлять UI на естественном языке («нажмите кнопку входа, заполните имя пользователя…»). |

---

## Вклад в опциональные навыки

Чтобы добавить в репозиторий новый опциональный навык:

1. Создайте каталог в `optional-skills/<category>/<skill-name>/`.
2. Добавьте `SKILL.md` со стандартным frontmatter (name, description, version, author).
3. Добавьте сопровождающие файлы в подкаталоги `references/`, `templates/` или `scripts/`.
4. Отправьте pull request — после мержа навык появится в этом каталоге и получит собственную docs-страницу.
