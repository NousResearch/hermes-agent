import type { TranslationOverride } from '@hermes/shared/i18n'
import type { Translations } from '../types'

export const extensionsZh: TranslationOverride<Pick<Translations, 'achievements' | 'kanban'>> = {
  achievements: {
    hero: {
      kicker: 'AI 成就猎人',
      title: 'Hermes 成就',
      subtitle:
        '根据真实会话历史解锁 Hermes 徽章。已知但尚未达成的成就会显示为「已发现」；秘密成就在首次检测到相关行为前保持隐藏。',
      scan_subtitle: '正在扫描 Hermes 会话历史。在历史记录较多时，首次扫描可能需要 5–10 秒。'
    },
    actions: {
      rescan: '重新扫描'
    },
    stats: {
      unlocked: '已解锁',
      unlocked_hint: '获得的徽章',
      discovered: '已发现',
      discovered_hint: '已知，但尚未解锁',
      secrets: '秘密',
      secrets_hint: '在首次信号出现前保持隐藏',
      highest_tier: '最高等级',
      highest_tier_hint: '青铜 → 白银 → 黄金 → 钻石 → 奥林匹斯',
      latest: '最新',
      latest_hint_empty: '继续使用 Hermes 即可解锁',
      none_yet: '暂无'
    },
    state: {
      unlocked: '已解锁',
      discovered: '已发现',
      secret: '秘密'
    },
    tier: {
      target: '目标 {tier}',
      hidden: '隐藏',
      complete: '已完成',
      objective: '目标'
    },
    progress: {
      hidden: '隐藏'
    },
    scan: {
      building_headline: '正在构建成就档案…',
      building_detail: '正在读取会话、工具调用、模型元数据和解锁状态。',
      starting_headline: '正在开始成就扫描…',
      progress_detail: '已扫描 {scanned} / {total} 个会话 · {pct}%。随着更多历史流入，徽章会陆续解锁。',
      idle_detail: '正在读取会话、工具调用、模型元数据和解锁状态。徽章解锁后将在此显示。'
    },
    guide: {
      tiers_header: '等级',
      secret_header: '秘密成就',
      secret_body: '秘密成就会隐藏其确切触发条件。一旦 Hermes 检测到相关信号，卡片将变为「已发现」并显示其要求。',
      scan_status_header: '扫描状态',
      scan_status_body: 'Hermes 正在对本地历史进行一次扫描，之后卡片会自动出现。即使这需要几秒钟，也没有卡住。',
      what_scanned_header: '扫描内容',
      what_scanned_body: '会话、工具调用、模型元数据、错误、成就和本地解锁状态。'
    },
    card: {
      share_title: '分享此成就',
      share_label: '分享 {name}',
      share_text: '分享',
      how_to_reveal: '如何揭示',
      what_counts: '解锁条件',
      evidence_label: '证据',
      evidence_session_fallback: '会话',
      no_evidence: '暂无证据'
    },
    latest: {
      header: '最近解锁'
    },
    empty: {
      no_secrets_header: '本次扫描未发现仍处于隐藏状态的秘密成就。',
      no_secrets_body:
        '提示：秘密成就通常源于异常故障或高级操作，例如端口冲突、权限阻止、缺少环境变量、YAML 错误、Docker 冲突、使用回滚或检查点、缓存命中，或在连续报错后成功完成修复。'
    },
    filters: {
      all_categories: '全部',
      visibility_all: '全部',
      visibility_unlocked: '已解锁',
      visibility_discovered: '已发现',
      visibility_secret: '秘密'
    },
    share: {
      dialog_label: '分享成就',
      header: '分享：{name}',
      close: '关闭',
      rendering: '渲染中…',
      card_alt: '{name} 分享卡片',
      error_generic: '发生错误。',
      x_title: '在 X 中打开预填好的帖子',
      x_button: '在 X 上分享',
      copy_title: '复制图片以粘贴到你的帖子中',
      copy_button: '复制图片',
      copied: '已复制 ✓',
      download_button: '下载 PNG',
      hint: '「在 X 上分享」会在新标签页中打开预填好的帖子。如果想附上 1200×630 的徽章，请先点击「复制图片」—— X 允许你直接粘贴到推文编辑器中。「下载 PNG」会将文件保存下来，可在任意位置使用。',
      clipboard_unsupported: '此浏览器不支持将图片复制到剪贴板，请改用「下载」。',
      tweet_text: '我刚刚在 Hermes Agent 中解锁了 {tier_part}「{name}」☤',
      tier_part: '{tier}等级 ',
      tier_suffix: '等级',
      unlocked_stamp: '已解锁'
    },
    secretDefinition: {
      name: '???',
      description: '秘密成就：在 Hermes 从会话历史中检测到第一个相关行为前保持隐藏。'
    },
    categories: {
      agent_autonomy: 'Agent 自主能力',
      debugging_chaos: '调试风暴',
      vibe_coding: '氛围编程',
      hermes_native: 'Hermes 原生能力',
      research_web: '研究与 Web',
      tool_mastery: '工具精通',
      model_lore: '模型见闻',
      lifestyle: '使用习惯'
    },
    tierNames: {
      copper: '青铜',
      silver: '白银',
      gold: '黄金',
      diamond: '钻石',
      olympian: '奥林匹斯'
    },
    criteria: {
      secret:
        '秘密成就：在 Hermes 检测到首个匹配信号前隐藏确切要求。继续在调试、工具、记忆、技能、插件和模型工作流中使用 Hermes 即可揭示。',
      threshold: '要求：{metric}。等级阶梯：{ladder}。',
      requirements: '要求：{requirements}。',
      default: '要求：完成匹配的 Hermes 行为。',
      separator: '；'
    },
    metrics: {
      max_tool_calls_in_session: '单个会话中的工具调用数',
      total_tool_calls: '累计 Hermes 工具调用数',
      max_distinct_tools_in_session: '单个会话使用的不同 Hermes 工具数',
      max_terminal_calls_in_session: '单个会话中的终端调用数',
      max_file_tool_calls_in_session: '单个会话中的文件、搜索和补丁调用数',
      max_web_browser_calls_in_session: '单个会话中的 Web 搜索、提取或浏览器调用数',
      total_delegate_calls: '累计 delegate_task 调用数',
      total_process_calls: '累计后台进程操作数',
      total_cron_calls: '累计定时任务操作数',
      total_errors: '观察到的错误、失败或回溯消息数',
      traceback_events: '回溯或异常出现次数',
      log_read_events: '日志检查次数',
      port_conflict_events: '开发服务器端口冲突次数',
      permission_denied_events: '权限被拒绝错误次数',
      install_error_events: '软件包安装失败次数',
      install_success_events: '软件包处理后的成功安装次数',
      restart_after_error_events: '错误集中出现后的重启或重载次数',
      env_var_error_events: '缺少认证、配置或环境变量的事件数',
      yaml_error_events: 'YAML 或配置解析事故数',
      docker_conflict_events: 'Docker 或容器名称冲突数',
      max_messages_in_session: '单个会话中的消息数',
      max_files_touched_in_session: '单个会话触及的文件数',
      frontend_activity_events: '前端、CSS、SVG 或 React 活动次数',
      git_events: 'Git 工作流命令数',
      css_activity_events: 'CSS、样式、Tailwind 或 className 活动次数',
      tiny_patch_after_errors_events: '错误集中出现后的微小修复次数',
      skill_events: 'Hermes 技能提及或工具使用次数',
      skill_manage_events: 'skill_manage 创建、修改或删除操作数',
      memory_events: '记忆或 Mnemosyne 工具事件数',
      memory_write_events: '持久记忆写入次数',
      context_events: '上下文、压缩、Token 或缓存压力相关次数',
      gateway_events: '网关、API 或聊天平台活动次数',
      plugin_events: '管理面板插件开发或使用信号数',
      rollback_events: '回滚或检查点恢复次数',
      total_web_calls: '累计 web_search 与 web_extract 调用数',
      total_web_extract_calls: '累计 web_extract 调用数',
      docs_activity_events: '文档、README 或 docs 活动次数',
      browser_calls: '累计浏览器自动化调用数',
      total_terminal_calls: '累计终端调用数',
      total_patch_calls: '累计定向补丁编辑数',
      total_file_reads_searches: '累计 read_file 与 search_files 调用数',
      image_vision_calls: '图像生成或视觉工具调用数',
      tts_calls: '文字转语音或语音工具调用数',
      model_events: '模型或服务商相关活动次数',
      openrouter_events: 'OpenRouter 提及次数',
      codex_events: 'Codex 提及次数',
      distinct_model_count: '会话元数据中出现的不同模型数',
      distinct_provider_count: '从会话元数据推断出的不同模型服务商数',
      claude_events: 'Claude 或 Anthropic 模型提及次数',
      gemini_events: 'Gemini 或 Google 模型提及次数',
      local_model_chat_sessions: '模型元数据为本地或开放权重模型的 Hermes 会话数',
      toolset_events: '工具集或工具系列提及次数',
      config_events: '配置、环境或清单文件活动次数',
      git_history_events: 'rebase、merge、fetch、push 或 tag 等 Git 历史操作数',
      test_events: '测试、检查或验证命令提及次数',
      screenshot_events: '截图、Playwright、PNG 或视觉检查活动次数',
      session_count: 'Hermes 会话数',
      weekend_sessions: '周末开始的会话数',
      night_sessions: '深夜或凌晨开始的会话数',
      cache_events: '提示词缓存或缓存命中次数'
    },
    definitions: {
      let_him_cook: {
        name: '让他放手做',
        description: '让 Hermes 在单个会话中运行一条真正有分量的自主工具链。'
      },
      autonomous_avalanche: {
        name: '自主雪崩',
        description: '跨会话累积如雪崩般的 Hermes 工具调用。'
      },
      toolchain_maxxer: {
        name: '工具链拉满',
        description: '在单个会话中使用种类广泛的 Hermes 工具。'
      },
      full_send: {
        name: '火力全开',
        description: '在一次真实任务中同时调动终端、文件和 Web 或浏览器。'
      },
      subagent_commander: {
        name: '子 Agent 指挥官',
        description: '协调委派给其他 Agent 的工作。'
      },
      background_process_enjoyer: {
        name: '后台进程爱好者',
        description: '启动或控制足够多的长时间运行进程。'
      },
      cron_necromancer: {
        name: '定时任务唤灵师',
        description: '让计划中的自主任务从沉睡中复活。'
      },
      red_text_connoisseur: {
        name: '红字鉴赏家',
        description: '经历足够多的错误，练就品鉴红字的眼光。'
      },
      stack_trace_sommelier: {
        name: '堆栈回溯品鉴师',
        description: '成批品尝回溯信息，而不是浅尝一口。'
      },
      actually_read_the_logs: {
        name: '真的去看日志了',
        description: '反复检查日志，而不是凭空猜测。'
      },
      port_3000_taken: {
        name: '3000 端口又被占了',
        description: '多次发现开发服务器端口冲突，直到习以为常。'
      },
      permission_denied_any_percent: {
        name: '权限拒绝速通',
        description: '以最快速度撞上权限墙。'
      },
      dependency_hell_tourist: {
        name: '依赖地狱游客',
        description: '软件包安装失败了，但生活仍以某种方式继续。'
      },
      the_fix_was_restarting: {
        name: '修复方法是重启',
        description: '在错误集中出现后反复重启，最终把它练成一门技术。'
      },
      forgot_the_env_var: {
        name: '忘了环境变量',
        description: '因为缺少环境变量而导致认证或配置失败。'
      },
      yaml_colon_incident: {
        name: 'YAML 冒号事故',
        description: '配置语法反咬了一口。'
      },
      docker_name_collision: {
        name: 'Docker 名称冲突',
        description: '容器名称已经存在。当然会这样。'
      },
      supposed_to_be_quick: {
        name: '本来应该很快的',
        description: '一个小需求变成了一整场远征。'
      },
      one_more_small_change: {
        name: '再改一个小地方',
        description: '在单个会话中编辑足够多的文件，让“小改动”这个说法彻底失效。'
      },
      vibe_architect: {
        name: '氛围架构师',
        description: '在一次项目会话中触及广泛的代码表面。'
      },
      pixel_goblin: {
        name: '像素哥布林',
        description: '持续进行前端、CSS、SVG 或视觉调整。'
      },
      ship_first_ask_later: {
        name: '先发再说',
        description: '在一条大型工具链之后继续推进 Git 操作。'
      },
      css_exorcist: {
        name: 'CSS 驱魔师',
        description: '反复把样式恶魔逐出界面。'
      },
      one_character_fix: {
        name: '一字符修复',
        description: '在成堆错误后完成一个微小改动。痛苦，但漂亮。'
      },
      skillsmith: {
        name: '技能锻造师',
        description: '频繁使用 Hermes 技能，留下清晰的工作痕迹。'
      },
      skill_issue_skill_created: {
        name: '技能问题？那就造个技能',
        description: '创建或修改持久流程，而不是一遍遍重复。'
      },
      memory_keeper: {
        name: '记忆守护者',
        description: '通过记忆或 Mnemosyne 保存持久知识。'
      },
      memory_palace: {
        name: '记忆宫殿',
        description: '构建一条扎实的持久记忆轨迹。'
      },
      context_dragon: {
        name: '上下文巨龙',
        description: '反复触及压缩、超大上下文或 Token 压力。'
      },
      gateway_dweller: {
        name: '网关常驻者',
        description: '长期使用连接网关的 Hermes 工作流。'
      },
      plugin_goblin: {
        name: '插件哥布林',
        description: '频繁使用或开发插件，连管理面板都注意到了。'
      },
      rollback_wizard: {
        name: '回滚法师',
        description: '施展回滚或检查点恢复魔法。'
      },
      rabbit_hole_certified: {
        name: '兔子洞认证',
        description: '搜索或提取足够多的 Web 内容，正式进入研究螺旋。'
      },
      citation_goblin: {
        name: '引用哥布林',
        description: '提取足够多的网页，成为一名小小图书管理员。'
      },
      docs_archaeologist: {
        name: '文档考古学家',
        description: '一遍又一遍挖掘文档来源。'
      },
      browser_possession: {
        name: '浏览器附身',
        description: '反复通过自动化接管浏览器。'
      },
      terminal_goblin: {
        name: '终端哥布林',
        description: '在 Shell 世界里投入大量时间。'
      },
      patch_wizard: {
        name: '补丁法师',
        description: '用定向补丁让文件听从你的意志。'
      },
      file_archaeologist: {
        name: '文件考古学家',
        description: '通过读取和搜索深入挖掘文件系统。'
      },
      image_whisperer: {
        name: '图像低语者',
        description: '频繁使用图像生成或视觉工具完成视觉工作。'
      },
      voice_of_the_machine: {
        name: '机器之声',
        description: '反复使用文字转语音或语音工具。'
      },
      model_hopper: {
        name: '模型跳跃者',
        description: '频繁切换或检查服务商与模型，直到成为习惯。'
      },
      openrouter_enjoyer: {
        name: 'OpenRouter 爱好者',
        description: '反复通过 OpenRouter 路由模型任务。'
      },
      codex_conjurer: {
        name: 'Codex 召唤师',
        description: '频繁召唤 Codex 风格的协助，形成固定仪式。'
      },
      multi_model_mage: {
        name: '多模型法师',
        description: '在 Hermes 历史中实际使用多种不同模型。'
      },
      five_model_flight: {
        name: '五模型巡礼',
        description: '尝试至少五种不同的大语言模型，而不是只守着第一个会回答的模型。'
      },
      provider_polyglot: {
        name: '多服务商通才',
        description: '在 Hermes 历史中使用来自多个服务商的模型。'
      },
      model_sommelier: {
        name: '模型品鉴师',
        description: '经历足够多的模型与服务商会话，形成自己的偏好。'
      },
      claude_confidant: {
        name: 'Claude 知己',
        description: '反复把 Claude 风格的推理带入工作流。'
      },
      gemini_cartographer: {
        name: 'Gemini 制图师',
        description: '绘制足够多的 Gemini 工作流，熟悉其中地形。'
      },
      open_weights_pilgrim: {
        name: '开放权重朝圣者',
        description: '通过 Hermes 会话元数据真正与本地或开放权重模型对话。'
      },
      toolset_cartographer: {
        name: '工具集制图师',
        description: '有意识地使用 Hermes 工具集，而不是把所有工具混成一团。'
      },
      config_surgeon: {
        name: '配置外科医生',
        description: '从容处理真实配置文件、清单、环境文件和管理面板设置。'
      },
      rebase_acrobat: {
        name: '变基杂技师',
        description: '处理真实的 Git 历史手术：变基、冲突、合并、拉取和推送。'
      },
      test_suite_tamer: {
        name: '测试套件驯兽师',
        description: '运行足够多的验证命令，让绿色输出成为固定仪式。'
      },
      screenshot_hunter: {
        name: '截图猎人',
        description: '捕获、检查并打磨视觉证据，而不是只声称它能工作。'
      },
      marathon_operator: {
        name: '马拉松操作员',
        description: '累积数量可观的 Hermes 会话。'
      },
      weekend_warrior: {
        name: '周末战士',
        description: '在周末运行足够多次 Hermes，把它变成生活方式。'
      },
      night_shift_operator: {
        name: '夜班操作员',
        description: '反复在深夜出没时段运行会话。'
      },
      cache_hit_appreciator: {
        name: '缓存命中鉴赏家',
        description: '留意并受益于提示词缓存行为。'
      }
    }
  },
  kanban: {
    loading: '正在加载看板…',
    loadFailed: '加载看板失败：',
    loadFailedHint: '后端会在首次读取时自动创建 kanban.db。如果问题持续，请检查管理面板日志。',
    board: '看板',
    newBoard: '+ 新建看板',
    newBoardTitle: '新建看板',
    newBoardDescription:
      '看板用于隔离互不相关的工作流，例如为每个项目、代码库或业务域分别创建看板。一个看板中的工作进程不会看到其他看板的任务。',
    slug: '标识',
    slugHint: '— 小写字母、连字符，例如 atm10-server',
    displayName: '显示名称',
    displayNameHint: '（可选）',
    description: '描述',
    descriptionHint: '（可选）',
    icon: '图标',
    iconHint: '（单个字符或表情）',
    switchAfterCreate: '创建后切换到此看板',
    cancel: '取消',
    creating: '创建中…',
    createBoard: '创建看板',
    search: '搜索',
    filterCards: '筛选卡片…',
    tenant: '租户',
    allTenants: '全部租户',
    assignee: '负责人',
    model: '模型',
    modelProfileDefault: '使用配置档案默认值',
    clickToEditModel: '点击覆盖此任务下次运行使用的模型',
    modelFreeTextPlaceholder: '模型名称（留空则使用配置档案默认值）',
    modelLoading: '正在加载模型…',
    modelProfileDefaultOption: '（使用配置档案默认值）',
    allProfiles: '全部配置档案',
    showArchived: '显示已归档',
    lanesByProfile: '按配置档案分组',
    nudgeDispatcher: '唤醒调度器',
    refresh: '刷新',
    selected: '已选中',
    complete: '完成',
    archive: '归档',
    apply: '应用',
    confirm: '确认',
    ok: '确定',
    bulkConfirmTitle: '应用批量更改',
    confirmTitle: '确认更改',
    common: {
      confirm: '确认',
      delete: '删除'
    },
    clear: '清除',
    createTask: '在此列创建任务',
    noTasks: '— 无任务 —',
    unassigned: '未分配',
    needsAssignee: '需要分配负责人',
    needsAssigneeHint: '依赖已满足，但在分配配置档案前，调度器会跳过此任务。',
    untitled: '（无标题）',
    loadingDetail: '加载中…',
    addComment: '添加评论…（按回车提交）',
    comment: '评论',
    status: '状态',
    workspace: '工作区',
    skills: '技能',
    createdBy: '创建者',
    result: '结果',
    comments: '评论',
    events: '事件',
    runHistory: '运行历史',
    workerLog: '工作日志',
    loadingLog: '正在加载日志…',
    noWorkerLog: '— 暂无工作日志（任务尚未启动或日志已轮转）—',
    noDescription: '— 无描述 —',
    noComments: '— 无评论 —',
    edit: '编辑',
    save: '保存',
    dependencies: '依赖',
    parents: '父任务：',
    children: '子任务：',
    none: '无',
    addParent: '— 添加父任务 —',
    addChild: '— 添加子任务 —',
    removeDependency: '移除依赖',
    block: '阻塞',
    unblock: '解除阻塞',
    notifyHomeChannels: '通知主频道',
    diagnostics: '诊断',
    hide: '隐藏',
    show: '显示',
    attention: '注意',
    tasksNeedAttention: '个任务需要关注',
    taskNeedsAttention: '1 个任务需要关注',
    diagnostic: '诊断',
    open: '打开',
    close: '关闭（Esc）',
    reassignTo: '重新分配给：',
    copied: '已复制',
    copyCommand: '复制命令到剪贴板',
    copyCommandPrompt: '复制此命令：',
    reclaim: '收回',
    reassign: '重新分配',
    renderingError: '看板标签页发生渲染错误',
    reloadView: '重新加载视图',
    wsAuthFailed: 'WebSocket 认证失败，请刷新页面以更新会话令牌。',
    markDone: '将 {n} 个任务标记为完成？',
    markArchived: '归档 {n} 个任务？',
    warning: '警告',
    phantomIds: '无效卡片 ID：',
    active: '运行中',
    ended: '已结束',
    noProfile: '（无配置档案）',
    showAllAttempts: '显示所有尝试',
    sendingUpdates: '正在发送更新到',
    sendNotifications: '发送完成 / 阻塞 / 放弃通知到',
    archiveBoardConfirm:
      '归档看板「{name}」？它将被移动到 boards/_archived/，以便以后恢复。此看板上的任务将不再出现在界面中。',
    archiveBoardTitle: '归档此看板',
    boardSwitcherHint: '看板可以将不相关的工作流分开',
    taskCreatedWarning: '任务已创建，但：',
    actionFailed: '操作失败：',
    moveFailed: '移动失败：',
    bulkFailed: '批量操作：',
    bulkMoveFailed: '批量移动失败：{total} 项中有 {failed} 项失败',
    bulkFailedDetails: '批量操作失败：{total} 项中有 {failed} 项失败：{details}',
    completionBlockedHallucination: '⚠ 无法完成：存在疑似虚构的卡片 ID',
    suspectedHallucinatedReferences: '⚠ 文本引用了疑似虚构的卡片 ID',
    pickProfileFirst: '请先选择一个配置档案。',
    unblockedMessage: '已解除阻塞 {id}。任务已准备好进入下一轮调度。',
    unblockFailed: '解除阻塞失败：',
    reclaimedMessage: '已收回 {id}。任务已回到就绪状态。',
    reclaimFailed: '收回失败：',
    reassignedMessage: '已将 {id} 重新分配给 {profile}。',
    reassignFailed: '重新分配失败：',
    selectForBulk: '选择以进行批量操作',
    clickToEdit: '点击编辑',
    clickToEditAssignee: '点击编辑负责人',
    emptyAssignee: '（留空 = 取消分配）',
    columnLabels: {
      triage: '待分类',
      todo: '待办',
      scheduled: '已调度',
      ready: '就绪',
      running: '进行中',
      blocked: '阻塞',
      done: '已完成',
      archived: '已归档'
    },
    columnHelp: {
      triage: '原始想法——任务细化器将补充完整规格',
      todo: '等待依赖项或未分配',
      scheduled: '等待已知的时间延迟或已调度的跟进',
      ready: '依赖项已满足；分配配置档案后即可调度',
      running: '工作进程已认领并正在执行',
      blocked: '工作进程正在等待人工输入',
      done: '已完成',
      archived: '已归档'
    },
    confirmDone: '将此任务标记为完成？工作进程将被释放，依赖此任务的子任务将变为就绪。',
    confirmArchive: '归档此任务？它将从默认看板视图中消失。',
    confirmBlocked: '将此任务标记为阻塞？工作进程将被释放。',
    confirmScheduled: '将此任务移至「已调度」？适用于已知的时间延迟，而非人工阻塞。',
    confirmDoneMany: '将 {n} 个任务标记为完成？工作进程将被释放，依赖这些任务的子任务将变为就绪。',
    confirmArchiveMany: '归档 {n} 个任务？它们将从默认看板视图中消失。',
    confirmBlockedMany: '将 {n} 个任务标记为阻塞？工作进程将被释放。',
    completionSummary: '{label} 的完成摘要。这将作为任务结果存储。',
    completionSummaryThisTask: '此任务',
    completionSummarySelectedTasks: '选中的 {count} 个任务',
    completionSummaryRequired: '在将任务标记为完成之前，必须提供完成摘要。',
    triagePlaceholder: '输入初步想法，AI 将补充完整规格…',
    taskTitlePlaceholder: '新任务标题…',
    specifier: '任务细化器',
    assigneePlaceholder: '负责人',
    priority: '优先级',
    skillsPlaceholder: '技能（可选，逗号分隔）：翻译、github-code-review',
    noParent: '— 无父任务 —',
    workspacePathDir: '工作区路径（必填，例如 ~/projects/my-app）',
    workspacePathOptional: '工作区路径（可选，留空则根据负责人推导）',
    logTruncated: '（仅显示最后 100 KB，完整日志位于 ',
    logAt: '）',
    newTaskTitle: '新建任务 — {column}',
    taskTitleLabel: '标题',
    assigneeLabel: '负责人',
    assigneeLabelHint: '（留空则由调度器选择）',
    skillsLabel: '技能',
    skillsLabelHint: '（可选，以英文逗号分隔）',
    parentLabel: '父任务',
    parentLabelHint: '（父任务完成前，子任务保持阻塞）',
    create: '创建',
    boardSettings: '设置',
    boardSettingsTitle: '看板设置 — 名称、描述和新任务默认继承的项目目录',
    boardSettingsTitleFor: '看板设置 — {name}',
    projectDirectoryOverrideHint: '新任务默认继承此目录作为工作区；仍可在创建任务时单独覆盖。',
    saving: '正在保存…',
    commentHint: '评论会在工作进程下次运行或调用 kanban_show() 时送达，无需先阻塞任务。',
    commentHintTitle:
      '评论是与任务工作进程沟通的通道，会立即写入任务线程，无需先阻塞任务。运行中的工作进程会在下次调用 kanban_show() 或重新启动时读取线程；只有希望工作进程停止并等待你的输入时，才需要阻塞任务。',
    attachments: '附件',
    childResults: '子任务结果',
    clearFilters: '清除筛选',
    confirmRemoveAttachment: '移除此附件？',
    delete: '删除',
    doneNoResult: '未记录最终结果。请到运行历史、日志或子任务中查看工作进程输出。',
    doneParentNote: '此卡片是编排器或父任务，请在子任务结果中查看实际工作成果。',
    finalResult: '最终结果（运行摘要）',
    goalMaxTurns: '最大轮次（默认 20）',
    goalEnabled: '已开启',
    goalEnabledMax: '已开启（最多 {turns} 轮）',
    goalMode: '目标模式',
    noAttachments: '— 无附件 —',
    noChildResult: '尚未记录结果。',
    projectDirectory: '项目目录',
    projectDirectoryExplanation: '设置任务文件的默认位置，以保留项目产出。',
    projectDirectoryHelp: 'Git 项目使用保留的工作树；其他文件夹直接使用该目录。仅临时工作可留空。',
    projectDirectoryHint: '（推荐）',
    projectDirectoryPlaceholder: '项目文件夹的绝对路径',
    removeAttachment: '移除附件',
    setPriority: '设置优先级',
    uploadFile: '上传文件',
    uploading: '正在上传…',
    workspaceDir: '目录——保留',
    workspaceScratch: '临时目录——完成后删除',
    workspaceScratchWarning: '任务完成后，此工作区及其中残留的文件都会被删除。',
    workspaceWorktree: 'Git 工作树——保留',
    trash: {
      confirm: '永久删除此任务？此操作无法撤销。',
      confirmMany: '永久删除选中的 {n} 个任务？此操作无法撤销。',
      confirmTitle: '删除任务？',
      confirmManyTitle: '删除 {n} 个任务？',
      dropHint: '拖放到此处删除'
    },
    hints: {
      assignee: '要分配的 Hermes 配置档案。留空时，调度器会在任务就绪后自动选择。',
      boardSwitcher: '每个看板都是独立工作流，拥有各自的任务、租户和负责人。',
      clearFilters: '清除搜索、租户、负责人和已归档状态等全部筛选条件。',
      createBoard: '为不相关的工作流、项目、团队或隔离的临时区域新建看板。',
      filterAssignee: '按 Hermes 配置档案筛选；配置档案是认领和执行任务的 Agent 身份。',
      filterArchived: '在看板中包含已归档任务；默认情况下它们会被隐藏。',
      filterSearch: '按 ID、标题或描述模糊匹配所有列中的任务。',
      filterTenant: '租户是任务上的自由标签，例如客户、项目或团队。可在任务抽屉或 kanban_create 中设置。',
      groupRunning: '按负责人配置档案对进行中的任务分组。',
      goalMaxTurns: '目标循环的轮次上限；留空使用后端默认值 20。',
      goalMode: '目标模式会让工作进程留在同一会话中，直到评审确认完成或轮次用尽。',
      hideUntilReload: '隐藏到下次刷新页面',
      nudgeDispatcher: '立即唤醒调度器认领就绪任务，不必等待下一轮。',
      parent: '可选的父任务；父任务完成前，子任务保持阻塞。',
      priority: '优先级越高，越先被调度器认领；0 为默认值。',
      refreshBoard: '从数据库重新读取看板；任务事件本身会自动刷新看板。',
      skills: '除内置 kanban-worker 技能外，强制为工作进程加载这些技能。',
      specifier: '负责细化此任务的 Hermes 配置档案；留空使用调度器配置。',
      workspace: '选择任务文件是临时保存，还是在完成后继续保留。'
    },
    bulk: {
      applyAssignee: '将所选负责人应用到全部选中任务。',
      archive: '归档选中任务；任务仍会保留在数据库中。',
      block: '阻塞选中任务并释放活动认领。',
      blockConfirm: '阻塞选中的 {n} 个任务？',
      clear: '清除选择',
      delete: '永久删除选中任务；此操作无法撤销。',
      deselectAll: '取消选择全部任务并隐藏此操作栏。',
      moveReady: '将选中任务移至就绪，并在下一轮交给调度器。',
      moveTodo: '将选中任务移至待办。',
      reassign: '— 重新分配 —',
      reassignHelp: '将选中任务重新分配给其他 Hermes 配置档案，或取消分配。',
      reclaimFirst: '先收回认领',
      reclaimFirstHelp: '重新分配前先收回活动认领。',
      selectAll: '选择全部可见任务',
      selectAllColumn: '选择此列中的全部任务',
      selectAllHelp: '选择各列中全部可见的卡片。',
      setPriorityHelp: '设置选中任务的优先级；数值越高越先被认领。',
      unassign: '（取消分配）',
      unblock: '解除选中任务的阻塞并移至就绪。',
      unblockConfirm: '解除选中的 {n} 个任务的阻塞？'
    },
    cardHints: {
      assignedProfile: '已分配给 Hermes 配置档案 @{profile}',
      childProgress: '{total} 个子任务中已完成 {done} 个',
      columnTasks: '此列有 {count} 个任务',
      comments: '此任务有 {count} 条评论',
      created: '创建于 {time}',
      dependencies: '{parents} 个父任务，{children} 个子任务；父任务完成前子任务保持阻塞。',
      diagnostics: '有 {count} 个活动诊断（严重程度：{severity}）。打开任务可查看详情。',
      noProfile: '尚未分配配置档案。',
      priority: '优先级 {priority}；优先级越高越先被认领。',
      selectAllColumn: '选择「{column}」中的全部任务',
      selectTask: '选择任务 {id}',
      task: '{title} — {id} — {status}',
      taskId: '任务 ID：{id}。可用于 kanban_show 或看板命令行。',
      tenant: '租户：{tenant}。用于任务分组的自由标签。'
    },
    boardForm: {
      descriptionPlaceholder: '这个看板用于什么工作？',
      slugRequired: '必须填写看板标识。'
    },
    docs: {
      ariaLabel: 'Hermes 看板文档',
      open: '在新标签页中打开 Hermes 看板文档'
    },
    orchestration: {
      auto: '自动',
      autoDecomposeLabel: '自动拆解待分类任务',
      autoDescription: '调度器会自动拆解新的待分类任务。',
      autoGenerate: '⚗ 自动生成',
      autoGenerateFailed: '自动生成失败：{error}',
      autoModeHelp: '自动编排会在每轮拆解新的待分类任务。点击切换为手动。',
      autoReview: '自动生成——待检查',
      configure: '配置看板编排器和配置档案路由。',
      defaultAssignee: '默认负责人',
      defaultProfile: '（默认）',
      defaultValue: '（默认：{profile}）',
      descriptionGenerated: '已为 {profile} 自动生成描述。',
      descriptionSaved: '已保存 {profile} 的描述。',
      generating: '正在生成…',
      label: '编排模式',
      loadFailed: '加载编排设置失败：{error}',
      loading: '加载中…',
      loadingMode: '正在加载模式…',
      manual: '手动',
      manualDescription: '待分类任务会等待你点击「⚗ 拆解」。',
      manualModeHelp: '手动编排会保留待分类任务，直到你点击「⚗ 拆解」。点击切换为自动。',
      mode: '编排模式',
      noDescription: '⚠ 无描述',
      noProfiles: '未安装配置档案。',
      orchestratorHelp: '负责拆解后的根任务并评审完成状态。拆解模型请在 auxiliary.kanban_decomposer 下配置。',
      profile: '编排器配置档案',
      profileDescriptionPlaceholder: '此配置档案擅长什么？',
      profileDescriptions: '配置档案描述',
      profileDescriptionsHelp: '描述用于指导路由。点击 ⚗ 自动生成，或手动编辑并保存。',
      reload: '重新加载',
      resolved: '实际使用：{profile}',
      saveDescription: '将此内容保存为用户编写的描述',
      saveFailed: '保存失败：{error}',
      saved: '设置已保存。',
      settings: '编排设置'
    },
    run: {
      earlier: '更早的 {count} 次',
      emptyLog: '（空）',
      metadata: '元数据',
      refreshLog: '刷新日志'
    },
    taskActions: {
      addChild: '+ 子任务',
      addParent: '+ 父任务',
      decompose: '⚗ 拆解',
      decomposeFailed: '拆解失败：{error}',
      decomposed: '已拆解为 {count} 个子任务：{ids}',
      decomposing: '正在拆解…',
      editDescription: '编辑描述',
      moveReady: '→ 就绪',
      moveTodo: '→ 待办',
      moveTriage: '→ 待分类',
      retitled: ' — 已改名为：{title}',
      singleTask: '保持为单个任务（未拆分）{suffix}',
      specify: '✨ 细化',
      specifyFailed: '细化失败：{error}',
      specified: '已完成细化{suffix}',
      specifying: '正在细化…',
      unknownError: '未知错误'
    }
  }
}
