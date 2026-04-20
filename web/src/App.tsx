import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, ClipboardEvent, ReactNode, RefObject } from 'react';
import {
  Archive,
  ArrowDown,
  BrainCircuit,
  Check,
  ChevronDown,
  Code2,
  Copy,
  Download,
  FileImage,
  FolderPlus,
  LogOut,
  Menu,
  Mic,
  PanelRightClose,
  Paperclip,
  PencilLine,
  Pin,
  PinOff,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  Upload,
  Volume2,
  X,
  MessageSquarePlus,
} from 'lucide-react';
import { getStoredAuthSession } from '@/auth/session';
import { LoginScreen } from '@/components/LoginScreen';
import { Markdown } from '@/components/Markdown';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { api } from '@/lib/api';
import { cn, isoTimeAgo } from '@/lib/utils';
import { buildAssistantDraft, stripMarkdown } from '@/chat/assistant';
import {
  appendMessage,
  archiveChat,
  APP_VERSION,
  createAttachmentRecord,
  createAssistantMessage,
  createChat,
  createMemory,
  createUserMessage,
  dedupeMemories,
  deriveChatTitle,
  deriveMemorySuggestions,
  getChatMessages,
  loadAppState,
  mergeChatAppStates,
  normalizeText,
  pinChat,
  removeChat,
  removeMemory,
  saveAppState,
  sortChats,
  unarchiveChat,
  updateChatSummary,
  upsertMemory,
} from '@/chat/persistence';
import type {
  ChatAppState,
  ChatAttachment,
  ChatMessage,
  ChatThread,
  ChatWorkspace,
  MemoryItem,
  MessageSegment,
  OutputMode,
} from '@/chat/types';


type UploadDraft = {
  id: string;
  name: string;
  mimeType: string;
  size: number;
  url: string;
};

type MobilePanel = 'chats' | 'memory' | 'actions' | null;

type LayoutState = {
  focusMode: boolean;
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  mobilePanel: MobilePanel;
};

const LAYOUT_STORAGE_KEY = 'hermes.chat-studio.layout.v3';

function createId(prefix: string) {
  const suffix = typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return `${prefix}_${suffix}`;
}

function loadLayoutState(): LayoutState {
  const fallback: LayoutState = {
    focusMode: false,
    leftCollapsed: false,
    rightCollapsed: false,
    mobilePanel: null,
  };

  if (typeof window === 'undefined') {
    return fallback;
  }

  try {
    const raw = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) {
      return fallback;
    }

    const parsed = JSON.parse(raw) as Partial<LayoutState> & { mobileSection?: string };
    const mobilePanel = parsed.mobilePanel === 'chats' || parsed.mobilePanel === 'memory' || parsed.mobilePanel === 'actions'
      ? parsed.mobilePanel
      : parsed.mobileSection && ['chats', 'memory', 'actions'].includes(parsed.mobileSection)
        ? (parsed.mobileSection as MobilePanel)
        : null;

    return {
      focusMode: Boolean(parsed.focusMode),
      leftCollapsed: Boolean(parsed.leftCollapsed),
      rightCollapsed: Boolean(parsed.rightCollapsed),
      mobilePanel,
    };
  } catch {
    return fallback;
  }
}

function saveLayoutState(state: LayoutState) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(state));
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('pt-BR', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: 'short',
  }).format(new Date(value));
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
  return `${Math.max(1, Math.round(size / (1024 * 1024)))} MB`;
}

function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error('Falha ao ler arquivo'));
    reader.readAsDataURL(file);
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function normalizeImportedAppState(raw: unknown): ChatAppState | null {
  const candidate = isRecord(raw)
    ? (isRecord(raw.state)
      ? raw.state
      : isRecord(raw.appState)
        ? raw.appState
        : raw)
    : null;

  if (!isRecord(candidate)) return null;
  if (!Array.isArray(candidate.workspaces) || !Array.isArray(candidate.chats)) return null;
  if (!Array.isArray(candidate.messages) || !Array.isArray(candidate.memories)) return null;

  const workspaces = candidate.workspaces as ChatWorkspace[];
  const chats = candidate.chats as ChatThread[];
  const messages = candidate.messages as ChatMessage[];
  const memories = candidate.memories as MemoryItem[];
  const activeWorkspaceId = typeof candidate.activeWorkspaceId === 'string'
    ? candidate.activeWorkspaceId
    : workspaces[0]?.id ?? '';
  const activeChatId = typeof candidate.activeChatId === 'string'
    ? candidate.activeChatId
    : chats[0]?.id ?? '';

  if (!activeWorkspaceId || !activeChatId) return null;

  return {
    version: APP_VERSION,
    workspaces,
    chats,
    messages,
    memories,
    activeWorkspaceId,
    activeChatId,
  };
}

function sanitizeFileStem(value: string) {
  const normalized = normalizeText(value).replace(/\s+/g, '-').replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
  return normalized || 'backup';
}

function isGenericTitle(title: string) {
  const normalized = normalizeText(title);
  return normalized === 'novo chat' || normalized === 'chat sem titulo' || normalized.startsWith('novo chat');
}

function extractMessageText(message: ChatMessage) {
  return stripMarkdown(message.content).trim() || message.content.trim();
}

function summarizeChatMessages(messages: ChatMessage[]) {
  const lastMeaningful = [...messages].reverse().find((message) => extractMessageText(message).length > 0);
  return lastMeaningful ? extractMessageText(lastMeaningful).slice(0, 160) : 'Conversa pronta para começar.';
}

function buildMemoryContext(memories: MemoryItem[], chatId: string) {
  const globalMemories = memories.filter((memory) => memory.scope === 'global');
  const chatMemories = memories.filter((memory) => memory.scope === 'chat' && memory.chatId === chatId);
  const sections: string[] = [];

  if (globalMemories.length) {
    sections.push(['Memória global:', ...globalMemories.map((memory) => `- ${memory.content}`)].join('\n'));
  }

  if (chatMemories.length) {
    sections.push(['Memória do chat:', ...chatMemories.map((memory) => `- ${memory.content}`)].join('\n'));
  }

  return sections.join('\n\n');
}

function buildSystemPrompt(params: {
  workspaceName?: string;
  chatTitle?: string;
  memoryContext: string;
  selectedModes: OutputMode[];
}) {
  const modes = params.selectedModes.filter((mode) => mode !== 'text');
  const modeHint = modes.length ? `Modos ativos: ${modes.join(', ')}.` : 'Modo ativo: texto.';

  return [
    'Você é Hermes, um assistente em português do Brasil.',
    'Responda de forma direta, clara e útil.',
    'Seja natural, sem rodeios, mas mantenha profundidade quando o assunto pedir.',
    modeHint,
    params.workspaceName ? `Workspace atual: ${params.workspaceName}.` : '',
    params.chatTitle ? `Chat atual: ${params.chatTitle}.` : '',
    params.memoryContext ? `\nMemória persistente:\n${params.memoryContext}` : '',
  ].filter(Boolean).join('\n');
}

function serializeHistoryForBackend(messages: ChatMessage[], activeChatId: string) {
  return messages
    .filter((message) => message.chatId === activeChatId)
    .filter((message) => !message.pending)
    .map((message) => {
      const attachmentSummary = message.attachments.length
        ? `\n\nAnexos:\n${message.attachments.map((attachment) => `- ${attachment.name} (${attachment.mimeType})`).join('\n')}`
        : '';

      return {
        role: message.role,
        content: `${message.content}${attachmentSummary}`,
      };
    });
}

function ModeChip({ mode, active, onToggle }: { mode: OutputMode; active: boolean; onToggle?: () => void; }) {
  const labels: Record<OutputMode, string> = {
    text: 'Texto',
    image: 'Imagem',
    audio: 'Áudio',
    code: 'Código',
  };
  const icons: Record<OutputMode, ReactNode> = {
    text: <MessageSquarePlus className="h-3.5 w-3.5" />,
    image: <FileImage className="h-3.5 w-3.5" />,
    audio: <Volume2 className="h-3.5 w-3.5" />,
    code: <Code2 className="h-3.5 w-3.5" />,
  };

  if (mode === 'text') {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1.5 text-[0.65rem] uppercase tracking-[0.18em] text-muted-foreground">
        {icons.text}
        texto
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[0.65rem] uppercase tracking-[0.18em] transition-all',
        active
          ? 'border-foreground bg-foreground text-background'
          : 'border-border bg-card/60 text-muted-foreground hover:border-foreground/35 hover:text-foreground',
      )}
    >
      {icons[mode]}
      {labels[mode]}
    </button>
  );
}

function AttachmentCard({ attachment }: { attachment: ChatAttachment }) {
  const isImage = attachment.mimeType.startsWith('image/');
  const isAudio = attachment.mimeType.startsWith('audio/');

  if (isImage) {
    return (
      <div className="overflow-hidden rounded-[1rem] border border-border bg-black/20">
        <img src={attachment.url} alt={attachment.name} className="max-h-80 w-full object-cover" />
        <div className="flex items-center justify-between gap-3 border-t border-border px-3 py-2 text-[0.72rem] text-muted-foreground">
          <span className="truncate">{attachment.name}</span>
          <span>{formatBytes(attachment.size)}</span>
        </div>
      </div>
    );
  }

  if (isAudio) {
    return (
      <div className="overflow-hidden rounded-[1rem] border border-border bg-card/60 p-3">
        <div className="mb-3 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-[0.9rem] border border-border bg-background/40">
            <Volume2 className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm">{attachment.name}</p>
            <p className="text-[0.72rem] text-muted-foreground">{formatBytes(attachment.size)}</p>
          </div>
        </div>
        <audio controls src={attachment.url} className="w-full" />
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-[1rem] border border-border bg-card/60 p-3 text-sm text-muted-foreground">
      <p className="truncate font-medium text-foreground">{attachment.name}</p>
      <p className="mt-1 text-[0.72rem]">{attachment.mimeType} • {formatBytes(attachment.size)}</p>
    </div>
  );
}

function SegmentCard({ segment }: { segment: MessageSegment }) {
  if (segment.kind === 'code') {
    return (
      <div className="overflow-hidden rounded-[1.1rem] border border-border bg-black/25">
        <div className="flex items-center justify-between gap-3 border-b border-border px-3 py-2">
          <div className="min-w-0">
            <p className="font-compressed text-[0.65rem] uppercase tracking-[0.18em] text-muted-foreground">{segment.title}</p>
            {segment.language && <p className="font-courier text-[0.68rem] text-muted-foreground/90">{segment.language}</p>}
          </div>
          <Badge variant="outline">código</Badge>
        </div>
        <pre className="overflow-x-auto p-3 text-[0.78rem] leading-relaxed text-foreground/95">
          <code className="font-courier">{segment.content}</code>
        </pre>
      </div>
    );
  }

  if (segment.kind === 'image') {
    return (
      <div className="overflow-hidden rounded-[1.1rem] border border-border bg-card/70">
        <div className="border-b border-border px-3 py-2">
          <p className="font-compressed text-[0.65rem] uppercase tracking-[0.18em] text-muted-foreground">{segment.title}</p>
        </div>
        {segment.imageUrl && <img src={segment.imageUrl} alt={segment.title} className="max-h-[28rem] w-full object-cover" />}
        {segment.subtitle && <p className="px-3 py-3 text-[0.8rem] text-muted-foreground">{segment.subtitle}</p>}
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-[1.1rem] border border-border bg-card/70 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-compressed text-[0.65rem] uppercase tracking-[0.18em] text-muted-foreground">{segment.title}</p>
          {segment.subtitle && <p className="mt-1 text-[0.8rem] text-muted-foreground">{segment.subtitle}</p>}
        </div>
        <Badge variant="outline">áudio</Badge>
      </div>
      {segment.audioText && (
        <>
          <p className="mt-3 max-h-28 overflow-auto rounded-[1rem] border border-border bg-background/40 p-3 text-[0.8rem] leading-relaxed text-foreground/90">
            {segment.audioText}
          </p>
          <Button
            type="button"
            variant="outline"
            className="mt-3 h-9 px-3"
            onClick={() => {
              if (typeof window === 'undefined' || !window.speechSynthesis) return;
              const utterance = new SpeechSynthesisUtterance(segment.audioText);
              utterance.lang = 'pt-BR';
              utterance.rate = 1;
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(utterance);
            }}
          >
            ouvir resposta
          </Button>
        </>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  onCopy,
  onPromote,
  onDelete,
}: {
  message: ChatMessage;
  onCopy: (message: ChatMessage) => void;
  onPromote: (message: ChatMessage, scope: 'chat' | 'global') => void;
  onDelete: (message: ChatMessage) => void;
}) {
  const isUser = message.role === 'user';

  return (
    <article
      className={cn(
        'group w-full max-w-[min(92%,52rem)] overflow-hidden rounded-[1.5rem] border shadow-[0_12px_32px_rgba(0,0,0,0.05)]',
        isUser
          ? 'ml-auto border-foreground/10 bg-foreground text-background'
          : 'mr-auto border-border bg-card/75 text-foreground backdrop-blur',
      )}
    >
      <div className={cn('flex items-start justify-between gap-3 border-b px-4 py-3', isUser ? 'border-background/10' : 'border-border')}>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className={cn('font-compressed text-[0.65rem] uppercase tracking-[0.2em]', isUser ? 'text-background/70' : 'text-muted-foreground')}>
              {isUser ? 'Você' : 'Hermes'}
            </p>
            <Badge variant={isUser ? 'secondary' : 'outline'}>{formatTime(message.createdAt)}</Badge>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button type="button" size="sm" variant="ghost" className={cn('h-8 px-2', isUser && 'text-background hover:bg-background/10')} onClick={() => onCopy(message)}>
            <Copy className="h-3.5 w-3.5" />
          </Button>
          {!isUser && (
            <>
              <Button type="button" size="sm" variant="ghost" className="h-8 px-2" onClick={() => onPromote(message, 'chat')}>
                <BrainCircuit className="h-3.5 w-3.5" />
              </Button>
              <Button type="button" size="sm" variant="ghost" className="h-8 px-2" onClick={() => onPromote(message, 'global')}>
                <Sparkles className="h-3.5 w-3.5" />
              </Button>
            </>
          )}
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className={cn('h-8 px-2 text-destructive hover:bg-destructive/10', isUser && 'text-background hover:bg-background/10')}
            onClick={() => onDelete(message)}
            title="Apagar mensagem"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div className="space-y-4 px-4 py-4">
        {isUser ? (
          <p className="whitespace-pre-wrap text-[0.95rem] leading-relaxed">{message.content}</p>
        ) : (
          <Markdown content={message.content} />
        )}

        {message.attachments.length > 0 && (
          <div className="grid gap-3">
            {message.attachments.map((attachment) => (
              <AttachmentCard key={attachment.id} attachment={attachment} />
            ))}
          </div>
        )}

        {message.segments.length > 0 && (
          <div className="grid gap-3">
            {message.segments.map((segment) => (
              <SegmentCard key={segment.id} segment={segment} />
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

function CollapsibleSection({
  title,
  description,
  defaultOpen = true,
  actions,
  children,
}: {
  title: string;
  description?: string;
  defaultOpen?: boolean;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="rounded-[1.25rem] border border-border bg-background/35 p-3">
      <div className="flex items-start justify-between gap-3">
        <button
          type="button"
          onClick={() => setOpen((current) => !current)}
          className="min-w-0 flex-1 text-left"
        >
          <div className="flex items-center gap-2">
            <p className="font-compressed text-[0.65rem] uppercase tracking-[0.2em] text-muted-foreground">{title}</p>
            <ChevronDown className={cn('h-3.5 w-3.5 transition-transform text-muted-foreground', open && 'rotate-180')} />
          </div>
          {description && <p className="mt-1 text-[0.78rem] leading-relaxed text-muted-foreground">{description}</p>}
        </button>
        {actions ? <div className="flex items-center gap-1">{actions}</div> : null}
      </div>
      {open && <div className="mt-3">{children}</div>}
    </section>
  );
}

function MobileSheet({
  open,
  title,
  description,
  onClose,
  children,
  footer,
}: {
  open: boolean;
  title: string;
  description?: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-[color:var(--color-background)]/96 backdrop-blur-xl lg:hidden">
      <div className="flex items-start justify-between gap-3 border-b border-border px-4 py-4">
        <div className="min-w-0">
          <p className="font-compressed text-[0.65rem] uppercase tracking-[0.22em] text-muted-foreground">{title}</p>
          {description && <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{description}</p>}
        </div>
        <Button type="button" variant="outline" size="icon" onClick={onClose} className="shrink-0">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {children}
      </div>

      {footer ? (
        <div className="border-t border-border bg-[color:var(--color-card)]/90 px-3 py-3">
          {footer}
        </div>
      ) : null}
    </div>
  );
}

function SidebarRail({
  collapsed,
  activeWorkspace,
  workspaces,
  chats,
  activeChatId,
  search,
  setSearch,
  showArchivedChats,
  setShowArchivedChats,
  workspaceDraft,
  setWorkspaceDraft,
  onCreateWorkspace,
  onCreateChat,
  onSelectWorkspace,
  onSelectChat,
  onPinChat,
  onArchiveChat,
  onUnarchiveChat,
  onDeleteChat,
  sessionEmail,
  activeWorkspaceChatsCount,
  archivedCount,
  onToggleCollapsed,
}: {
  collapsed: boolean;
  activeWorkspace: ChatWorkspace | undefined;
  workspaces: ChatWorkspace[];
  chats: ChatThread[];
  activeChatId: string;
  search: string;
  setSearch: (value: string) => void;
  showArchivedChats: boolean;
  setShowArchivedChats: (value: boolean) => void;
  workspaceDraft: string;
  setWorkspaceDraft: (value: string) => void;
  onCreateWorkspace: () => void;
  onCreateChat: () => void;
  onSelectWorkspace: (workspaceId: string) => void;
  onSelectChat: (chatId: string) => void;
  onPinChat: (chatId: string, pinned: boolean) => void;
  onArchiveChat: (chatId: string) => void;
  onUnarchiveChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
  sessionEmail?: string;
  activeWorkspaceChatsCount: number;
  archivedCount: number;
  onToggleCollapsed: () => void;
}) {
  if (collapsed) {
    return (
      <Card className="flex h-full flex-col overflow-hidden rounded-[2rem] bg-[color:var(--color-card)]/65 backdrop-blur-xl">
        <CardHeader className="gap-3 border-b border-border p-4">
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle className="text-sm">Chats</CardTitle>
              <CardDescription>{activeWorkspace?.name ?? 'Workspace'}</CardDescription>
            </div>
            <Button type="button" variant="ghost" size="icon" onClick={onToggleCollapsed}>
              <Menu className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">{activeWorkspaceChatsCount}</Badge>
            <Badge variant="outline">{archivedCount} arquivados</Badge>
          </div>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 overflow-y-auto p-3">
          <div className="grid gap-2">
            {chats.slice(0, 10).map((chat) => (
              <button
                key={chat.id}
                type="button"
                onClick={() => onSelectChat(chat.id)}
                className={cn(
                  'w-full rounded-[1rem] border px-3 py-2 text-left text-sm transition-colors',
                  activeChatId === chat.id ? 'border-foreground bg-foreground text-background' : 'border-border bg-card/60 hover:border-foreground/35',
                )}
              >
                <p className="truncate font-medium">{chat.title}</p>
                <p className={cn('mt-1 truncate text-[0.72rem]', activeChatId === chat.id ? 'text-background/75' : 'text-muted-foreground')}>
                  {chat.summary}
                </p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="flex h-full flex-col overflow-hidden rounded-[2rem] bg-[color:var(--color-card)]/65 backdrop-blur-xl">
      <CardHeader className="gap-3 border-b border-border p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle>Chats e workspaces</CardTitle>
            <CardDescription>Separe os assuntos e mantenha o contexto por conversa.</CardDescription>
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={onToggleCollapsed}>
            <PanelRightClose className="h-4 w-4" />
          </Button>
        </div>

        <p className="mt-3 text-sm text-muted-foreground">
          Os atalhos rápidos foram movidos para a seção de configuração abaixo para economizar espaço.
        </p>
      </CardHeader>

      <CardContent className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <CollapsibleSection
          title="configuração rápida"
          description="Sessão ativa e atalhos para começar sem perder espaço na tela."
          actions={<Badge variant="outline">{sessionEmail ?? 'sessão local'}</Badge>}
        >
          <div className="grid gap-3">
            <div className="grid gap-2 sm:grid-cols-2">
              <Button type="button" variant="outline" className="h-10 justify-start px-3" onClick={onCreateChat}>
                <MessageSquarePlus className="h-4 w-4" />
                novo chat
              </Button>
              <Button
                type="button"
                variant="outline"
                className="h-10 justify-start px-3"
                onClick={() => {
                  const name = workspaceDraft.trim();
                  if (!name) return;
                  onCreateWorkspace();
                }}
                disabled={!workspaceDraft.trim()}
              >
                <FolderPlus className="h-4 w-4" />
                novo workspace
              </Button>
            </div>
            <div className="grid gap-2 rounded-[1rem] border border-border bg-card/55 p-3 text-sm text-muted-foreground">
              <p>• Use workspaces para separar assuntos e chats para manter foco.</p>
              <p>• Apague chats ou mensagens pelo ícone de lixeira.</p>
              <p>• Recolha os painéis laterais para ganhar espaço em tela.</p>
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="workspace ativo"
          description="Separe os assuntos e mantenha o contexto por conversa."
          actions={<Badge variant="outline">{activeWorkspaceChatsCount} chats</Badge>}
        >
          <div className="flex flex-wrap gap-2">
            {workspaces.map((workspace) => (
              <button
                key={workspace.id}
                type="button"
                onClick={() => onSelectWorkspace(workspace.id)}
                className={cn(
                  'rounded-full border px-3 py-1.5 text-[0.72rem] transition-colors',
                  workspace.id === activeWorkspace?.id ? 'border-foreground bg-foreground text-background' : 'border-border bg-card/60 text-muted-foreground hover:text-foreground',
                )}
              >
                {workspace.name}
              </button>
            ))}
          </div>
          <Input
            value={workspaceDraft}
            onChange={(event) => setWorkspaceDraft(event.target.value)}
            placeholder="Nome do novo workspace"
            className="mt-3 h-10 rounded-[1rem]"
          />
          <p className="mt-2 text-[0.78rem] text-muted-foreground">Use workspaces para agrupar assuntos: pessoal, projeto, estratégia, etc.</p>
        </CollapsibleSection>

        <CollapsibleSection
          title="buscar"
          description="Filtre chats, mensagens e históricos."
          actions={<Badge variant="outline">{activeWorkspaceChatsCount + archivedCount}</Badge>}
        >
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <p className="font-compressed text-[0.65rem] uppercase tracking-[0.2em] text-muted-foreground">buscar</p>
          </div>
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Buscar chats e mensagens"
            className="mt-2 h-10 rounded-[1rem]"
          />
          <label className="mt-3 flex items-center justify-between gap-3 rounded-[1rem] border border-border bg-card/60 px-3 py-2 text-sm">
            <span className="text-muted-foreground">Mostrar arquivados</span>
            <input type="checkbox" checked={showArchivedChats} onChange={(event) => setShowArchivedChats(event.target.checked)} />
          </label>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge variant="outline">{activeWorkspaceChatsCount} ativos</Badge>
            <Badge variant="outline">{archivedCount} arquivados</Badge>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="lista de chats"
          description={showArchivedChats ? 'Mostrando ativos e arquivados.' : 'Mostrando apenas ativos.'}
          actions={<Badge variant="outline">{chats.length}</Badge>}
        >
          <div className="grid gap-2">
            {chats.length === 0 ? (
              <div className="rounded-[1rem] border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
                Nenhum chat encontrado.
              </div>
            ) : (
              chats.map((chat) => {
                const isActive = chat.id === activeChatId;
                return (
                  <article
                    key={chat.id}
                    className={cn(
                      'rounded-[1.1rem] border p-3 transition-colors',
                      isActive ? 'border-foreground bg-foreground text-background' : 'border-border bg-card/65',
                    )}
                  >
                    <button type="button" onClick={() => onSelectChat(chat.id)} className="w-full text-left">
                      <div className="flex items-center gap-2">
                        {chat.pinned && <Pin className={cn('h-3.5 w-3.5', isActive ? 'text-background' : 'text-amber-600')} />}
                        <p className="min-w-0 flex-1 truncate font-medium">{chat.title}</p>
                        {isActive && <Check className="h-4 w-4" />}
                      </div>
                      <p className={cn('mt-1 line-clamp-2 text-[0.75rem] leading-relaxed', isActive ? 'text-background/75' : 'text-muted-foreground')}>
                        {chat.summary}
                      </p>
                    </button>

                    <div className="mt-3 flex flex-wrap gap-1.5">
                      <Button type="button" size="sm" variant="ghost" className={cn('h-8 px-2', isActive && 'text-background hover:bg-background/10')} onClick={() => onPinChat(chat.id, !chat.pinned)}>
                        {chat.pinned ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
                      </Button>
                      {chat.archivedAt ? (
                        <Button type="button" size="sm" variant="ghost" className={cn('h-8 px-2', isActive && 'text-background hover:bg-background/10')} onClick={() => onUnarchiveChat(chat.id)}>
                          <Archive className="h-3.5 w-3.5" />
                        </Button>
                      ) : (
                        <Button type="button" size="sm" variant="ghost" className={cn('h-8 px-2', isActive && 'text-background hover:bg-background/10')} onClick={() => onArchiveChat(chat.id)}>
                          <Archive className="h-3.5 w-3.5" />
                        </Button>
                      )}
                      <Button type="button" size="sm" variant="ghost" className={cn('h-8 px-2 text-destructive hover:bg-destructive/10', isActive && 'text-background hover:bg-background/10')} onClick={() => onDeleteChat(chat.id)}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </article>
                );
              })
            )}
          </div>
        </CollapsibleSection>
      </CardContent>
    </Card>
  );
}

function MemoryRail({
  collapsed,
  activeChat,
  globalMemories,
  chatMemories,
  memoryDraft,
  setMemoryDraft,
  onAddMemory,
  onDeleteMemory,
  onTogglePin,
  onMoveScope,
  onClearChat,
  onExportState,
  onImportStateClick,
  onImportState,
  stateBackupInputRef,
  onToggleCollapsed,
}: {
  collapsed: boolean;
  activeChat: ChatThread | null;
  globalMemories: MemoryItem[];
  chatMemories: MemoryItem[];
  memoryDraft: string;
  setMemoryDraft: (value: string) => void;
  onAddMemory: () => void;
  onDeleteMemory: (memoryId: string) => void;
  onTogglePin: (memory: MemoryItem) => void;
  onMoveScope: (memory: MemoryItem, scope: 'global' | 'chat') => void;
  onClearChat: () => void;
  onExportState: () => void;
  onImportStateClick: () => void;
  onImportState: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  stateBackupInputRef: RefObject<HTMLInputElement | null>;
  onToggleCollapsed: () => void;
}) {
  const allMemories = [...globalMemories, ...chatMemories].sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updatedAt.localeCompare(a.updatedAt));

  if (collapsed) {
    return (
      <Card className="flex h-full flex-col overflow-hidden rounded-[2rem] bg-[color:var(--color-card)]/65 backdrop-blur-xl">
        <CardHeader className="gap-3 border-b border-border p-4">
          <div className="flex items-start justify-between gap-2">
            <div>
              <CardTitle className="text-sm">Memória</CardTitle>
              <CardDescription>{allMemories.length} itens</CardDescription>
            </div>
            <div className="flex items-center gap-1">
              <Button type="button" variant="ghost" size="icon" onClick={onExportState} title="Exportar backup">
                <Download className="h-4 w-4" />
              </Button>
              <Button type="button" variant="ghost" size="icon" onClick={onImportStateClick} title="Restaurar backup">
                <Upload className="h-4 w-4" />
              </Button>
              <Button type="button" variant="ghost" size="icon" onClick={onToggleCollapsed}>
                <PanelRightClose className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <Badge variant="outline">{activeChat?.title ?? 'sem chat'}</Badge>
          <input
            ref={stateBackupInputRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={onImportState}
          />
        </CardHeader>
        <CardContent className="min-h-0 flex-1 overflow-y-auto p-3">
          <div className="grid gap-2">
            {allMemories.slice(0, 10).map((memory) => (
              <div key={memory.id} className="rounded-[1rem] border border-border bg-card/60 p-3 text-sm">
                <p className="line-clamp-3 text-[0.8rem] leading-relaxed text-foreground">{memory.content}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="flex h-full flex-col overflow-hidden rounded-[2rem] bg-[color:var(--color-card)]/65 backdrop-blur-xl">
      <CardHeader className="gap-3 border-b border-border p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle>Memória e contexto</CardTitle>
            <CardDescription>Global + por chat, editável no navegador.</CardDescription>
          </div>
          <div className="flex items-center gap-1">
            <Button type="button" variant="ghost" size="icon" onClick={onExportState} title="Exportar backup">
              <Download className="h-4 w-4" />
            </Button>
            <Button type="button" variant="ghost" size="icon" onClick={onImportStateClick} title="Restaurar backup">
              <Upload className="h-4 w-4" />
            </Button>
            <Button type="button" variant="ghost" size="icon" onClick={onToggleCollapsed}>
              <PanelRightClose className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">{globalMemories.length} global</Badge>
          <Badge variant="outline">{chatMemories.length} chat</Badge>
        </div>
        <input
          ref={stateBackupInputRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={onImportState}
        />
      </CardHeader>

      <CardContent className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <CollapsibleSection
          title="adicionar memória"
          description="Capture decisões importantes sem sair do chat."
          actions={<Badge variant="outline">{activeChat?.title ?? 'sem chat'}</Badge>}
        >
          <textarea
            value={memoryDraft}
            onChange={(event) => setMemoryDraft(event.target.value)}
            placeholder="Escreva uma lembrança útil para este chat"
            className="min-h-24 w-full resize-none rounded-[1rem] border border-border bg-background/45 px-3 py-3 font-courier text-[0.9rem] outline-none placeholder:text-muted-foreground"
          />
          <div className="mt-3 flex items-center gap-2">
            <Button type="button" className="h-10 px-4" onClick={onAddMemory} disabled={!memoryDraft.trim()}>
              <Plus className="h-4 w-4" />
              salvar
            </Button>
            <Button type="button" variant="outline" className="h-10 px-4" onClick={onClearChat} disabled={!activeChat}>
              limpar chat
            </Button>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="memórias"
          description="Global + por chat, editável no navegador."
          actions={<Badge variant="outline">{allMemories.length}</Badge>}
        >
          <div className="grid gap-2">
            {allMemories.length === 0 ? (
              <div className="rounded-[1rem] border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
                Nenhuma memória ainda.
              </div>
            ) : (
              allMemories.map((memory) => (
                <article key={memory.id} className="rounded-[1.1rem] border border-border bg-card/65 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant={memory.scope === 'global' ? 'default' : 'outline'}>{memory.scope}</Badge>
                        <Badge variant="outline">{memory.kind}</Badge>
                        {memory.pinned && <Badge variant="warning">fixada</Badge>}
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-foreground">{memory.content}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <Button type="button" size="sm" variant="ghost" className="h-8 px-2" onClick={() => onTogglePin(memory)}>
                        <Pin className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        className="h-8 px-2"
                        onClick={() => onMoveScope(memory, memory.scope === 'global' ? 'chat' : 'global')}
                      >
                        <Archive className="h-3.5 w-3.5" />
                      </Button>
                      <Button type="button" size="sm" variant="ghost" className="h-8 px-2 text-destructive hover:bg-destructive/10" onClick={() => onDeleteMemory(memory.id)}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <p className="mt-2 text-[0.72rem] text-muted-foreground">Atualizada {isoTimeAgo(memory.updatedAt)}</p>
                </article>
              ))
            )}
          </div>
        </CollapsibleSection>
      </CardContent>
    </Card>
  );
}

function ChatPane({
  activeWorkspace,
  activeChat,
  chatMessages,
  selectedModes,
  onToggleMode,
  draft,
  setDraft,
  uploads,
  onRemoveUpload,
  onSend,
  onCopyMessage,
  onPromoteMessage,
  onDeleteMessage,
  onCopyLastReply,
  onPinChat,
  onArchiveChat,
  onDeleteChat,
  onRenameChat,
  onClearChat,
  onExportChat,
  isPending,
  showJumpToLatest,
  onJumpToLatest,
  onDropFiles,
  onPasteFiles,
  onToggleRecording,
  isRecordingAudio,
  recordingSeconds,
  recordingError,
  canSend,
  imageInputRef,
  audioInputRef,
  scrollContainerRef,
  onScroll,
  compactHeader = false,
}: {
  activeWorkspace: ChatWorkspace | undefined;
  activeChat: ChatThread | null;
  chatMessages: ChatMessage[];
  selectedModes: OutputMode[];
  onToggleMode: (mode: Exclude<OutputMode, 'text'>) => void;
  draft: string;
  setDraft: (value: string) => void;
  uploads: UploadDraft[];
  onRemoveUpload: (id: string) => void;
  onSend: () => Promise<void>;
  onCopyMessage: (message: ChatMessage) => void;
  onPromoteMessage: (message: ChatMessage, scope: 'chat' | 'global') => void;
  onDeleteMessage: (message: ChatMessage) => void;
  onCopyLastReply: () => void;
  onPinChat: () => void;
  onArchiveChat: () => void;
  onDeleteChat: (chatId: string) => void;
  onRenameChat: () => void;
  onClearChat: () => void;
  onExportChat: () => void;
  isPending: boolean;
  showJumpToLatest: boolean;
  onJumpToLatest: () => void;
  onDropFiles: (files: FileList | File[] | null) => Promise<void>;
  onPasteFiles: (event: ClipboardEvent<HTMLTextAreaElement>) => Promise<void>;
  onToggleRecording: () => void;
  isRecordingAudio: boolean;
  recordingSeconds: number;
  recordingError: string;
  canSend: boolean;
  imageInputRef: RefObject<HTMLInputElement | null>;
  audioInputRef: RefObject<HTMLInputElement | null>;
  scrollContainerRef: RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  compactHeader?: boolean;
}) {
  const [dragActive, setDragActive] = useState(false);

  return (
    <Card className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-[2rem] bg-[color:var(--color-card)]/58 backdrop-blur-xl">
      <CardHeader className={cn('border-b border-border', compactHeader ? 'px-3 py-3' : 'px-4 py-4 sm:px-6')}>
        <div className={cn('flex flex-col', compactHeader ? 'gap-3' : 'gap-4')}>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">{activeWorkspace?.name ?? 'Workspace'}</Badge>
                <Badge variant="success">{chatMessages.length} mensagens</Badge>
                {!compactHeader && activeChat?.pinned && <Badge variant="warning">fixado</Badge>}
                {!compactHeader && isPending && <Badge variant="default">respondendo</Badge>}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <h2 className={cn('font-display text-2xl text-foreground', compactHeader ? 'sm:text-[1.55rem]' : 'sm:text-[2.25rem]')}>
                  {activeChat?.title ?? 'Sem chat ativo'}
                </h2>
                {activeChat && <Badge variant="outline">{isoTimeAgo(activeChat.updatedAt)}</Badge>}
              </div>
              {!activeChat ? (
                <p className={cn('mt-2 text-sm text-muted-foreground', compactHeader && 'text-[0.82rem] leading-relaxed')}>
                  Crie ou selecione um chat para começar.
                </p>
              ) : (
                <p className={cn('mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground', compactHeader && 'max-w-none text-[0.82rem] leading-snug sm:text-[0.9rem]')}>
                  {activeChat.summary || 'Crie um chat, mande texto ou imagem e deixe a memória cuidar do contexto.'}
                </p>
              )}
            </div>

            {!compactHeader && (
              <div className="flex flex-wrap items-center gap-2">
                {activeChat && (
                  <>
                    <Button type="button" variant="outline" className="h-10 px-3" onClick={onCopyLastReply} disabled={!chatMessages.some((message) => message.role === 'assistant')}>
                      <Copy className="h-4 w-4" />
                      copiar última
                    </Button>
                    <Button type="button" variant="outline" className="h-10 px-3" onClick={onExportChat}>
                      <Download className="h-4 w-4" />
                      exportar
                    </Button>
                    <Button type="button" variant="outline" className="h-10 px-3" onClick={onPinChat}>
                      {activeChat.pinned ? <PinOff className="h-4 w-4" /> : <Pin className="h-4 w-4" />}
                      {activeChat.pinned ? 'desafixar' : 'fixar'}
                    </Button>
                    <Button type="button" variant="outline" className="h-10 px-3" onClick={onArchiveChat}>
                      <Archive className="h-4 w-4" />
                      arquivar
                    </Button>
                    <Button type="button" variant="outline" className="h-10 px-3 text-destructive" onClick={() => onDeleteChat(activeChat.id)}>
                      <Trash2 className="h-4 w-4" />
                      apagar chat
                    </Button>
                    <Button type="button" variant="outline" className="h-10 px-3" onClick={onRenameChat}>
                      <PencilLine className="h-4 w-4" />
                      renomear
                    </Button>
                    <Button type="button" variant="outline" className="h-10 px-3 text-destructive" onClick={onClearChat}>
                      <Trash2 className="h-4 w-4" />
                      limpar
                    </Button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </CardHeader>

      <div
        ref={scrollContainerRef}
        onScroll={onScroll}
        className={cn(
          'flex-1 overflow-y-auto overscroll-contain px-4 pb-6 sm:px-6',
          dragActive && 'bg-[color:var(--color-foreground)]/[0.03]',
        )}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) {
            setDragActive(false);
          }
        }}
        onDrop={async (event) => {
          event.preventDefault();
          setDragActive(false);
          await onDropFiles(event.dataTransfer.files);
        }}
      >
        {showJumpToLatest && (
          <button
            type="button"
            onClick={onJumpToLatest}
            className="absolute bottom-[7.5rem] right-5 z-20 inline-flex items-center gap-2 rounded-full border border-border bg-card/92 px-3 py-2 text-[0.66rem] uppercase tracking-[0.18em] shadow-[0_16px_40px_rgba(0,0,0,0.16)] backdrop-blur"
          >
            <ArrowDown className="h-3.5 w-3.5" />
            voltar ao fim
          </button>
        )}

        <div className="flex flex-col gap-4 pb-6 pt-4">
          {chatMessages.length === 0 ? (
            <div className="rounded-[1.5rem] border border-dashed border-border bg-card/55 p-6 sm:p-8">
              <p className="font-compressed text-[0.65rem] uppercase tracking-[0.2em] text-muted-foreground">novo começo</p>
              <h3 className="mt-2 font-display text-2xl">Abra um assunto e mande a primeira mensagem.</h3>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
                Você pode enviar texto, imagem ou áudio e manter o contexto dentro deste chat.
              </p>
            </div>
          ) : (
            chatMessages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onCopy={onCopyMessage}
                onPromote={onPromoteMessage}
                onDelete={onDeleteMessage}
              />
            ))
          )}

          {isPending && (
            <article className="mr-auto max-w-[min(92%,52rem)] rounded-[1.4rem] border border-border bg-card/55 px-4 py-3">
              <div className="flex items-center gap-2 text-[0.72rem] uppercase tracking-[0.18em] text-muted-foreground">
                <div className="h-2 w-2 animate-pulse rounded-full bg-foreground" />
                Hermes está pensando
              </div>
            </article>
          )}
        </div>
      </div>

      <div className={cn('border-t border-border bg-[color:var(--color-card)]/92 backdrop-blur-xl', compactHeader ? 'px-3 py-3' : 'px-4 py-3 sm:px-6')}>
        <div className={cn('mb-3 flex flex-wrap items-center gap-2', compactHeader && 'gap-1.5')}>
          <ModeChip mode="text" active />
          <ModeChip mode="image" active={selectedModes.includes('image')} onToggle={() => onToggleMode('image')} />
          <ModeChip mode="audio" active={selectedModes.includes('audio')} onToggle={() => onToggleMode('audio')} />
          <ModeChip mode="code" active={selectedModes.includes('code')} onToggle={() => onToggleMode('code')} />
          {isRecordingAudio ? (
            <Badge variant="warning" className="ml-auto">gravando {Math.floor(recordingSeconds / 60).toString().padStart(2, '0')}:{(recordingSeconds % 60).toString().padStart(2, '0')}</Badge>
          ) : (
            <Badge variant="outline" className="ml-auto">imagem e áudio prontos</Badge>
          )}
        </div>

        <div className={cn('rounded-[1.5rem] border border-border bg-background/35 p-3 shadow-[0_16px_50px_rgba(0,0,0,0.12)]', dragActive && 'ring-2 ring-foreground/10')}>
          {dragActive && (
            <div className="mb-3 rounded-[1rem] border border-dashed border-foreground/25 bg-foreground/[0.04] px-3 py-2 text-[0.72rem] uppercase tracking-[0.18em] text-muted-foreground">
              solte imagem ou áudio aqui
            </div>
          )}

          {uploads.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-3">
              {uploads.map((upload) => (
                <div key={upload.id} className={cn('relative overflow-hidden rounded-[1rem] border border-border bg-black/20', upload.mimeType.startsWith('image/') ? 'h-24 w-24' : 'flex w-64 flex-col gap-3 p-3')}>
                  {upload.mimeType.startsWith('image/') ? (
                    <img src={upload.url} alt={upload.name} className="h-full w-full object-cover" />
                  ) : upload.mimeType.startsWith('audio/') ? (
                    <>
                      <div className="flex items-center gap-3 pr-7">
                        <div className="flex h-11 w-11 items-center justify-center rounded-[0.9rem] border border-border bg-card/60 text-muted-foreground">
                          <Volume2 className="h-5 w-5" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm">{upload.name}</p>
                          <p className="text-[0.72rem] text-muted-foreground">áudio anexado</p>
                        </div>
                      </div>
                      <audio controls src={upload.url} className="w-full" />
                    </>
                  ) : (
                    <div className="p-3 text-[0.8rem] text-muted-foreground">Arquivo pronto para envio.</div>
                  )}
                  <button
                    type="button"
                    onClick={() => onRemoveUpload(upload.id)}
                    className="absolute right-1 top-1 inline-flex h-7 w-7 items-center justify-center rounded-full border border-border bg-background/80 text-foreground"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onPaste={onPasteFiles}
            onKeyDown={async (event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                await onSend();
              }
            }}
            placeholder="Escreva sua mensagem. Enter envia, Shift+Enter quebra linha."
            className="min-h-[7.5rem] w-full resize-none rounded-[1.1rem] border border-border bg-[color:var(--color-background)]/55 px-4 py-4 font-courier text-[0.95rem] leading-relaxed outline-none placeholder:text-muted-foreground"
          />

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={async (event) => {
                await onDropFiles(event.target.files);
                event.target.value = '';
              }}
            />
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              multiple
              className="hidden"
              onChange={async (event) => {
                await onDropFiles(event.target.files);
                event.target.value = '';
              }}
            />
            <Button type="button" variant="outline" className="h-10 px-3" onClick={() => imageInputRef.current?.click()}>
              <Paperclip className="h-4 w-4" />
              anexar imagem
            </Button>
            <Button type="button" variant="outline" className="h-10 px-3" onClick={() => audioInputRef.current?.click()}>
              <Volume2 className="h-4 w-4" />
              anexar áudio
            </Button>
            <Button
              type="button"
              variant={isRecordingAudio ? 'destructive' : 'outline'}
              className={cn('h-10 px-3', isRecordingAudio && 'animate-pulse')}
              onClick={onToggleRecording}
            >
              <Mic className="h-4 w-4" />
              {isRecordingAudio ? `parar gravação ${Math.floor(recordingSeconds / 60).toString().padStart(2, '0')}:${(recordingSeconds % 60).toString().padStart(2, '0')}` : 'gravar áudio'}
            </Button>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              {recordingError ? <Badge variant="warning">{recordingError}</Badge> : <Badge variant="outline">salvo localmente</Badge>}
              <Button type="button" className="h-10 px-4" disabled={!canSend} onClick={() => void onSend()}>
                <Send className="h-4 w-4" />
                enviar
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

function App() {
  const [authSession, setAuthSession] = useState(() => getStoredAuthSession());
  const [authReady, setAuthReady] = useState(false);
  const [chatStateReady, setChatStateReady] = useState(false);
  const [state, setState] = useState<ChatAppState>(() => loadAppState());
  const [layout, setLayout] = useState<LayoutState>(() => loadLayoutState());
  const [search, setSearch] = useState('');
  const [showArchivedChats, setShowArchivedChats] = useState(false);
  const [workspaceDraft, setWorkspaceDraft] = useState('');
  const [memoryDraft, setMemoryDraft] = useState('');
  const [draft, setDraft] = useState('');
  const [uploads, setUploads] = useState<UploadDraft[]>([]);
  const [selectedModes, setSelectedModes] = useState<OutputMode[]>(['text', 'code']);
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingError, setRecordingError] = useState('');
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const nearBottomRef = useRef(true);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const stateBackupInputRef = useRef<HTMLInputElement>(null);
  const recordingMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<BlobPart[]>([]);
  const recordingTimerRef = useRef<number | null>(null);
  const stateSaveTimerRef = useRef<number | null>(null);
  const chatStateRevisionRef = useRef(0);
  const lastSyncedSnapshotRef = useRef('');

  useEffect(() => {
    saveAppState(state);
  }, [state]);

  useEffect(() => {
    saveLayoutState(layout);
  }, [layout]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const session = getStoredAuthSession();
      if (!session) {
        if (!cancelled) {
          setAuthSession(null);
          setAuthReady(true);
          setChatStateReady(true);
        }
        return;
      }

      try {
        await api.me();
        if (!cancelled) {
          setAuthSession(session);
          setAuthReady(true);
        }
      } catch {
        if (!cancelled) {
          setAuthSession(null);
          setAuthReady(true);
          setChatStateReady(true);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!authSession) {
      setChatStateReady(true);
      chatStateRevisionRef.current = 0;
      lastSyncedSnapshotRef.current = '';
      return () => {
        cancelled = true;
      };
    }

    setChatStateReady(false);

    (async () => {
      try {
        const remoteState = await api.getChatState();
        if (!cancelled) {
          setState(remoteState.state);
          chatStateRevisionRef.current = remoteState.revision;
          lastSyncedSnapshotRef.current = JSON.stringify(remoteState.state);
          saveAppState(remoteState.state);
        }
      } catch (error) {
        if (!cancelled) {
          const status = typeof error === 'object' && error && 'status' in error ? Number((error as { status?: number }).status) : null;
          if (status !== 404) {
            // Keep the local cache as the source of truth until the next successful save.
          }
        }
      } finally {
        if (!cancelled) {
          setChatStateReady(true);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [authSession?.token, authSession?.email]);

  useEffect(() => {
    if (stateSaveTimerRef.current) {
      window.clearTimeout(stateSaveTimerRef.current);
      stateSaveTimerRef.current = null;
    }

    if (!authSession || !chatStateReady) {
      return;
    }

    const snapshot = JSON.stringify(state);
    if (snapshot === lastSyncedSnapshotRef.current) {
      return;
    }

    stateSaveTimerRef.current = window.setTimeout(() => {
      void (async () => {
        try {
          const response = await api.saveChatState(state, chatStateRevisionRef.current);
          chatStateRevisionRef.current = response.revision;
          lastSyncedSnapshotRef.current = snapshot;
        } catch (error) {
          const status = typeof error === 'object' && error && 'status' in error ? Number((error as { status?: number }).status) : null;
          if (status === 409) {
            try {
              const remoteState = await api.getChatState();
              chatStateRevisionRef.current = remoteState.revision;
              lastSyncedSnapshotRef.current = JSON.stringify(remoteState.state);
              const merged = mergeChatAppStates(remoteState.state, state);
              saveAppState(merged);
              setState(merged);
            } catch {
              // Keep the local copy; a future change will retry.
            }
          }
        }
      })();
    }, 250);

    return () => {
      if (stateSaveTimerRef.current) {
        window.clearTimeout(stateSaveTimerRef.current);
        stateSaveTimerRef.current = null;
      }
    };
  }, [state, authSession?.token, chatStateReady]);

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        window.clearInterval(recordingTimerRef.current);
      }
      recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
      recordingMediaRecorderRef.current = null;
      recordingStreamRef.current = null;
      recordingChunksRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (!state.chats.length) return;
    const activeStillExists = state.chats.some((chat) => chat.id === state.activeChatId);
    if (!activeStillExists) {
      const fallback = sortChats(state.chats.filter((chat) => chat.workspaceId === state.activeWorkspaceId))[0] ?? sortChats(state.chats)[0];
      if (fallback) {
        setState((current) => ({
          ...current,
          activeWorkspaceId: fallback.workspaceId,
          activeChatId: fallback.id,
        }));
      }
    }
  }, [state.activeChatId, state.activeWorkspaceId, state.chats]);

  const activeWorkspace = useMemo(
    () => state.workspaces.find((workspace) => workspace.id === state.activeWorkspaceId) ?? state.workspaces[0],
    [state.activeWorkspaceId, state.workspaces],
  );

  const activeChat = useMemo(
    () => state.chats.find((chat) => chat.id === state.activeChatId) ?? null,
    [state.activeChatId, state.chats],
  );

  const chatMessages = useMemo(() => (activeChat ? getChatMessages(state, activeChat.id) : []), [state, activeChat]);
  const globalMemories = useMemo(() => state.memories.filter((memory) => memory.scope === 'global'), [state.memories]);
  const chatMemories = useMemo(
    () => state.memories.filter((memory) => memory.scope === 'chat' && memory.chatId === activeChat?.id),
    [state.memories, activeChat?.id],
  );

  const visibleChats = useMemo(() => {
    const query = normalizeText(search);
    const source = state.chats.filter((chat) => chat.workspaceId === state.activeWorkspaceId && (showArchivedChats ? Boolean(chat.archivedAt) : !chat.archivedAt));

    return sortChats(source.filter((chat) => {
      if (!query) return true;
      const haystack = normalizeText([
        chat.title,
        chat.summary,
        ...state.messages.filter((message) => message.chatId === chat.id).map((message) => message.content),
      ].join(' '));
      return haystack.includes(query);
    }));
  }, [search, showArchivedChats, state.activeWorkspaceId, state.chats, state.messages]);

  const activeWorkspaceChatsCount = state.chats.filter((chat) => chat.workspaceId === state.activeWorkspaceId && !chat.archivedAt).length;
  const archivedCount = state.chats.filter((chat) => chat.workspaceId === state.activeWorkspaceId && chat.archivedAt).length;

  const focusRailSnapshotRef = useRef<{ leftCollapsed: boolean; rightCollapsed: boolean } | null>(null);
  const desktopLeftCollapsed = layout.focusMode ? true : layout.leftCollapsed;
  const desktopRightCollapsed = layout.focusMode ? true : layout.rightCollapsed;
  const canSend = ((draft.trim().length > 0) || uploads.length > 0) && !isRecordingAudio && !pendingChatId;
  const closeMobilePanel = () => setLayout((current) => ({ ...current, mobilePanel: null }));
  const openMobilePanel = (panel: Exclude<MobilePanel, null>) => setLayout((current) => ({ ...current, mobilePanel: panel }));

  const scrollChatToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  const handleChatScroll = () => {
    if (!scrollRef.current) return;
    const distanceFromBottom = scrollRef.current.scrollHeight - scrollRef.current.scrollTop - scrollRef.current.clientHeight;
    const nearBottom = distanceFromBottom < 96;
    nearBottomRef.current = nearBottom;
    setShowJumpToLatest(!nearBottom);
  };

  useEffect(() => {
    scrollChatToBottom();
    nearBottomRef.current = true;
    setShowJumpToLatest(false);
  }, [state.activeChatId]);

  useEffect(() => {
    if (pendingChatId === state.activeChatId || nearBottomRef.current) {
      scrollChatToBottom();
      nearBottomRef.current = true;
      setShowJumpToLatest(false);
      return;
    }

    setShowJumpToLatest(true);
  }, [state.messages, state.activeChatId, pendingChatId]);

  const setFocusMode = (enabled: boolean) => {
    setLayout((current) => {
      if (enabled) {
        focusRailSnapshotRef.current = {
          leftCollapsed: current.leftCollapsed,
          rightCollapsed: current.rightCollapsed,
        };

        return {
          ...current,
          focusMode: true,
          leftCollapsed: true,
          rightCollapsed: true,
          mobilePanel: null,
        };
      }

      const restoredRails = focusRailSnapshotRef.current;
      focusRailSnapshotRef.current = null;

      return {
        ...current,
        focusMode: false,
        leftCollapsed: restoredRails?.leftCollapsed ?? current.leftCollapsed,
        rightCollapsed: restoredRails?.rightCollapsed ?? current.rightCollapsed,
        mobilePanel: null,
      };
    });
  };

  const toggleLeftRail = () => setLayout((current) => ({ ...current, leftCollapsed: !current.leftCollapsed, mobilePanel: null }));
  const toggleRightRail = () => setLayout((current) => ({ ...current, rightCollapsed: !current.rightCollapsed, mobilePanel: null }));

  const handleSelectWorkspace = (workspaceId: string) => {
    const workspaceChats = sortChats(state.chats.filter((chat) => chat.workspaceId === workspaceId && !chat.archivedAt));
    const fallback = workspaceChats[0] ?? sortChats(state.chats.filter((chat) => chat.workspaceId === workspaceId))[0];

    if (fallback) {
      setState((current) => ({ ...current, activeWorkspaceId: workspaceId, activeChatId: fallback.id }));
    } else {
      const newChat = createChat(workspaceId, 'Novo chat');
      setState((current) => ({
        ...current,
        chats: [newChat, ...current.chats],
        activeWorkspaceId: workspaceId,
        activeChatId: newChat.id,
      }));
    }

    setLayout((current) => ({ ...current, mobilePanel: null }));
  };

  const handleCreateWorkspace = () => {
    const name = workspaceDraft.trim();
    if (!name) return;

    const accents: ChatWorkspace['accent'][] = ['amber', 'teal', 'violet'];
    const accent = accents[Math.abs(name.length + name.charCodeAt(0)) % accents.length];
    const workspace: ChatWorkspace = {
      id: createId('ws'),
      name,
      description: 'Workspace criado manualmente.',
      accent,
    };
    const chat = createChat(workspace.id, 'Novo chat');

    setState((current) => ({
      ...current,
      workspaces: [workspace, ...current.workspaces],
      chats: [chat, ...current.chats],
      activeWorkspaceId: workspace.id,
      activeChatId: chat.id,
    }));
    setWorkspaceDraft('');
    setLayout((current) => ({ ...current, mobilePanel: null }));
  };

  const handleCreateChat = () => {
    if (!state.activeWorkspaceId) return;
    const chat = createChat(state.activeWorkspaceId, 'Novo chat');
    setState((current) => ({
      ...current,
      chats: [chat, ...current.chats],
      activeWorkspaceId: state.activeWorkspaceId,
      activeChatId: chat.id,
    }));
    setDraft('');
    setUploads([]);
    setLayout((current) => ({ ...current, mobilePanel: null }));
  };

  const handleSelectChat = (chatId: string) => {
    const chat = state.chats.find((item) => item.id === chatId);
    if (!chat) return;
    setState((current) => ({ ...current, activeWorkspaceId: chat.workspaceId, activeChatId: chatId }));
    setLayout((current) => ({ ...current, mobilePanel: null }));
  };

  const handleFiles = async (files: FileList | File[] | null) => {
    if (!files) return;
    const accepted = Array.from(files).filter((file) => file.type.startsWith('image/') || file.type.startsWith('audio/'));
    if (!accepted.length) return;

    const converted = await Promise.all(accepted.map(async (file) => ({
      id: createId('upload'),
      name: file.name,
      mimeType: file.type,
      size: file.size,
      url: await fileToDataUrl(file),
    })));

    setUploads((current) => [...current, ...converted]);
  };

  const handleCopyMessage = async (message: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(extractMessageText(message));
    } catch {
      // ignore clipboard errors
    }
  };

  const handleCopyLastReply = async () => {
    const lastAssistantMessage = [...chatMessages].reverse().find((message) => message.role === 'assistant');
    if (!lastAssistantMessage) return;
    await handleCopyMessage(lastAssistantMessage);
  };

  const handlePromoteMessageToMemory = (message: ChatMessage, scope: 'chat' | 'global') => {
    const content = extractMessageText(message);
    if (!content || !activeChat) return;

    const memory = createMemory({
      scope,
      chatId: scope === 'chat' ? activeChat.id : null,
      kind: message.role === 'assistant' ? 'note' : 'preference',
      content,
      sourceMessageId: message.id,
      confidence: message.role === 'assistant' ? 0.84 : 0.92,
      pinned: false,
    });

    setState((current) => upsertMemory(current, memory));
    setMemoryDraft(content);
  };

  const handleAddMemory = () => {
    const value = memoryDraft.trim();
    if (!value || !activeChat) return;

    const memory = createMemory({
      scope: 'chat',
      chatId: activeChat.id,
      kind: 'note',
      content: value,
      sourceMessageId: null,
      confidence: 0.9,
      pinned: false,
    });

    setState((current) => upsertMemory(current, memory));
    setMemoryDraft('');
  };

  const handleDeleteMemory = (memoryId: string) => {
    setState((current) => removeMemory(current, memoryId));
  };

  const handleTogglePinMemory = (memory: MemoryItem) => {
    setState((current) => upsertMemory(current, { ...memory, pinned: !memory.pinned, updatedAt: new Date().toISOString() }));
  };

  const handleMoveMemoryScope = (memory: MemoryItem, scope: 'global' | 'chat') => {
    if (!activeChat) return;
    setState((current) => upsertMemory(current, {
      ...memory,
      scope,
      chatId: scope === 'chat' ? activeChat.id : null,
      updatedAt: new Date().toISOString(),
    }));
  };

  const handleClearChat = () => {
    if (!activeChat) return;
    if (!window.confirm('Remover todas as mensagens deste chat?')) return;
    setState((current) => ({
      ...current,
      messages: current.messages.filter((message) => message.chatId !== activeChat.id),
      chats: current.chats.map((chat) => (
        chat.id === activeChat.id
          ? { ...chat, summary: 'Conversa pronta para começar.', updatedAt: new Date().toISOString() }
          : chat
      )),
    }));
  };

  const handleDeleteMessage = (message: ChatMessage) => {
    if (!activeChat || message.chatId !== activeChat.id) return;
    if (!window.confirm('Apagar esta mensagem?')) return;

    setState((current) => {
      const nextMessages = current.messages.filter((item) => item.id !== message.id);
      const remaining = nextMessages.filter((item) => item.chatId === activeChat.id);
      const summary = summarizeChatMessages(remaining);
      return {
        ...current,
        messages: nextMessages,
        chats: current.chats.map((chat) => (
          chat.id === activeChat.id
            ? { ...chat, summary, updatedAt: new Date().toISOString() }
            : chat
        )),
      };
    });
  };

  const handleDeleteChat = (chatId: string) => {
    if (!window.confirm('Apagar este chat e toda a sua memória?')) return;
    setState((current) => {
      const next = removeChat(current, chatId);
      if (next.chats.length) return next;
      const fallbackWorkspace = next.workspaces[0];
      if (!fallbackWorkspace) return next;
      const freshChat = createChat(fallbackWorkspace.id, 'Novo chat');
      return {
        ...next,
        chats: [freshChat],
        activeWorkspaceId: fallbackWorkspace.id,
        activeChatId: freshChat.id,
      };
    });
  };

  const handleRenameChat = () => {
    if (!activeChat) return;
    const nextTitle = window.prompt('Nome do chat', activeChat.title)?.trim();
    if (!nextTitle) return;
    setState((current) => ({
      ...current,
      chats: current.chats.map((chat) => (chat.id === activeChat.id ? { ...chat, title: nextTitle, updatedAt: new Date().toISOString() } : chat)),
    }));
  };

  const handlePinChat = () => {
    if (!activeChat) return;
    setState((current) => pinChat(current, activeChat.id, !activeChat.pinned));
  };

  const handleArchiveChat = () => {
    if (!activeChat) return;
    setState((current) => archiveChat(current, activeChat.id));
  };

  const handleUnarchiveChat = () => {
    if (!activeChat) return;
    setState((current) => unarchiveChat(current, activeChat.id));
  };

  const handleExportChat = () => {
    if (!activeChat) return;
    const exportPayload = {
      exportedAt: new Date().toISOString(),
      workspace: activeWorkspace,
      chat: activeChat,
      messages: chatMessages,
      memories: state.memories.filter((memory) => memory.scope === 'global' || memory.chatId === activeChat.id),
    };

    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${normalizeText(activeChat.title || 'chat').replace(/\s+/g, '-') || 'chat'}-export.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const handleExportBackup = () => {
    const exportPayload = {
      kind: 'hermes-chat-studio-backup',
      exportedAt: new Date().toISOString(),
      state,
    };

    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const workspaceLabel = sanitizeFileStem(activeWorkspace?.name ?? 'hermes');
    link.download = `${workspaceLabel}-chat-backup-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const handleImportBackupClick = () => {
    stateBackupInputRef.current?.click();
  };

  const handleImportBackup = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    let parsed: unknown;
    try {
      parsed = JSON.parse(await file.text());
    } catch {
      window.alert('O arquivo selecionado não é um backup JSON válido.');
      return;
    }

    const imported = normalizeImportedAppState(parsed);
    if (!imported) {
      window.alert('Esse arquivo não parece um backup do Hermes Chat Studio.');
      return;
    }

    const confirmRestore = window.confirm('Restaurar este backup vai substituir os chats e memórias atuais neste dispositivo. Continuar?');
    if (!confirmRestore) return;

    try {
      const response = await api.saveChatState(imported, chatStateRevisionRef.current, true);
      chatStateRevisionRef.current = response.revision;
      lastSyncedSnapshotRef.current = JSON.stringify(imported);
      saveAppState(imported);
      setState(imported);
      setDraft('');
      setMemoryDraft('');
      setUploads([]);
      setPendingChatId(null);
      setShowJumpToLatest(false);
      setLayout((current) => ({ ...current, mobilePanel: null }));
      window.alert('Backup restaurado com sucesso.');
    } catch {
      window.alert('Não foi possível restaurar o backup agora.');
    }
  };

  const toggleMode = (mode: Exclude<OutputMode, 'text'>) => {
    setSelectedModes((current) => {
      const next = current.includes(mode) ? current.filter((item) => item !== mode) : [...current, mode];
      return next.includes('text') ? next : ['text', ...next];
    });
  };

  const clearRecordingState = () => {
    if (recordingTimerRef.current) {
      window.clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    recordingStreamRef.current = null;
    recordingMediaRecorderRef.current = null;
    recordingChunksRef.current = [];
    setIsRecordingAudio(false);
    setRecordingSeconds(0);
  };

  const startAudioRecording = async () => {
    if (isRecordingAudio) return;
    setRecordingError('');

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setRecordingError('Seu navegador não suporta gravação de áudio.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
      const mimeType = mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      recordingStreamRef.current = stream;
      recordingMediaRecorderRef.current = recorder;
      recordingChunksRef.current = [];
      setRecordingSeconds(0);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setRecordingError('Não foi possível gravar o áudio.');
        clearRecordingState();
      };

      recorder.onstop = async () => {
        const chunks = [...recordingChunksRef.current];
        const type = mimeType ?? 'audio/webm';
        clearRecordingState();

        if (!chunks.length) return;

        try {
          const blob = new Blob(chunks, { type });
          const url = await fileToDataUrl(new File([blob], `gravação-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`, { type: blob.type || type }));
          setUploads((current) => [
            ...current,
            {
              id: createId('upload'),
              name: `gravação-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`,
              mimeType: blob.type || type,
              size: blob.size,
              url,
            },
          ]);
        } catch {
          setRecordingError('Gravamos o áudio, mas não foi possível converter o arquivo.');
        }
      };

      recorder.start();
      setIsRecordingAudio(true);
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingSeconds((value) => value + 1);
      }, 1000);
    } catch (error) {
      clearRecordingState();
      setRecordingError(error instanceof Error ? error.message : 'Não foi possível acessar o microfone.');
    }
  };

  const stopAudioRecording = () => {
    if (recordingMediaRecorderRef.current?.state === 'recording') {
      recordingMediaRecorderRef.current.stop();
    }
  };

  const handleToggleAudioRecording = () => {
    if (isRecordingAudio) {
      stopAudioRecording();
      return;
    }
    void startAudioRecording();
  };

  const sendMessage = async () => {
    if (!activeChat || pendingChatId || (!draft.trim() && uploads.length === 0)) return;

    const activeChatId = activeChat.id;
    const currentChat = activeChat;
    const hasImageUpload = uploads.some((upload) => upload.mimeType.startsWith('image/'));
    const hasAudioUpload = uploads.some((upload) => upload.mimeType.startsWith('audio/'));
    const userText = draft.trim() || (hasAudioUpload && !hasImageUpload ? 'áudio enviado' : hasImageUpload ? 'imagem enviada' : 'mídia enviada');
    const userMessageId = createId('msg');
    const attachments = uploads.map((upload) => createAttachmentRecord({
      chatId: activeChatId,
      messageId: userMessageId,
      name: upload.name,
      mimeType: upload.mimeType,
      url: upload.url,
      size: upload.size,
    }));

    const userMessage = createUserMessage({
      chatId: activeChatId,
      content: draft.trim() || '[mídia enviada]',
      attachments,
    });

    const fallbackDraft = buildAssistantDraft({
      userText,
      chat: currentChat,
      memories: state.memories,
      attachments,
      selectedModes,
      _messages: chatMessages,
    });

    const history = serializeHistoryForBackend(chatMessages, activeChatId);
    const memoryContext = buildMemoryContext(state.memories, activeChatId);
    const systemMessage = buildSystemPrompt({
      workspaceName: activeWorkspace?.name,
      chatTitle: currentChat.title,
      memoryContext,
      selectedModes,
    });

    setState((current) => {
      let next = appendMessage(current, userMessage);
      next = updateChatSummary(next, activeChatId, userText.slice(0, 160));
      next = {
        ...next,
        chats: next.chats.map((chat) => (
          chat.id === activeChatId && isGenericTitle(chat.title)
            ? { ...chat, title: deriveChatTitle(userText), updatedAt: new Date().toISOString() }
            : chat
        )),
      };

      const suggestions = dedupeMemories(next.memories, deriveMemorySuggestions({
        chatId: activeChatId,
        messageId: userMessage.id,
        userText,
      }));

      return {
        ...next,
        memories: [...next.memories, ...suggestions],
        chats: next.chats.map((chat) => (chat.id === activeChatId ? { ...chat, updatedAt: new Date().toISOString() } : chat)),
      };
    });

    setDraft('');
    setUploads([]);
    setPendingChatId(activeChatId);

    let assistantContent = '';
    let assistantSegments = fallbackDraft.segments;
    try {
      const response = await api.chat({
        chat_id: activeChatId,
        chat_title: currentChat.title,
        workspace_name: activeWorkspace?.name ?? undefined,
        system_message: systemMessage,
        conversation_history: history,
        current_user_message: userText,
        attachments: uploads.map((upload) => ({
          name: upload.name,
          mimeType: upload.mimeType,
          size: upload.size,
          url: upload.url,
        })),
        selected_modes: selectedModes,
        memory_context: memoryContext,
      });
      assistantContent = (response.reply || '').trim();
      if (!assistantContent) {
        assistantContent = fallbackDraft.content;
        assistantSegments = fallbackDraft.segments;
      } else {
        assistantSegments = [];
      }
    } catch {
      assistantContent = fallbackDraft.content;
      assistantSegments = fallbackDraft.segments;
    }

    setState((current) => {
      if (!current.chats.some((chat) => chat.id === activeChatId)) return current;
      const assistantMessage = createAssistantMessage({
        chatId: activeChatId,
        content: assistantContent,
        segments: assistantSegments,
      });
      let next = appendMessage(current, assistantMessage);
      next = updateChatSummary(next, activeChatId, stripMarkdown(assistantContent).slice(0, 180));
      return {
        ...next,
        chats: next.chats.map((chat) => (
          chat.id === activeChatId
            ? { ...chat, updatedAt: new Date().toISOString() }
            : chat
        )),
      };
    });

    setPendingChatId(null);
  };

  const handleLogout = async () => {
    await api.logout();
    setAuthSession(null);
  };

  if (!authReady) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[color:var(--color-background)] text-muted-foreground">
        carregando autenticação...
      </div>
    );
  }

  if (!authSession) {
    return <LoginScreen onLogin={async (payload) => {
      const session = await api.login(payload);
      setAuthSession(session);
    }} />;
  }

  if (!chatStateReady) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[color:var(--color-background)] text-muted-foreground">
        carregando chats salvos...
      </div>
    );
  }

  const workspaceCards = state.workspaces;

  const desktopShell = layout.focusMode ? (
    <div className="relative mx-auto grid h-full min-h-0 flex-1 gap-3 px-3 py-3 lg:grid-cols-1 lg:px-6 lg:py-6">
      <section className="min-h-0 lg:col-span-full">
        <ChatPane
          activeWorkspace={activeWorkspace}
          activeChat={activeChat}
          chatMessages={chatMessages}
          selectedModes={selectedModes}
          onToggleMode={toggleMode}
          draft={draft}
          setDraft={setDraft}
          uploads={uploads}
          onRemoveUpload={(id) => setUploads((current) => current.filter((item) => item.id !== id))}
          onSend={sendMessage}
          onCopyMessage={handleCopyMessage}
          onPromoteMessage={handlePromoteMessageToMemory}
          onDeleteMessage={handleDeleteMessage}
          onCopyLastReply={handleCopyLastReply}
          onPinChat={handlePinChat}
          onArchiveChat={activeChat?.archivedAt ? handleUnarchiveChat : handleArchiveChat}
          onDeleteChat={handleDeleteChat}
          onRenameChat={handleRenameChat}
          onClearChat={handleClearChat}
          onExportChat={handleExportChat}
          isPending={Boolean(pendingChatId)}
          showJumpToLatest={showJumpToLatest}
          onJumpToLatest={scrollChatToBottom}
          onDropFiles={handleFiles}
          onPasteFiles={async (event) => {
            const files = Array.from(event.clipboardData.files).filter((file) => file.type.startsWith('image/') || file.type.startsWith('audio/'));
            if (!files.length) return;
            event.preventDefault();
            await handleFiles(files);
          }}
          onToggleRecording={handleToggleAudioRecording}
          isRecordingAudio={isRecordingAudio}
          recordingSeconds={recordingSeconds}
          recordingError={recordingError}
          canSend={canSend}
          imageInputRef={imageInputRef}
          audioInputRef={audioInputRef}
          scrollContainerRef={scrollRef}
          onScroll={handleChatScroll}
        />
      </section>
    </div>
  ) : (
    <div
      className="relative mx-auto grid h-full max-w-[1600px] min-h-0 flex-1 gap-3 px-3 py-3 lg:grid-cols-[var(--left-rail-width)_minmax(0,1fr)_var(--right-rail-width)] lg:px-6 lg:py-6"
      style={{ '--left-rail-width': desktopLeftCollapsed ? '6.5rem' : '19rem', '--right-rail-width': desktopRightCollapsed ? '6.5rem' : '21.5rem' } as any}
    >
      <aside className="hidden min-h-0 lg:flex">
        <SidebarRail
          collapsed={desktopLeftCollapsed}
          activeWorkspace={activeWorkspace}
          workspaces={workspaceCards}
          chats={visibleChats}
          activeChatId={state.activeChatId}
          search={search}
          setSearch={setSearch}
          showArchivedChats={showArchivedChats}
          setShowArchivedChats={setShowArchivedChats}
          workspaceDraft={workspaceDraft}
          setWorkspaceDraft={setWorkspaceDraft}
          onCreateWorkspace={handleCreateWorkspace}
          onCreateChat={handleCreateChat}
          onSelectWorkspace={handleSelectWorkspace}
          onSelectChat={handleSelectChat}
          onPinChat={(chatId, pinned) => setState((current) => pinChat(current, chatId, pinned))}
          onArchiveChat={(chatId) => setState((current) => archiveChat(current, chatId))}
          onUnarchiveChat={(chatId) => setState((current) => unarchiveChat(current, chatId))}
          onDeleteChat={handleDeleteChat}
          sessionEmail={authSession.email}
          activeWorkspaceChatsCount={activeWorkspaceChatsCount}
          archivedCount={archivedCount}
          onToggleCollapsed={toggleLeftRail}
        />
      </aside>

      <section className="min-h-0">
        <ChatPane
          activeWorkspace={activeWorkspace}
          activeChat={activeChat}
          chatMessages={chatMessages}
          selectedModes={selectedModes}
          onToggleMode={toggleMode}
          draft={draft}
          setDraft={setDraft}
          uploads={uploads}
          onRemoveUpload={(id) => setUploads((current) => current.filter((item) => item.id !== id))}
          onSend={sendMessage}
          onCopyMessage={handleCopyMessage}
          onPromoteMessage={handlePromoteMessageToMemory}
          onDeleteMessage={handleDeleteMessage}
          onCopyLastReply={handleCopyLastReply}
          onPinChat={handlePinChat}
          onArchiveChat={activeChat?.archivedAt ? handleUnarchiveChat : handleArchiveChat}
          onDeleteChat={handleDeleteChat}
          onRenameChat={handleRenameChat}
          onClearChat={handleClearChat}
          onExportChat={handleExportChat}
          isPending={Boolean(pendingChatId)}
          showJumpToLatest={showJumpToLatest}
          onJumpToLatest={scrollChatToBottom}
          onDropFiles={handleFiles}
          onPasteFiles={async (event) => {
            const files = Array.from(event.clipboardData.files).filter((file) => file.type.startsWith('image/') || file.type.startsWith('audio/'));
            if (!files.length) return;
            event.preventDefault();
            await handleFiles(files);
          }}
          onToggleRecording={handleToggleAudioRecording}
          isRecordingAudio={isRecordingAudio}
          recordingSeconds={recordingSeconds}
          recordingError={recordingError}
          canSend={canSend}
          imageInputRef={imageInputRef}
          audioInputRef={audioInputRef}
          scrollContainerRef={scrollRef}
          onScroll={handleChatScroll}
        />
      </section>

      <aside className="hidden min-h-0 lg:flex">
        <MemoryRail
          collapsed={desktopRightCollapsed}
          activeChat={activeChat}
          globalMemories={globalMemories}
          chatMemories={chatMemories}
          memoryDraft={memoryDraft}
          setMemoryDraft={setMemoryDraft}
          onAddMemory={handleAddMemory}
          onDeleteMemory={handleDeleteMemory}
          onTogglePin={handleTogglePinMemory}
          onMoveScope={handleMoveMemoryScope}
          onClearChat={handleClearChat}
          onExportState={handleExportBackup}
          onImportStateClick={handleImportBackupClick}
          onImportState={handleImportBackup}
          stateBackupInputRef={stateBackupInputRef}
          onToggleCollapsed={toggleRightRail}
        />
      </aside>
    </div>
  );

  const mobileShell = (
    <div className="flex min-h-0 flex-1 flex-col gap-3 px-3 py-3 lg:hidden">
      <div className="min-h-0 flex-1">
        <ChatPane
          activeWorkspace={activeWorkspace}
          activeChat={activeChat}
          chatMessages={chatMessages}
          selectedModes={selectedModes}
          onToggleMode={toggleMode}
          draft={draft}
          setDraft={setDraft}
          uploads={uploads}
          onRemoveUpload={(id) => setUploads((current) => current.filter((item) => item.id !== id))}
          onSend={sendMessage}
          onCopyMessage={handleCopyMessage}
          onPromoteMessage={handlePromoteMessageToMemory}
          onDeleteMessage={handleDeleteMessage}
          onCopyLastReply={handleCopyLastReply}
          onPinChat={handlePinChat}
          onArchiveChat={activeChat?.archivedAt ? handleUnarchiveChat : handleArchiveChat}
          onDeleteChat={handleDeleteChat}
          onRenameChat={handleRenameChat}
          onClearChat={handleClearChat}
          onExportChat={handleExportChat}
          isPending={Boolean(pendingChatId)}
          showJumpToLatest={showJumpToLatest}
          onJumpToLatest={scrollChatToBottom}
          onDropFiles={handleFiles}
          onPasteFiles={async (event) => {
            const files = Array.from(event.clipboardData.files).filter((file) => file.type.startsWith('image/') || file.type.startsWith('audio/'));
            if (!files.length) return;
            event.preventDefault();
            await handleFiles(files);
          }}
          onToggleRecording={handleToggleAudioRecording}
          isRecordingAudio={isRecordingAudio}
          recordingSeconds={recordingSeconds}
          recordingError={recordingError}
          canSend={canSend}
          imageInputRef={imageInputRef}
          audioInputRef={audioInputRef}
          scrollContainerRef={scrollRef}
          onScroll={handleChatScroll}
          compactHeader
        />
      </div>

      <nav className="grid grid-cols-4 gap-2 rounded-[1.5rem] border border-border bg-[color:var(--color-card)]/88 px-2 pt-2 pb-[calc(0.5rem+env(safe-area-inset-bottom))] shadow-[0_18px_50px_rgba(0,0,0,0.12)] backdrop-blur-xl">
        {([
          ['chat', 'Chat', MessageSquarePlus],
          ['chats', 'Chats', Menu],
          ['memory', 'Memória', BrainCircuit],
          ['actions', 'Ações', Sparkles],
        ] as const).map(([key, label, Icon]) => {
          const active = key === 'chat' ? layout.mobilePanel === null : layout.mobilePanel === key;
          return (
            <button
              key={key}
              type="button"
              onClick={() => {
                if (key === 'chat') {
                  closeMobilePanel();
                  return;
                }
                openMobilePanel(key);
              }}
              className={cn(
                'flex min-h-14 flex-col items-center justify-center gap-1 rounded-[1rem] border px-2 py-2 text-[0.65rem] uppercase tracking-[0.16em] transition-colors',
                active
                  ? 'border-foreground bg-foreground text-background'
                  : 'border-transparent bg-transparent text-muted-foreground hover:border-foreground/25 hover:text-foreground',
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          );
        })}
      </nav>

      <MobileSheet
        open={layout.mobilePanel === 'chats'}
        title="chats e workspaces"
        description="Escolha um chat, crie outro ou reorganize os assuntos sem sair do celular."
        onClose={closeMobilePanel}
      >
        <SidebarRail
          collapsed={false}
          activeWorkspace={activeWorkspace}
          workspaces={workspaceCards}
          chats={visibleChats}
          activeChatId={state.activeChatId}
          search={search}
          setSearch={setSearch}
          showArchivedChats={showArchivedChats}
          setShowArchivedChats={setShowArchivedChats}
          workspaceDraft={workspaceDraft}
          setWorkspaceDraft={setWorkspaceDraft}
          onCreateWorkspace={handleCreateWorkspace}
          onCreateChat={handleCreateChat}
          onSelectWorkspace={handleSelectWorkspace}
          onSelectChat={handleSelectChat}
          onPinChat={(chatId, pinned) => setState((current) => pinChat(current, chatId, pinned))}
          onArchiveChat={(chatId) => setState((current) => archiveChat(current, chatId))}
          onUnarchiveChat={(chatId) => setState((current) => unarchiveChat(current, chatId))}
          onDeleteChat={handleDeleteChat}
          sessionEmail={authSession.email}
          activeWorkspaceChatsCount={activeWorkspaceChatsCount}
          archivedCount={archivedCount}
          onToggleCollapsed={closeMobilePanel}
        />
      </MobileSheet>

      <MobileSheet
        open={layout.mobilePanel === 'memory'}
        title="memória"
        description="Revise, fixe e mova memórias entre global e chat com um gesto curto."
        onClose={closeMobilePanel}
      >
        <MemoryRail
          collapsed={false}
          activeChat={activeChat}
          globalMemories={globalMemories}
          chatMemories={chatMemories}
          memoryDraft={memoryDraft}
          setMemoryDraft={setMemoryDraft}
          onAddMemory={handleAddMemory}
          onDeleteMemory={handleDeleteMemory}
          onTogglePin={handleTogglePinMemory}
          onMoveScope={handleMoveMemoryScope}
          onClearChat={handleClearChat}
          onExportState={handleExportBackup}
          onImportStateClick={handleImportBackupClick}
          onImportState={handleImportBackup}
          stateBackupInputRef={stateBackupInputRef}
          onToggleCollapsed={closeMobilePanel}
        />
      </MobileSheet>

      <MobileSheet
        open={layout.mobilePanel === 'actions'}
        title="ações rápidas"
        description="Comandos do chat atual, foco e sessão em um único painel."
        onClose={closeMobilePanel}
      >
        <div className="grid gap-3">
          <div className="grid gap-2 sm:grid-cols-2">
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { handleCreateChat(); closeMobilePanel(); }}>
              <MessageSquarePlus className="h-4 w-4" />
              novo chat
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { handleCreateWorkspace(); closeMobilePanel(); }} disabled={!workspaceDraft.trim()}>
              <FolderPlus className="h-4 w-4" />
              novo workspace
            </Button>
            <Button type="button" variant={layout.focusMode ? 'default' : 'outline'} className="h-11 justify-start px-3" onClick={() => { setFocusMode(!layout.focusMode); closeMobilePanel(); }}>
              <Sparkles className="h-4 w-4" />
              {layout.focusMode ? 'sair do foco' : 'modo foco'}
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { handleExportChat(); closeMobilePanel(); }} disabled={!activeChat}>
              <Download className="h-4 w-4" />
              exportar chat
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { handleCopyLastReply(); closeMobilePanel(); }} disabled={!chatMessages.some((message) => message.role === 'assistant')}>
              <Copy className="h-4 w-4" />
              copiar última resposta
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { if (!activeChat) return; handleRenameChat(); closeMobilePanel(); }} disabled={!activeChat}>
              <PencilLine className="h-4 w-4" />
              renomear chat
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { if (!activeChat) return; handlePinChat(); closeMobilePanel(); }} disabled={!activeChat}>
              {activeChat?.pinned ? <PinOff className="h-4 w-4" /> : <Pin className="h-4 w-4" />}
              {activeChat?.pinned ? 'desafixar chat' : 'fixar chat'}
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { if (!activeChat) return; (activeChat.archivedAt ? handleUnarchiveChat : handleArchiveChat)(); closeMobilePanel(); }} disabled={!activeChat}>
              <Archive className="h-4 w-4" />
              {activeChat?.archivedAt ? 'desarquivar' : 'arquivar'}
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3 text-destructive" onClick={() => { if (!activeChat) return; handleClearChat(); closeMobilePanel(); }} disabled={!activeChat}>
              <Trash2 className="h-4 w-4" />
              limpar chat
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3 text-destructive" onClick={() => { if (!activeChat) return; handleDeleteChat(activeChat.id); closeMobilePanel(); }} disabled={!activeChat}>
              <Trash2 className="h-4 w-4" />
              apagar chat
            </Button>
            <Button type="button" variant="outline" className="h-11 justify-start px-3" onClick={() => { void handleLogout(); closeMobilePanel(); }}>
              <LogOut className="h-4 w-4" />
              sair
            </Button>
          </div>

          <div className="grid gap-2 rounded-[1.1rem] border border-border bg-card/65 p-3 text-sm text-muted-foreground">
            <p>• O chat é a superfície principal no celular.</p>
            <p>• Chats e memória abrem como painéis dedicados.</p>
            <p>• O modo foco reduz ruído visual e navegação.</p>
          </div>
        </div>
      </MobileSheet>
    </div>
  );

  return (
    <div className="relative flex h-[100dvh] flex-col overflow-hidden bg-[color:var(--color-background)] text-[color:var(--color-foreground)]">
      <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(circle_at_10%_10%,rgba(251,191,36,0.14),transparent_30%),radial-gradient(circle_at_90%_8%,rgba(59,130,246,0.08),transparent_24%),radial-gradient(circle_at_50%_100%,rgba(16,185,129,0.06),transparent_28%)]" />
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] mix-blend-multiply bg-[repeating-linear-gradient(90deg,transparent_0,transparent_23px,rgba(15,23,42,0.06)_24px),repeating-linear-gradient(0deg,transparent_0,transparent_23px,rgba(15,23,42,0.04)_24px)]" />

      <header className="relative z-40 border-b border-[color:var(--color-border)] bg-[color:var(--color-background)]/82 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] items-center gap-3 px-3 py-3 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-[1rem] border border-[color:var(--color-border)] bg-[color:var(--color-card)]/70 shadow-[0_14px_48px_rgba(0,0,0,0.18)]">
              <Sparkles className="h-5 w-5 text-[color:var(--color-foreground)]" />
            </div>
            <div className="min-w-0">
              <p className="font-expanded text-[0.72rem] uppercase tracking-[0.25em] text-[color:var(--color-muted-foreground)]">
                Hermes chat studio
              </p>
              <h1 className="truncate font-display text-lg text-[color:var(--color-foreground)]/95 sm:text-2xl">
                Workspace pessoal com memória persistente
              </h1>
            </div>
          </div>

          <div className="hidden flex-1 items-center justify-center gap-2 xl:flex">
            <div className="flex min-w-[18rem] max-w-[30rem] flex-1 items-center gap-2 rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-card)]/60 px-3 py-2">
              <Search className="h-4 w-4 text-[color:var(--color-muted-foreground)]" />
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Buscar chats, mensagens e memórias"
                className="w-full bg-transparent text-sm text-[color:var(--color-foreground)] outline-none placeholder:text-[color:var(--color-muted-foreground)]"
              />
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">{state.messages.length} mensagens</Badge>
              <Badge variant="outline">{state.memories.length} memórias</Badge>
              <Badge variant="success">local-first</Badge>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <div className="hidden items-center gap-2 xl:flex">
              <Badge variant="outline" className="max-w-[18rem] truncate">{authSession.email}</Badge>
              <Button type="button" variant="outline" className="h-10 px-3" onClick={() => void handleLogout()}>
                <LogOut className="h-4 w-4" />
                sair
              </Button>
            </div>
            <Button type="button" variant={layout.focusMode ? 'default' : 'outline'} className="hidden h-10 px-3 lg:inline-flex" onClick={() => setFocusMode(!layout.focusMode)}>
              {layout.focusMode ? 'sair do foco' : 'foco'}
            </Button>
            <Button type="button" variant="outline" className="hidden h-10 px-3 lg:inline-flex" onClick={toggleLeftRail}>
              <Menu className="h-4 w-4" />
              chats
            </Button>
            <Button type="button" variant="outline" className="hidden h-10 px-3 lg:inline-flex" onClick={toggleRightRail}>
              <PanelRightClose className="h-4 w-4" />
              memória
            </Button>
            <Button type="button" className="h-10 px-4" onClick={handleCreateChat}>
              <Plus className="h-4 w-4" />
              novo chat
            </Button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto flex min-h-0 w-full max-w-[1600px] flex-1 flex-col overflow-hidden">
        <div className="hidden min-h-0 flex-1 lg:flex">
          {desktopShell}
        </div>
        {mobileShell}
      </main>
    </div>
  );
}

export default App;
