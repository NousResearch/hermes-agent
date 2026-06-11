/**
 * SidebarReorder — drag-and-drop reorderable list for sidebar nav items.
 *
 * Uses @dnd-kit/sortable for reliable cross-platform drag-and-drop.
 * Supports two modes:
 * - folded=false: two independent lists (core + plugin) with separate order
 * - folded=true: single unified list
 */

import { useState, useCallback, useRef } from "react";
import {
  DndContext,
  DragOverlay,
  closestCorners,
  KeyboardSensor,
  PointerSensor,
  TouchSensor,
  useSensor,
  useSensors,
  type DragStartEvent,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Menu } from "lucide-react";
import { cn } from "@/lib/utils";

interface ReorderItem {
  id: string;
  label: string;
}

/* ── Sortable item ──────────────────────────────────────────────── */

function SortableItem({
  item,
  isOverlay = false,
}: {
  item: ReorderItem;
  isOverlay?: boolean;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: item.id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.3 : 1,
    zIndex: isDragging ? 999 : undefined,
  };

  return (
    <li
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={cn(
        "flex items-center gap-2 px-4 py-2 cursor-grab active:cursor-grabbing select-none",
        "transition-colors hover:bg-secondary/30",
        "border-b border-border last:border-b-0",
        isDragging && "shadow-lg rounded bg-background-base",
        isOverlay && "shadow-2xl rounded bg-background-base ring-2 ring-midground/30",
      )}
    >
      <Menu className="h-3.5 w-3.5 shrink-0 text-text-tertiary" />
      <span className="text-sm truncate">{item.label}</span>
    </li>
  );
}

/* ── Overlay item (drag preview) ────────────────────────────────── */

function DragOverlayItem({ item }: { item: ReorderItem }) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 px-4 py-2 cursor-grabbing select-none",
        "border-b border-border last:border-b-0",
        "shadow-2xl rounded bg-background-base ring-2 ring-midground/30",
      )}
    >
      <Menu className="h-3.5 w-3.5 shrink-0 text-text-tertiary" />
      <span className="text-sm truncate">{item.label}</span>
    </div>
  );
}

/* ── Sortable list ──────────────────────────────────────────────── */

function SidebarReorderList({
  items,
  onReorder,
  label,
}: {
  items: ReorderItem[];
  onReorder: (items: ReorderItem[]) => void;
  label?: string;
}) {
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const [activeId, setActiveId] = useState<string | null>(null);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 3 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const handleDragStart = useCallback((event: DragStartEvent) => {
    setActiveId(String(event.active.id));
  }, []);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      setActiveId(null);
      const { active, over } = event;
      if (!over || active.id === over.id) return;

      const currentItems = itemsRef.current;
      const oldIndex = currentItems.findIndex((i) => i.id === active.id);
      const newIndex = currentItems.findIndex((i) => i.id === over.id);
      if (oldIndex === -1 || newIndex === -1) return;

      onReorder(arrayMove(currentItems, oldIndex, newIndex));
    },
    [onReorder],
  );

  const activeItem = activeId ? items.find((i) => i.id === activeId) : null;

  return (
    <div className="flex flex-col gap-1">
      {label && (
        <span className="px-4 pt-2 pb-1 text-xs font-medium tracking-[0.12em] text-text-tertiary uppercase">
          {label}
        </span>
      )}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCorners}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={items.map((i) => i.id)}
          strategy={verticalListSortingStrategy}
        >
          <ul className="flex flex-col">
            {items.map((item) => (
              <SortableItem key={item.id} item={item} />
            ))}
          </ul>
        </SortableContext>
        <DragOverlay dropAnimation={null}>
          {activeItem ? <DragOverlayItem item={activeItem} /> : null}
        </DragOverlay>
      </DndContext>
    </div>
  );
}

/* ── Main component ─────────────────────────────────────────────── */

interface SidebarReorderProps {
  coreItems: ReorderItem[];
  pluginItems: ReorderItem[];
  unifiedItems: ReorderItem[];
  folded: boolean;
  onCoreReorder: (items: ReorderItem[]) => void;
  onPluginReorder: (items: ReorderItem[]) => void;
  onUnifiedReorder: (items: ReorderItem[]) => void;
  onFoldToggle: (folded: boolean) => void;
  mainItemsLabel: string;
  pluginItemsLabel: string;
  unifiedItemsLabel: string;
}

export function SidebarReorder({
  coreItems,
  pluginItems,
  unifiedItems,
  folded,
  onCoreReorder,
  onPluginReorder,
  onUnifiedReorder,
  onFoldToggle,
  mainItemsLabel,
  pluginItemsLabel,
  unifiedItemsLabel,
}: SidebarReorderProps) {
  return (
    <div className="flex flex-col gap-3">
      {/* Fold toggle */}
      <label
        className={cn(
          "flex cursor-pointer items-center justify-between gap-4",
          "px-4 py-3 transition-colors hover:bg-secondary/30",
          "border-b border-border",
        )}
      >
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium">Fold plugins into sidebar</span>
          <span className="text-xs text-text-tertiary">
            When enabled, plugin items merge into the main nav and can be reordered together.
          </span>
        </div>
        <div
          role="switch"
          aria-checked={folded}
          tabIndex={0}
          onClick={() => onFoldToggle(!folded)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onFoldToggle(!folded);
            }
          }}
          className={cn(
            "relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center",
            "rounded-full border border-current/20 transition-colors",
            folded ? "bg-midground/30" : "bg-transparent",
          )}
        >
          <span
            className={cn(
              "inline-block h-4 w-4 rounded-full transition-transform",
              "bg-midground shadow-sm",
              folded ? "translate-x-[1.375rem]" : "translate-x-1",
            )}
          />
        </div>
      </label>

      {/* Reorder lists */}
      {folded ? (
        <SidebarReorderList
          items={unifiedItems}
          onReorder={onUnifiedReorder}
          label={unifiedItemsLabel}
        />
      ) : (
        <>
          <SidebarReorderList
            items={coreItems}
            onReorder={onCoreReorder}
            label={mainItemsLabel}
          />
          {pluginItems.length > 0 && (
            <SidebarReorderList
              items={pluginItems}
              onReorder={onPluginReorder}
              label={pluginItemsLabel}
            />
          )}
        </>
      )}
    </div>
  );
}
