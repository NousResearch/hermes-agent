import { atom } from 'nanostores'

export const $sessionSidebarVisible = atom(false)

export const toggleSessionSidebar = () => $sessionSidebarVisible.set(!$sessionSidebarVisible.get())
