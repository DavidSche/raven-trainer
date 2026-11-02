import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Global application store.
 *
 * Currently manages:
 * - sidebarCollapsed: whether the left navigation sidebar is in collapsed (icon-only) mode.
 *   Persisted to localStorage so the user's preference survives page reloads.
 *
 * This store is intentionally small. Page-specific state (datasets, training runs, etc.)
 * is managed locally in each page component or via React Query for server state.
 * As the app grows, additional cross-cutting UI preferences (theme, density, etc.)
 * should be added here rather than introducing a second global store.
 */
interface AppState {
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
    }),
    {
      name: 'raven-trainer-ui',
      // Only persist the sidebar preference, not the setter functions
      partialize: (state) => ({ sidebarCollapsed: state.sidebarCollapsed }),
    },
  ),
);
