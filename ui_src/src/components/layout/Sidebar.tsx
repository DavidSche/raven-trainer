import React from 'react';
import { NavLink } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  LayoutDashboard, Database, BookTemplate, RefreshCw, FlaskConical, Box, AlertTriangle, FolderOpen,
  FileText, ListTodo, CirclePlay, ChevronLeft, ChevronRight,
  Cpu, Activity, Globe, HardDrive, Settings,
} from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, labelKey: 'nav.dashboard' },
  { to: '/storage', icon: HardDrive, labelKey: 'nav.storage' },
  { to: '/ingestion-profiles', icon: RefreshCw, labelKey: 'nav.ingestionProfiles' },
  { to: '/datasets', icon: Database, labelKey: 'nav.datasets' },
  { to: '/recipes', icon: BookTemplate, labelKey: 'nav.recipes' },
  { to: '/workspaces', icon: FolderOpen, labelKey: 'nav.workspaces' },
  { to: '/pipeline', icon: Activity, labelKey: 'nav.pipeline' },
  { to: '/jobs', icon: CirclePlay, labelKey: 'nav.jobs' },
  { to: '/training', icon: FlaskConical, labelKey: 'nav.training' },
  { to: '/models', icon: Box, labelKey: 'nav.models' },
  { to: '/tasks', icon: ListTodo, labelKey: 'nav.tasks' },
  { to: '/misdata', icon: AlertTriangle, labelKey: 'nav.feedback' },
  { to: '/reports', icon: FileText, labelKey: 'nav.reports' },
  { to: '/settings', icon: Settings, labelKey: 'nav.settings' },
];

export function Sidebar() {
  const { t, i18n } = useTranslation();
  const { sidebarCollapsed, toggleSidebar } = useAppStore();

  const toggleLang = () => {
    i18n.changeLanguage(i18n.language === 'zh' ? 'en' : 'zh');
  };

  return (
    <aside
      className={`bg-gray-900 text-white flex flex-col transition-all duration-200 ${
        sidebarCollapsed ? 'w-16' : 'w-56'
      }`}
      aria-label={t('common.sidebar')}
    >
      <div className="flex items-center gap-2 px-4 h-14 border-b border-gray-700 shrink-0">
        <Cpu className="w-6 h-6 text-blue-400 shrink-0" />
        {!sidebarCollapsed && <span className="font-semibold text-sm truncate">{t('app.title')}</span>}
      </div>

      <nav className="flex-1 py-2 overflow-y-auto" aria-label={t('common.mainNavigation')}>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            aria-label={t(item.labelKey)}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`
            }
          >
            <item.icon className="w-5 h-5 shrink-0" />
            {!sidebarCollapsed && <span className="truncate">{t(item.labelKey)}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-gray-700">
        {!sidebarCollapsed && (
          <button
            onClick={toggleLang}
            className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-gray-300 hover:bg-gray-800 hover:text-white transition-colors"
            aria-label={i18n.language === 'zh' ? t('lang.switchTo') : t('lang.switchToZh')}
          >
            <Globe className="w-5 h-5 shrink-0" />
            <span>{i18n.language === 'zh' ? t('lang.en') : t('lang.zh')}</span>
          </button>
        )}
        <button
          onClick={toggleSidebar}
          className="flex items-center justify-center w-full h-10 text-gray-400 hover:text-white shrink-0"
          aria-label={sidebarCollapsed ? t('common.expandSidebar') : t('common.collapseSidebar')}
          aria-expanded={!sidebarCollapsed}
        >
          {sidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </aside>
  );
}



