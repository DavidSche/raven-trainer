import React from 'react';
import { Menu } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAppStore } from '../../stores/appStore';

export function TopBar() {
  const { t } = useTranslation();
  const { toggleSidebar } = useAppStore();

  return (
    <header className="h-14 bg-white border-b flex items-center px-4 gap-3 shrink-0">
      <button
        onClick={toggleSidebar}
        className="text-gray-500 hover:text-gray-700 lg:hidden"
        aria-label={t('common.toggleSidebar')}
      >
        <Menu className="w-5 h-5" />
      </button>
      <div className="text-sm text-gray-500">{t('topbar.version')}</div>
      <div className="flex-1" />
    </header>
  );
}
