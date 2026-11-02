import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Settings, Info, Cpu, HardDrive } from 'lucide-react';
import { api } from '../api/client';

export default function SettingsPage() {
  const { t } = useTranslation();
  const [status, setStatus] = useState<Record<string, unknown>>({});

  useEffect(() => {
    api.get<Record<string, unknown>>('/system/status')
      .then(setStatus)
      .catch((e) => console.error('Failed to load system status:', e));
  }, []);

  return (
    <div className="p-6 max-w-4xl">
      <h1 className="text-xl font-semibold mb-6">{t('settings.title')}</h1>

      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <h2 className="font-medium mb-4 flex items-center gap-2">
          <Info className="w-4 h-4" /> {t('settings.about')}
        </h2>
        <div className="space-y-3 text-sm">
          <div className="flex justify-between py-2 border-b">
            <span className="text-gray-500">{t('settings.version')}</span>
            <span className="font-mono">{status.version as string || 'N/A'}</span>
          </div>
          <div className="flex justify-between py-2 border-b">
            <span className="text-gray-500">{t('settings.dataRoot')}</span>
            <span className="font-mono text-xs">{status.data_root as string || 'N/A'}</span>
          </div>
          <div className="flex justify-between py-2 border-b">
            <span className="text-gray-500">{t('settings.datasets')}</span>
            <span>{status.datasets as number || 0}</span>
          </div>
          <div className="flex justify-between py-2 border-b">
            <span className="text-gray-500">{t('settings.totalImages')}</span>
            <span>{(status.total_images as number || 0).toLocaleString()}</span>
          </div>
          <div className="flex justify-between py-2 border-b">
            <span className="text-gray-500">{t('settings.totalVersions')}</span>
            <span>{status.total_versions as number || 0}</span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-gray-500">{t('settings.models')}</span>
            <span>{status.models as number || 0}</span>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="font-medium mb-4 flex items-center gap-2">
          <Cpu className="w-4 h-4" /> {t('settings.consoleIntegration')}
        </h2>
        <div className="text-sm text-gray-500 space-y-2">
          <p>{t('settings.embedDesc')}</p>
          <p>{t('settings.embedHint')}</p>
          <pre className="bg-gray-50 rounded-lg p-3 text-xs font-mono mt-2">
{`window.__RAVEN_CONSOLE_EMBED = true;
window.__RAVEN_CONSOLE_BASE = true;

// Then import or load the trainer bundle
// All routes will be prefixed with /trainer/
// Sidebar and TopBar will be hidden (Raven-Console provides its own)`}
          </pre>
        </div>
      </div>
    </div>
  );
}
