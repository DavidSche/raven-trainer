import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import {
  Database, Image, FlaskConical, Box, Layers, TrendingUp,
} from 'lucide-react';
import { fetchSystemStatus, fetchActivities } from '../api/datasets';
import type { SystemStatus, ActivityItem } from '../types';

export default function DashboardPage() {
  const { t } = useTranslation();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [activities, setActivities] = useState<ActivityItem[]>([]);

  useEffect(() => {
    fetchSystemStatus().then(setStatus).catch((e) => console.error('Failed to load system status:', e));
    fetchActivities(10).then(setActivities).catch((e) => console.error('Failed to load activities:', e));
  }, []);

  const colorMap: Record<string, { bg: string; text: string }> = {
    blue:   { bg: 'bg-blue-50',   text: 'text-blue-600' },
    violet: { bg: 'bg-violet-50', text: 'text-violet-600' },
    green:  { bg: 'bg-green-50',  text: 'text-green-600' },
    orange: { bg: 'bg-orange-50', text: 'text-orange-600' },
  };

  const cards = [
    { label: t('dashboard.datasets'), value: status?.datasets ?? '-', icon: Database, color: 'blue', to: '/datasets' },
    { label: t('dashboard.images'), value: status?.total_images?.toLocaleString() ?? '-', icon: Image, color: 'violet', to: '/datasets' },
    { label: t('dashboard.versions'), value: status?.total_versions ?? '-', icon: Layers, color: 'green', to: '/datasets' },
    { label: t('dashboard.models'), value: status?.models ?? '-', icon: Box, color: 'orange', to: '/models' },
  ];

  return (
    <div className="p-6 max-w-6xl">
      <h1 className="text-xl font-semibold mb-6">{t('dashboard.title')}</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {cards.map((card) => (
          <Link key={card.label} to={card.to} className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-sm transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <span className="text-2xl font-bold">{card.value}</span>
              <div className={`p-2 rounded-lg ${colorMap[card.color]?.bg ?? 'bg-gray-50'}`}>
                <card.icon className={`w-5 h-5 ${colorMap[card.color]?.text ?? 'text-gray-600'}`} />
              </div>
            </div>
            <div className="text-sm text-gray-500">{card.label}</div>
          </Link>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="font-medium text-sm mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" /> {t('dashboard.recentActivities')}
          </h2>
          {activities.length === 0 ? (
            <div className="text-gray-400 text-sm py-6 text-center">{t('dashboard.noActivities')}</div>
          ) : (
            <div className="space-y-2">
              {activities.map((act, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm py-1.5 border-b border-gray-50 last:border-0">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${
                      act.status === 'completed' ? 'bg-green-500' :
                      act.status === 'running' ? 'bg-blue-500' : 'bg-yellow-500'
                    }`} />
                    <span className="truncate text-gray-700">{act.action}</span>
                  </div>
                  <span className="text-xs text-gray-400 shrink-0 ml-2">{act.time?.slice(0, 16).replace('T', ' ')}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="font-medium text-sm mb-3">{t('dashboard.quickLinks')}</h2>
          <div className="grid grid-cols-2 gap-2">
            <Link to="/jobs" className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 text-sm">{t('nav.jobs')}</Link>
            <Link to="/training" className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 text-sm">{t('nav.training')}</Link>
            <Link to="/reports" className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 text-sm">{t('nav.reports')}</Link>
            <Link to="/tasks" className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 text-sm">{t('nav.tasks')}</Link>
          </div>
          <div className="mt-4 text-xs text-gray-400">
            {t('dashboard.dataRoot')}: {status?.data_root ?? 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );
}
