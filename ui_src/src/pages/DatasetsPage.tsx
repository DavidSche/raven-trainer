import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchDatasets, createDataset } from '../api/datasets';
import { DatasetCard } from '../components/dataset/DatasetCard';
import type { Dataset } from '../types';
import { Search, Plus, X } from 'lucide-react';
import { LoadingState, ErrorState, EmptyState } from '../components/common/StateViews';

export default function DatasetsPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createTask, setCreateTask] = useState('detection');

  const { data, isLoading, error, refetch } = useQuery<Dataset[]>({
    queryKey: ['datasets'],
    queryFn: fetchDatasets,
  });

  const createMutation = useMutation({
    mutationFn: () => createDataset(createName.trim(), createTask),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setShowCreate(false);
      setCreateName('');
      setCreateTask('detection');
    },
    onError: (e) => console.error('Failed to create dataset:', e),
  });

  const handleCreate = () => {
    if (!createName.trim()) return;
    createMutation.mutate();
  };

  const handleDeleted = () => {
    queryClient.invalidateQueries({ queryKey: ['datasets'] });
  };

  const datasets = data ?? [];
  const filtered = datasets.filter((d) =>
    d.name.toLowerCase().includes(filter.toLowerCase()) ||
    d.display_name?.toLowerCase().includes(filter.toLowerCase())
  );

  const taskOptions = [
    { value: 'detection', label: t('datasets.detection') },
    { value: 'pose', label: t('datasets.pose') },
    { value: 'segmentation', label: t('datasets.segmentation') },
    { value: 'classification', label: t('datasets.classification') },
    { value: 'anomaly', label: t('datasets.anomaly') },
  ];

  return (
    <div className="p-6 max-w-6xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">{t('datasets.title')}</h1>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text" placeholder={t('common.search')}
              className="pl-9 pr-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={filter} onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <button onClick={() => setShowCreate(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">
            <Plus className="w-4 h-4" /> {t('datasets.newDataset')}
          </button>
        </div>
      </div>

      {showCreate && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center" onClick={() => setShowCreate(false)}>
          <div className="bg-white rounded-xl p-6 w-96 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">{t('datasets.createDataset')}</h2>
              <button onClick={() => setShowCreate(false)} className="text-gray-400 hover:text-gray-600"><X className="w-5 h-5" /></button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">{t('datasets.datasetName')}</label>
                <input type="text" value={createName} onChange={(e) => setCreateName(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g. person_detection" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">{t('datasets.task')}</label>
                <select value={createTask} onChange={(e) => setCreateTask(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {taskOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <button onClick={() => setShowCreate(false)}
                className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50">{t('common.cancel')}</button>
              <button onClick={handleCreate} disabled={!createName.trim() || createMutation.isPending}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
                {createMutation.isPending ? t('datasets.creating') : t('common.create')}
              </button>
            </div>
          </div>
        </div>
      )}

      {isLoading ? (
        <LoadingState />
      ) : error ? (
        <ErrorState error={error instanceof Error ? error.message : undefined} onRetry={() => refetch()} />
      ) : filtered.length === 0 ? (
        <EmptyState
          message={datasets.length === 0 ? t('datasets.noDatasets') : t('datasets.noMatching')}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((ds) => (
            <DatasetCard key={ds.name} dataset={ds} onDeleted={handleDeleted} />
          ))}
        </div>
      )}
    </div>
  );
}
