import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchExperiments, stopTraining, fetchTrainingStatus, deleteExperiment, createTrainingProject } from '../api/training';
import { StatusBadge } from '../components/common/StatusBadge';
import { ExperimentCompare } from '../components/training/ExperimentCompare';
import { TrainingProgress } from '../components/training/TrainingProgress';
import { NewTrainingForm } from '../components/training/NewTrainingForm';
import { ConfirmDialog } from '../components/ui/ConfirmDialog';
import type { Experiment } from '../types';
import { FlaskConical, Square, BarChart3, Trash2, Eye, ShieldCheck } from 'lucide-react';
import { LoadingState, ErrorState, EmptyState } from '../components/common/StateViews';

type Tab = 'list' | 'new' | 'compare';

export default function TrainingPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<Tab>('list');
  const [selectedCompare, setSelectedCompare] = useState<string[]>([]);
  const [viewingLog, setViewingLog] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const [searchParams] = useSearchParams();
  const initialDatasetYaml = searchParams.get('dataset_yaml') || '';
  const initialProject = searchParams.get('project') || '';
  const [autoCreatingProject, setAutoCreatingProject] = useState(false);

  // Experiments list + live training status, both managed by React Query.
  // Poll every 30s while there are running/pending experiments.
  const experimentsQuery = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => fetchExperiments(),
    refetchInterval: (query) =>
      query.state.data?.some((e) => e.status === 'running' || e.status === 'pending')
        ? 30000
        : false,
  });

  const statusQuery = useQuery<Record<string, unknown>>({
    queryKey: ['trainingStatus'],
    queryFn: () => fetchTrainingStatus(),
    refetchInterval: () =>
      experimentsQuery.data?.some((e) => e.status === 'running' || e.status === 'pending')
        ? 30000
        : false,
  });

  const experiments = experimentsQuery.data ?? [];
  const runningStatus = statusQuery.data ?? {};

  // Auto-create training project when coming from workspace.
  const autoCreateMutation = useMutation({
    mutationFn: createTrainingProject,
    onSuccess: () => setActiveTab('new'),
    onError: (e) => {
      console.error('Failed to auto-create project (may already exist):', e);
      setActiveTab('new');
    },
    onSettled: () => setAutoCreatingProject(false),
  });

  useEffect(() => {
    if (initialDatasetYaml && initialProject) {
      setAutoCreatingProject(true);
      autoCreateMutation.mutate({
        name: initialProject,
        dataset_yaml: initialDatasetYaml,
        task: 'detection',
        backend: 'ultralytics',
        model: 'yolo11n',
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (initialDatasetYaml) {
      setActiveTab('new');
    }
  }, [initialDatasetYaml]);

  const stopMutation = useMutation({
    mutationFn: stopTraining,
    onSuccess: () => {
      // Give the backend a moment to update the experiment status, then refresh.
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['experiments'] });
        queryClient.invalidateQueries({ queryKey: ['trainingStatus'] });
      }, 1000);
    },
    onError: (e) => console.error('Failed to stop:', e),
  });

  const handleStopTraining = (experimentId: string) => {
    stopMutation.mutate(experimentId);
  };

  const deleteMutation = useMutation({
    mutationFn: deleteExperiment,
    onSuccess: () => {
      setDeleteConfirm(null);
      queryClient.invalidateQueries({ queryKey: ['experiments'] });
    },
    onError: (e) => console.error('Failed to delete:', e),
  });

  const handleDeleteExperiment = (experimentId: string) => {
    deleteMutation.mutate(experimentId);
  };

  const runningId = Object.keys(runningStatus).find(
    (k) => (runningStatus[k] as Record<string, string>)?.status === 'running'
  );
  const isRunning = !!runningId;

  const getMap50_95 = (exp: Experiment) => {
    const m = exp.metrics || {};
    return m['mAP50-95'] ?? m['metrics/mAP50-95(B)'] ?? m['final_metrics/mAP50-95(B)'] ?? null;
  };

  return (
    <div className="p-6 max-w-7xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">{t('training.title')}</h1>
        <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => { setActiveTab('list'); setViewingLog(null); }}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              activeTab === 'list' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <FlaskConical className="w-4 h-4 inline mr-1" /> {t('training.history')}
          </button>
          <button
            onClick={() => setActiveTab('new')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              activeTab === 'new' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-1" /> {t('training.new')}
          </button>
          <button
            onClick={() => setActiveTab('compare')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              activeTab === 'compare' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-1" /> {t('training.compare')}
          </button>
        </div>
      </div>

      {activeTab === 'new' && (
        <NewTrainingForm datasetYaml={initialDatasetYaml} initialProject={initialProject} onTrainingStarted={(expId) => {
          setViewingLog(expId);
          setActiveTab('list');
          setTimeout(() => {
            queryClient.invalidateQueries({ queryKey: ['experiments'] });
            queryClient.invalidateQueries({ queryKey: ['trainingStatus'] });
          }, 2000);
        }} />
      )}

      {activeTab === 'compare' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <ExperimentCompare
            experiments={experiments}
            selected={selectedCompare}
            onToggle={(id) =>
              setSelectedCompare((prev) =>
                prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
              )
            }
          />
        </div>
      )}

      {activeTab === 'list' && (
        <>
          {viewingLog && (
            <div className="mb-6">
              <TrainingProgress
                experimentId={viewingLog}
                isRunning={isRunning && viewingLog === runningId}
                onClose={() => setViewingLog(null)}
              />
            </div>
          )}

          {experimentsQuery.isLoading ? (
            <LoadingState />
          ) : experimentsQuery.error ? (
            <ErrorState error={experimentsQuery.error instanceof Error ? experimentsQuery.error.message : undefined} onRetry={() => experimentsQuery.refetch()} />
          ) : experiments.length === 0 ? (
            <EmptyState
              icon={<FlaskConical className="w-8 h-8 text-gray-300 mb-3" />}
              message={t('training.noExperiments')}
              hint={`${t('training.newTrainingTab')} → ${t('training.tab')}`}
            />
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left px-4 py-3 font-medium">{t('training.tableExperiment')}</th>
                    <th className="text-left px-4 py-3 font-medium">{t('training.tableProject')}</th>
                    <th className="text-left px-4 py-3 font-medium">{t('training.tableModel')}</th>
                    <th className="text-left px-4 py-3 font-medium">{t('training.tableStatus')}</th>
                    <th className="text-right px-4 py-3 font-medium">{t('training.tableMap50')}</th>
                    <th className="text-right px-4 py-3 font-medium">{t('training.tableMap5095')}</th>
                    <th className="text-right px-4 py-3 font-medium">{t('training.tableEpochs')}</th>
                    <th className="text-center px-4 py-3 font-medium">{t('training.tableOnnx')}</th>
                    <th className="text-right px-4 py-3 font-medium">{t('training.tableActions')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {experiments.map((exp) => {
                    const m = exp.metrics || {};
                    const map50 = m['mAP50'] ?? m['metrics/mAP50(B)'] ?? null;
                    const map50_95 = getMap50_95(exp);
                    return (
                      <tr key={exp.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3">
                          <Link to={'/training/' + exp.id} className="text-blue-600 hover:underline font-medium">
                            {exp.id}
                          </Link>
                        </td>
                        <td className="px-4 py-3 text-gray-600">{exp.project}</td>
                        <td className="px-4 py-3">{exp.model}</td>
                        <td className="px-4 py-3"><StatusBadge status={exp.status} /></td>
                        <td className="px-4 py-3 text-right font-mono text-xs">
                          {map50 !== null ? Number(map50).toFixed(3) : '-'}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs">
                          {map50_95 !== null ? Number(map50_95).toFixed(3) : '-'}
                        </td>
                        <td className="px-4 py-3 text-right">{exp.epochs}</td>
                        <td className="px-4 py-3 text-center">
                          {exp.status === 'completed' && (
                            <span className="text-green-500" title={t('training.onnxExported')}>
                              <ShieldCheck className="w-4 h-4 inline" />
                            </span>
                          )}
                          {exp.status === 'running' && (
                            <span className="text-gray-300 text-xs">{t('training.pending')}</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <div className="flex items-center justify-end gap-1">
                            {exp.status === 'running' && (
                              <button
                                onClick={() => handleStopTraining(exp.id)}
                                className="text-red-600 hover:underline text-xs flex items-center gap-1 px-2 py-1 rounded hover:bg-red-50"
                              >
                                <Square className="w-3 h-3" /> {t('training.stop')}
                              </button>
                            )}
                            {(exp.status !== 'running') && (
                              <>
                                <button
                                  onClick={() => setViewingLog(exp.id === viewingLog ? null : exp.id)}
                                  className={`transition-colors text-xs p-1 rounded ${
                                    viewingLog === exp.id ? 'text-blue-600 bg-blue-50' : 'text-gray-400 hover:text-blue-600 hover:bg-gray-100'
                                  }`}
                                  title={t('training.logFor', { name: exp.id })}
                                >
                                  <Eye className="w-3.5 h-3.5" />
                                </button>
                                <Link to={'/training/' + exp.id} className="px-2 py-1 text-blue-600 hover:bg-blue-50 rounded text-xs">
                                  {t('common.view')}
                                </Link>
                                <button
                                  onClick={() => setDeleteConfirm(exp.id)}
                                  className="p-1 text-red-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                                  title={t('common.delete')}
                                  aria-label={t('common.delete')}
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <ConfirmDialog
        open={!!deleteConfirm}
        title={t('confirm.deleteExperimentTitle')}
        message={t('confirm.deleteExperimentMessage', { name: deleteConfirm })}
        confirmLabel={t('confirm.confirmDeleteExperiment')}
        cancelLabel={t('common.cancel')}
        onConfirm={() => deleteConfirm && handleDeleteExperiment(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
        variant="danger"
      />
    </div>
  );
}
