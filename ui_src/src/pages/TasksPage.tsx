import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ListTodo, Plus, Trash2, CheckCircle2, Clock,
  Loader2, ChevronDown, ChevronRight, Filter,
} from 'lucide-react';
import { fetchDatasets, fetchDatasetTasks } from '../api/datasets';
import { api } from '../api/client';
import type { Dataset, AnnotationTask } from '../types';
import { LoadingState, ErrorState, EmptyState } from '../components/common/StateViews';

type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'reviewed';

const STATUS_CONFIG: Record<TaskStatus, { color: string; bg: string; icon: React.ReactNode }> = {
  pending: { color: 'text-yellow-600', bg: 'bg-yellow-50', icon: <Clock className="w-3.5 h-3.5" /> },
  in_progress: { color: 'text-blue-600', bg: 'bg-blue-50', icon: <Loader2 className="w-3.5 h-3.5 animate-spin" /> },
  completed: { color: 'text-green-600', bg: 'bg-green-50', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
  reviewed: { color: 'text-purple-600', bg: 'bg-purple-50', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
};

export default function TasksPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<TaskStatus | 'all'>('all');
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newTask, setNewTask] = useState({ name: '', platform: '', notes: '' });
  const [actionError, setActionError] = useState<string | null>(null);

  // ── React Query: datasets list ──
  const datasetsQuery = useQuery<Dataset[]>({
    queryKey: ['datasets'],
    queryFn: fetchDatasets,
  });

  const datasets = datasetsQuery.data ?? [];

  // Auto-select first dataset when list loads
  useEffect(() => {
    if (datasets.length > 0 && !selectedDataset) {
      setSelectedDataset(datasets[0].name);
    }
  }, [datasets, selectedDataset]);

  // ── React Query: tasks for selected dataset ──
  // retry: false — API may not support tasks endpoint yet; treat errors as empty
  const tasksQuery = useQuery<AnnotationTask[]>({
    queryKey: ['tasks', selectedDataset],
    queryFn: () => fetchDatasetTasks(selectedDataset),
    enabled: !!selectedDataset,
    retry: false,
  });

  const tasks = tasksQuery.data ?? [];

  // ── Mutations ──

  const createMutation = useMutation({
    mutationFn: () => api.post(`/datasets/${selectedDataset}/tasks`, {
      name: newTask.name.trim(),
      platform: newTask.platform || 'internal',
      notes: newTask.notes,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks', selectedDataset] });
      setNewTask({ name: '', platform: '', notes: '' });
      setShowCreateForm(false);
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : String(e);
      setActionError(t('tasks.createFailed') + ': ' + msg);
    },
  });

  const updateStatusMutation = useMutation({
    mutationFn: ({ taskId, status }: { taskId: string; status: TaskStatus }) =>
      api.put(`/datasets/${selectedDataset}/tasks/${taskId}`, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks', selectedDataset] });
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : String(e);
      setActionError(t('tasks.updateFailed') + ': ' + msg);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (taskId: string) => api.del(`/datasets/${selectedDataset}/tasks/${taskId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks', selectedDataset] });
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : String(e);
      setActionError(t('tasks.updateFailed') + ': ' + msg);
    },
  });

  // ── Handlers ──

  const handleCreateTask = () => {
    if (!selectedDataset || !newTask.name.trim()) return;
    setActionError(null);
    createMutation.mutate();
  };

  const handleUpdateStatus = (taskId: string, status: TaskStatus) => {
    setActionError(null);
    updateStatusMutation.mutate({ taskId, status });
  };

  const handleDeleteTask = (taskId: string, taskName: string) => {
    if (!window.confirm(t('tasks.confirmDelete', { name: taskName }))) return;
    setActionError(null);
    deleteMutation.mutate(taskId);
  };

  const toggleRow = (taskId: string) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  };

  // ── Derived data ──

  const filteredTasks = statusFilter === 'all'
    ? tasks
    : tasks.filter(task => task.status === statusFilter);

  const statusCounts = tasks.reduce((acc, task) => {
    acc[task.status] = (acc[task.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // ── Early returns for full-page states ──

  if (datasetsQuery.isLoading) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <LoadingState />
      </div>
    );
  }

  if (datasetsQuery.error) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <ErrorState
          error={datasetsQuery.error instanceof Error ? datasetsQuery.error.message : undefined}
          onRetry={() => datasetsQuery.refetch()}
        />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <ListTodo className="w-6 h-6 text-blue-500" />
          <h1 className="text-xl font-semibold">{t('tasks.title')}</h1>
        </div>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          disabled={!selectedDataset}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          <Plus className="w-4 h-4" /> {t('tasks.createTask')}
        </button>
      </div>

      <p className="text-sm text-gray-500 mb-4">{t('tasks.description')}</p>

      {/* Dataset selector */}
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm font-medium text-gray-600 whitespace-nowrap">
          {t('tasks.selectDataset')}:
        </label>
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm min-w-[200px]"
        >
          <option value="">{t('tasks.allDatasets')}</option>
          {datasets.map(ds => (
            <option key={ds.name} value={ds.name}>
              {ds.display_name || ds.name}
            </option>
          ))}
        </select>

        {/* Status filter */}
        <div className="flex items-center gap-1 ml-auto">
          <Filter className="w-4 h-4 text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as TaskStatus | 'all')}
            className="border border-gray-300 rounded-lg px-2 py-1.5 text-xs"
          >
            <option value="all">{t('tasks.allStatuses')}</option>
            <option value="pending">{t('tasks.statusPending')}</option>
            <option value="in_progress">{t('tasks.statusInProgress')}</option>
            <option value="completed">{t('tasks.statusCompleted')}</option>
            <option value="reviewed">{t('tasks.statusReviewed')}</option>
          </select>
        </div>
      </div>

      {/* Status summary chips */}
      {tasks.length > 0 && (
        <div className="flex gap-2 mb-4">
          {(['pending', 'in_progress', 'completed', 'reviewed'] as TaskStatus[]).map(status => {
            const count = statusCounts[status] || 0;
            const config = STATUS_CONFIG[status];
            return (
              <div key={status} className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs ${config.bg} ${config.color}`}>
                {config.icon}
                <span>{t(`tasks.status${status.charAt(0).toUpperCase() + status.slice(1)}`)}</span>
                <span className="font-semibold">{count}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Create form */}
      {showCreateForm && (
        <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-500 block mb-1">{t('tasks.taskName')}</label>
              <input
                type="text"
                value={newTask.name}
                onChange={(e) => setNewTask({ ...newTask, name: e.target.value })}
                placeholder={t('tasks.taskNamePlaceholder')}
                className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">{t('tasks.platform')}</label>
              <input
                type="text"
                value={newTask.platform}
                onChange={(e) => setNewTask({ ...newTask, platform: e.target.value })}
                placeholder={t('tasks.platformPlaceholder')}
                className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
              />
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">{t('tasks.notes')}</label>
            <input
              type="text"
              value={newTask.notes}
              onChange={(e) => setNewTask({ ...newTask, notes: e.target.value })}
              placeholder={t('tasks.notesPlaceholder')}
              className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
            />
          </div>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-3 py-1.5 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50"
            >
              {t('common.cancel')}
            </button>
            <button
              onClick={handleCreateTask}
              disabled={createMutation.isPending || !newTask.name.trim()}
              className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              {createMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
              {t('common.create')}
            </button>
          </div>
        </div>
      )}

      {/* Action error */}
      {actionError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-sm text-red-600">
          {actionError}
        </div>
      )}

      {/* Tasks table */}
      {!selectedDataset ? (
        <EmptyState message={t('tasks.hint')} />
      ) : tasksQuery.isLoading ? (
        <LoadingState />
      ) : filteredTasks.length === 0 ? (
        <EmptyState
          icon={<ListTodo className="w-8 h-8 text-gray-300 mb-3" />}
          message={t('tasks.noTasks')}
          hint={t('tasks.noTasksHint')}
        />
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-8 px-4 py-2"></th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('tasks.taskName')}</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('tasks.platform')}</th>
                <th className="text-right px-4 py-2 font-medium text-gray-500">{t('tasks.imageCount')}</th>
                <th className="text-right px-4 py-2 font-medium text-gray-500">{t('tasks.labelCount')}</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('tasks.status')}</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('tasks.createdAt')}</th>
                <th className="text-right px-4 py-2 font-medium text-gray-500">{t('datasetDetail.actions')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filteredTasks.map(task => {
                const isExpanded = expandedRows.has(task.task_id);
                const status = (task.status as TaskStatus) || 'pending';
                const config = STATUS_CONFIG[status] || STATUS_CONFIG.pending;
                return (
                  <React.Fragment key={task.task_id}>
                    <tr className="hover:bg-gray-50">
                      <td className="px-4 py-2">
                        <button
                          onClick={() => toggleRow(task.task_id)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                        </button>
                      </td>
                      <td className="px-4 py-2 font-medium text-gray-700">{task.name}</td>
                      <td className="px-4 py-2 text-gray-500">{task.platform}</td>
                      <td className="px-4 py-2 text-right text-gray-500">{task.image_count}</td>
                      <td className="px-4 py-2 text-right text-gray-500">{task.label_count}</td>
                      <td className="px-4 py-2">
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${config.bg} ${config.color}`}>
                          {config.icon}
                          {t(`tasks.status${status.charAt(0).toUpperCase() + status.slice(1)}`)}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-xs text-gray-400">
                        {task.created_at ? new Date(task.created_at).toLocaleDateString() : '-'}
                      </td>
                      <td className="px-4 py-2 text-right">
                        <div className="flex items-center justify-end gap-1">
                          <select
                            value={status}
                            onChange={(e) => handleUpdateStatus(task.task_id, e.target.value as TaskStatus)}
                            className="text-xs border border-gray-200 rounded px-1.5 py-0.5"
                            title={t('tasks.updateStatus')}
                          >
                            <option value="pending">{t('tasks.statusPending')}</option>
                            <option value="in_progress">{t('tasks.statusInProgress')}</option>
                            <option value="completed">{t('tasks.statusCompleted')}</option>
                            <option value="reviewed">{t('tasks.statusReviewed')}</option>
                          </select>
                          <button
                            onClick={() => handleDeleteTask(task.task_id, task.name)}
                            className="p-1 text-red-500 hover:bg-red-50 rounded"
                            title={t('tasks.deleteTask')}
                            aria-label={t('tasks.deleteTask')}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr className="bg-gray-50">
                        <td></td>
                        <td colSpan={7} className="px-4 py-3">
                          <div className="space-y-2 text-xs">
                            <div className="grid grid-cols-3 gap-4">
                              <div>
                                <span className="text-gray-400">{t('tasks.taskName')}:</span>
                                <span className="ml-2 text-gray-700">{task.name}</span>
                              </div>
                              <div>
                                <span className="text-gray-400">{t('tasks.platform')}:</span>
                                <span className="ml-2 text-gray-700">{task.platform}</span>
                              </div>
                              <div>
                                <span className="text-gray-400">{t('tasks.createdAt')}:</span>
                                <span className="ml-2 text-gray-700">
                                  {task.created_at ? new Date(task.created_at).toLocaleString() : '-'}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-400">{t('tasks.completedAt')}:</span>
                                <span className="ml-2 text-gray-700">
                                  {task.completed_at ? new Date(task.completed_at).toLocaleString() : '-'}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-400">{t('tasks.imageCount')}:</span>
                                <span className="ml-2 text-gray-700">{task.image_count}</span>
                              </div>
                              <div>
                                <span className="text-gray-400">{t('tasks.labelCount')}:</span>
                                <span className="ml-2 text-gray-700">{task.label_count}</span>
                              </div>
                            </div>
                            {task.notes && (
                              <div>
                                <span className="text-gray-400">{t('tasks.notes')}:</span>
                                <span className="ml-2 text-gray-700">{task.notes}</span>
                              </div>
                            )}
                            <div className="pt-2">
                              <Link
                                to={`/datasets/${selectedDataset}`}
                                className="text-blue-500 hover:text-blue-700 text-xs"
                              >
                                {t('tasks.viewTask')} &rarr;
                              </Link>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
