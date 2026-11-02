import React, { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { fetchJobs, fetchJobLog } from '../api';
import { StatusBadge } from '../components/common/StatusBadge';
import type { Job } from '../types';
import { CirclePlay, FileText, Filter } from 'lucide-react';
import { LoadingState, ErrorState, EmptyState } from '../components/common/StateViews';

const JOB_TYPES = ['ingest', 'train', 'eval', 'export', 'build', 'pipeline'];
const JOB_STATUSES = ['pending', 'running', 'completed', 'failed', 'cancelled'];

export default function JobsPage() {
  const { t } = useTranslation();
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const [filterType, setFilterType] = useState<string>('');
  const [filterStatus, setFilterStatus] = useState<string>('');

  const { data, isLoading, error, refetch } = useQuery<Job[]>({
    queryKey: ['jobs'],
    queryFn: () => fetchJobs(),
    // Poll every 30s while there are running/pending jobs; stop otherwise.
    refetchInterval: (query) =>
      query.state.data?.some((j) => j.status === 'running' || j.status === 'pending')
        ? 30000
        : false,
  });

  const jobs = data ?? [];

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      if (filterType && job.type !== filterType) return false;
      if (filterStatus && job.status !== filterStatus) return false;
      return true;
    });
  }, [jobs, filterType, filterStatus]);

  const handleShowLog = async (jobId: string) => {
    setSelectedJob(jobId);
    try {
      const logContent = await fetchJobLog(jobId);
      setLog(logContent);
    } catch (err) {
      console.error('Failed to load log for job', jobId, err);
      setLog(t('jobs.failedToLoadLog'));
    }
  };

  return (
    <div className="p-6 max-w-6xl">
      <h1 className="text-xl font-semibold mb-6">{t('jobs.title')}</h1>

      {isLoading ? (
        <LoadingState />
      ) : error ? (
        <ErrorState error={error instanceof Error ? error.message : undefined} onRetry={() => refetch()} />
      ) : jobs.length === 0 ? (
        <EmptyState
          icon={<CirclePlay className="w-8 h-8 text-gray-300 mb-3" />}
          message={t('jobs.noJobs')}
        />
      ) : (
        <>
          <div className="flex items-center gap-3 mb-4">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              aria-label={t('jobs.filterByType')}
            >
              <option value="">{t('jobs.allTypes')}</option>
              {JOB_TYPES.map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
            <select
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              aria-label={t('jobs.filterByStatus')}
            >
              <option value="">{t('jobs.allStatuses')}</option>
              {JOB_STATUSES.map((status) => (
                <option key={status} value={status}>{status}</option>
              ))}
            </select>
            {(filterType || filterStatus) && (
              <button
                onClick={() => { setFilterType(''); setFilterStatus(''); }}
                className="text-xs text-gray-500 hover:text-gray-700 underline"
              >
                {t('common.all')}
              </button>
            )}
          </div>

          {filteredJobs.length === 0 ? (
            <EmptyState
              icon={<Filter className="w-8 h-8 text-gray-300 mb-3" />}
              message={t('jobs.noJobsMatchFilter')}
            />
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left px-4 py-3 font-medium">{t('jobs.tableId')}</th>
                    <th className="text-left px-4 py-3 font-medium">{t('jobs.tableType')}</th>
                    <th className="text-left px-4 py-3 font-medium">{t('jobs.tableStatus')}</th>
                    <th className="text-left px-4 py-3 font-medium">{t('jobs.tableCreated')}</th>
                    <th className="text-right px-4 py-3 font-medium">{t('jobs.tableActions')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {filteredJobs.map((job) => (
                    <tr key={job.job_id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-mono text-xs">{job.job_id}</td>
                      <td className="px-4 py-3">{job.type}</td>
                      <td className="px-4 py-3"><StatusBadge status={job.status} /></td>
                      <td className="px-4 py-3 text-gray-400">{job.created_at.slice(0, 19)}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleShowLog(job.job_id)}
                          className="text-blue-600 hover:underline text-xs flex items-center gap-1 ml-auto"
                        >
                          <FileText className="w-3 h-3" /> {t('jobs.log')}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {selectedJob && (
        <div className="mt-6 bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-white text-sm font-mono">{t('jobs.logFor', { name: selectedJob })}</h3>
            <button onClick={() => setSelectedJob(null)} className="text-gray-400 hover:text-white text-xs">{t('jobs.close')}</button>
          </div>
          <pre className="text-green-400 text-xs font-mono max-h-96 overflow-y-auto">{log || t('jobs.noLogContent')}</pre>
        </div>
      )}
    </div>
  );
}
