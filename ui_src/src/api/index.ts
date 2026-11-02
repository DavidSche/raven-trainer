import { api } from './client';
import type { Job, ReportItem, IngestionRun } from '../types';

export async function fetchJobs(limit = 50): Promise<Job[]> {
  return api.get<Job[]>(`/jobs?limit=${limit}`);
}

export async function fetchJobLog(jobId: string): Promise<string> {
  const resp = await api.get<{ log: string }>(`/jobs/${jobId}/log`);
  return resp.log;
}

export async function fetchReports(limit = 50): Promise<ReportItem[]> {
  return api.get<ReportItem[]>(`/reports?limit=${limit}`);
}

export async function fetchIngestionRuns(limit = 50): Promise<IngestionRun[]> {
  return api.get<IngestionRun[]>(`/ingestion/runs?limit=${limit}`);
}

export * from './feedback';
export * from './models';
