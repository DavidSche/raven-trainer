import { api } from './client';
import type { MisdataItem } from '../types';

export async function fetchFeedbackDatasets(): Promise<string[]> {
  return api.get<string[]>('/feedback/datasets');
}

export async function fetchFeedback(
  datasetName: string,
  category?: string,
  includeThumbnail = true,
): Promise<{ items: MisdataItem[]; summary: Record<string, number>; total: number }> {
  let url = `/feedback/${datasetName}?include_thumbnail=${includeThumbnail}`;
  if (category) url += `&category=${category}`;
  return api.get(url);
}

export async function fetchFeedbackCategories(datasetName: string): Promise<Record<string, { total: number; pending: number; approved: number; rejected: number }>> {
  return api.get(`/feedback/${datasetName}/categories`);
}

export async function approveFeedback(datasetName: string, category: string, filename: string): Promise<Record<string, unknown>> {
  return api.post(`/feedback/${datasetName}/approve`, { category, filename });
}

export async function rejectFeedback(datasetName: string, category: string, filename: string): Promise<Record<string, unknown>> {
  return api.post(`/feedback/${datasetName}/reject`, { category, filename });
}

export async function mergeFeedback(datasetName: string, version?: string): Promise<Record<string, unknown>> {
  return api.post(`/feedback/${datasetName}/merge`, { version: version || '' });
}
