import { api } from './client';
import type {
  Dataset, Version, IntelligenceReport, HealthReport,
  AnnotationTask, SystemStatus, ActivityItem, MisdataItem,
  DatasetRecipe, ChangelogEntry,
  WorkspaceBuildResult, BuildJob, BuildProfile,
  PreLabelRequest, PreLabelResponse,
  ReviewStatusType, ReviewStatusResult, ReviewStatusUpdateResult, BatchReviewStatusResult,
  BatchAnnotationRequest, BatchAnnotationResponse,
} from '../types';

export async function fetchDatasets(): Promise<Dataset[]> {
  return api.get<Dataset[]>('/datasets');
}

// ── Dataset Recipe API ──

export async function fetchRecipes(name: string): Promise<string[]> {
  return api.get<string[]>(`/datasets/${name}/recipes`);
}

export async function fetchRecipe(name: string, recipeName = "default"): Promise<DatasetRecipe> {
  return api.get<DatasetRecipe>(`/datasets/${name}/recipes/${recipeName}`);
}

export async function saveRecipe(name: string, recipe: DatasetRecipe, recipeName = "default"): Promise<{status: string; path: string}> {
  return api.post<{status: string; path: string}>(`/datasets/${name}/recipes/${recipeName}`, recipe);
}

export async function createDefaultRecipe(name: string): Promise<{status: string; path: string}> {
  return api.post<{status: string; path: string}>(`/datasets/${name}/recipes/default`, {});
}

export async function buildFromRecipe(
  name: string,
  recipeName = "default",
  outputDir?: string,
  force = false,
): Promise<WorkspaceBuildResult> {
  return api.post<WorkspaceBuildResult>(`/datasets/${name}/build-recipe`, {
    recipe_name: recipeName,
    output_dir: outputDir,
    force
});
}

export async function fetchBuildJobs(name?: string): Promise<BuildJob[]> {
  const params = new URLSearchParams({ type: 'build' });
  if (name) params.set('dataset_name', name);
  return api.get<BuildJob[]>('/jobs?' + params.toString());
}

export async function fetchProfiles(): Promise<BuildProfile[]> {
  return api.get<BuildProfile[]>('/profiles');
}

export async function fetchDataset(name: string): Promise<Dataset> {
  return api.get<Dataset>(`/datasets/${name}`);
}

export async function createDataset(name: string, task = 'detection', displayName = ''): Promise<{ status: string; name: string }> {
  return api.post<{ status: string; name: string }>('/datasets', {
    name, task, display_name: displayName
});
}

export async function deleteDataset(name: string): Promise<{ status: string }> {
  return api.del<{ status: string }>(`/datasets/${name}`);
}

export async function fetchVersions(name: string): Promise<Version[]> {
  return api.get<Version[]>(`/datasets/${name}/versions`);
}

export async function createVersion(name: string, source = 'manual', notes = ''): Promise<{ status: string; version: string }> {
  return api.post<{ status: string; version: string }>(`/datasets/${name}/versions`, { source, notes, version: source });
}

export async function buildDataset(name: string, valRatio = 0.2, version = 'v1'): Promise<{
  status: string; workspace_path: string; train_images: number; val_images: number; dataset_yaml: string;
}> {
  return api.post<{
    status: string; workspace_path: string; train_images: number; val_images: number; dataset_yaml: string;
  }>(`/datasets/${name}/build-recipe`, { val_ratio: valRatio, version });
}

export async function fetchIntelligence(name: string, version = 'v1'): Promise<IntelligenceReport> {
  return api.get<IntelligenceReport>(`/datasets/${name}/analyze?version=${version}`);
}

export async function fetchHealth(name: string, version = 'v1'): Promise<HealthReport> {
  return api.get<HealthReport>(`/datasets/${name}/doctor?version=${version}`);
}

export async function runHealthCheck(name: string, version = 'v1'): Promise<HealthReport> {
  return api.post<HealthReport>(`/datasets/${name}/doctor`, { version });
}

export async function fetchTags(name: string): Promise<{ name: string; color: string; description: string }[]> {
  return api.get<{ name: string; color: string; description: string }[]>(`/datasets/${name}/tags`);
}

export async function addTag(name: string, tagName: string, color = '', description = ''): Promise<{ status: string }> {
  return api.post<{ status: string }>(`/datasets/${name}/tags`, { name: tagName, color, description });
}

export async function removeTag(name: string, tagName: string): Promise<{ status: string }> {
  return api.del<{ status: string }>(`/datasets/${name}/tags/${tagName}`);
}

export async function fetchMisdata(name: string, category?: string): Promise<MisdataItem[]> {
  const query = category ? `?category=${category}` : '';
  return api.get<MisdataItem[]>(`/datasets/${name}/misdata${query}`);
}

export async function approveMisdata(name: string, filename: string, category = 'false_positive'): Promise<{ status: string }> {
  return api.post<{ status: string }>(`/datasets/${name}/misdata/approve`, { category, filename });
}

export async function rejectMisdata(name: string, filename: string, category = 'false_positive'): Promise<{ status: string }> {
  return api.post<{ status: string }>(`/datasets/${name}/misdata/reject`, { category, filename });
}

export async function fetchFeedbackStats(name: string): Promise<Record<string, number>> {
  return api.get<Record<string, number>>(`/datasets/${name}/feedback/stats`);
}

export interface ChangelogResult {
  entries: ChangelogEntry[];
  total: number;
}

export async function fetchChangelog(name: string, limit = 100): Promise<ChangelogResult> {
  return api.get<ChangelogResult>(`/datasets/${name}/changelog?limit=${limit}`);
}

export async function fetchStorageSourceRefs(name: string): Promise<{ sources: { source: string; media_type: string; file_count: number }[] }> {
  return api.get(`/storage/sources/for-dataset/${name}`);
}

export async function fetchDatasetTasks(name: string, version = 'v1'): Promise<AnnotationTask[]> {
  return api.get<AnnotationTask[]>(`/datasets/${name}/tasks?version=${version}`);
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  return api.get<SystemStatus>('/system/status');
}

export async function fetchActivities(limit = 10): Promise<ActivityItem[]> {
  return api.get<ActivityItem[]>(`/system/activities?limit=${limit}`);
}

export async function fetchLineage(name: string): Promise<{ nodes: unknown[]; edges: unknown[] }> {
  return api.get<{ nodes: unknown[]; edges: unknown[] }>(`/datasets/${name}/lineage`);
}

export async function fetchBrowseImages(name: string, version: string): Promise<{ total: number; annotated: number; images: BrowseImage[] }> {
  return api.get(`/datasets/${name}/versions/${version}/browse`);
}

export interface BrowseImage {
  index: number;
  image_id: string;
  storage_path: string;
  width: number;
  height: number;
  annotation_count: number;
  camera: string;
  capture_time: string;
  source: string;
}

// ── AI Pre-labeling ──

export async function prelabelImages(
  name: string,
  version: string,
  params: PreLabelRequest,
): Promise<PreLabelResponse> {
  return api.post<PreLabelResponse>(
    `/datasets/${name}/versions/${version}/prelabel`,
    params,
  );
}

// ── Review Status ──

export async function fetchReviewStatus(
  name: string,
  version: string,
): Promise<ReviewStatusResult> {
  return api.get<ReviewStatusResult>(
    `/datasets/${name}/versions/${version}/review-status`,
  );
}

export async function setReviewStatus(
  name: string,
  version: string,
  index: number,
  status: ReviewStatusType,
  reviewer = '',
): Promise<ReviewStatusUpdateResult> {
  return api.put<ReviewStatusUpdateResult>(
    `/datasets/${name}/versions/${version}/review-status/${index}`,
    { status, reviewer },
  );
}

export async function batchSetReviewStatus(
  name: string,
  version: string,
  indices: number[],
  status: ReviewStatusType,
): Promise<BatchReviewStatusResult> {
  return api.put<BatchReviewStatusResult>(
    `/datasets/${name}/versions/${version}/review-status`,
    { indices, status },
  );
}

// ── Batch Annotation Operations ──

export async function batchAnnotationOperation(
  name: string,
  version: string,
  params: BatchAnnotationRequest,
): Promise<BatchAnnotationResponse> {
  return api.post<BatchAnnotationResponse>(
    `/datasets/${name}/versions/${version}/batch-annotations`,
    params,
  );
}




