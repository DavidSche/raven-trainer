import { api } from "./client";

// ── PipelineProfile CRUD ──

export interface PipelineProfileSummary {
  name: string;
  description: string;
  dataset_name: string;
  ingestion_enabled: boolean;
  ingestion_mode: 'profile_ref' | 'inline';
  ingestion_profile_ref: string;
  build_enabled: boolean;
  build_mode: 'recipe_ref' | 'inline';
  build_recipe_ref: string;
  build_recipe_dataset: string;
  training_enabled: boolean;
  training_model: string;
  training_device: string;
  annotation_auto: boolean;
  created_at: string;
  last_run?: { job_id: string; status: string; created_at: string };
}

export interface PipelineProfileDetail {
  name: string;
  version: string;
  description: string;
  created_at: string;
  ingestion: {
    enabled: boolean;
    mode: 'profile_ref' | 'inline';
    profile_ref: string;
    raw_video_path: string;
    pipeline_stages: string;
  };
  dataset: {
    name: string;
    task: string;
    source_dirs: string[];
  };
  annotation: {
    auto_annotate: boolean;
    model: string;
    conf_threshold: number;
  };
  build: {
    enabled: boolean;
    mode: 'recipe_ref' | 'inline';
    recipe_ref: string;
    recipe_dataset: string;
    output_dir: string;
    force: boolean;
  };
  training: {
    enabled: boolean;
    model: string;
    epochs: number;
    batch: number;
    imgsz: number;
    device: string;
    lr: number;
    patience: number;
  };
}

export async function listPipelineProfiles(): Promise<{ profiles: PipelineProfileSummary[] }> {
  return api.get("/pipeline/profiles");
}

export async function getPipelineProfile(name: string): Promise<PipelineProfileDetail> {
  return api.get("/pipeline/profiles/" + encodeURIComponent(name));
}

export async function createPipelineProfile(data: {
  name: string;
  description?: string;
  ingestion?: any;
  dataset?: any;
  annotation?: any;
  build?: any;
  training?: any;
}): Promise<{ status: string; path: string; profile: PipelineProfileDetail }> {
  return api.post("/pipeline/profiles", data);
}

export async function updatePipelineProfile(name: string, data: any): Promise<{ status: string; path: string; profile: PipelineProfileDetail }> {
  return api.put("/pipeline/profiles/" + encodeURIComponent(name), data);
}

export async function deletePipelineProfile(name: string): Promise<{ status: string; name: string }> {
  return api.del("/pipeline/profiles/" + encodeURIComponent(name));
}

export async function runPipelineProfile(name: string): Promise<{ status: string; job_id: string; message: string }> {
  return api.post("/pipeline/profiles/" + encodeURIComponent(name) + "/run");
}

// ── Pipeline job status ──

export async function listPipelineRuns(limit?: number): Promise<{ run_id: string; status: string; dataset: string; profile: string; created_at: string; log_count: number }[]> {
  const params = limit ? "?limit=" + limit : "";
  return api.get("/pipeline/runs" + params);
}

export async function getPipelineRun(runId: string): Promise<{
  run_id: string; status: string; config: any;
  created_at: string; finished_at: string; error: string;
  log: string; result: any;
}> {
  return api.get("/pipeline/runs/" + runId);
}

// ── Legacy utility endpoints ──

export async function listStorageSources(): Promise<{ name: string; path: string; image_count: number }[]> {
  return api.get("/pipeline/storage-sources");
}

export async function runPipelineInline(config: {
  dataset_name: string; task?: string; source_dirs?: string[];
  auto_annotate?: boolean; model_name?: string; val_ratio?: number;
}): Promise<{ status: string; job_id: string }> {
  return api.post("/pipeline/run", config);
}
