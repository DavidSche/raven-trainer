export interface Dataset {
  name: string;
  display_name: string;
  task: string;
  status: string;
  current_version: string;
  version_count: number;
  total_images: number;
  total_annotations: number;
  health_score: number;
  created_at: string;
  updated_at: string;
  tags: string[];
  owner: string;
}

export interface Version {
  version: string;
  image_count: number;
  annotation_count: number;
  source: string;
  created_at: string;
  parent_version?: string;
}

export interface Experiment {
  id: string;
  project: string;
  model: string;
  backend: string;
  status: string;
  epochs: number;
  batch: number;
  imgsz: number;
  device?: string;
  metrics: Record<string, number>;
  config?: Record<string, unknown>;
  created_at: string;
  finished_at?: string;
  dataset: string;
  workspace: string;
}

export interface TrainingProject {
  name: string;
  display_name: string;
  task: string;
  model: string;
  backend: string;
}

export interface CsvDataPoint {
  headers: string[];
  rows: Record<string, number | string>[];
}

export interface ResultImage {
  name: string;
  path: string;
  size: number;
}

export interface ExperimentResults {
  exists: boolean;
  message?: string;
  run_dir?: string;
  csv?: CsvDataPoint;
  images?: ResultImage[];
  args_yaml?: string;
}

export interface OnnxVerification {
  status: string;
  pt_vs_onnx?: {
    images_compared: number;
    total_detections_compared: number;
    mismatched_image_counts: number;
    max_iou_diff: number;
    max_score_diff: number;
    max_class_diff: number;
    passed: boolean;
  };
  onnx_val?: {
    map50?: number;
    map50_95?: number;
    per_class_map50_95?: Record<string, number>;
  };
}

export interface ExperimentConfig {
  experiment_id: string;
  config: Record<string, unknown>;
  reproducibility: Record<string, unknown>;
}

export interface Job {
  job_id: string;
  type: string;
  status: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  config: Record<string, unknown>;
}

export interface Model {
  id: string;
  project: string;
  version: string;
  type: string;
  format: string;
  path: string;
  metrics: Record<string, number>;
  created_at: string;
  experiment_id: string;
}

export interface AnnotationTask {
  task_id: string;
  platform: string;
  name: string;
  status: string;
  image_count: number;
  label_count: number;
  created_at: string;
  completed_at?: string;
  notes: string;
}

export interface ReportItem {
  id: string;
  name: string;
  type: string;
  path: string;
  created_at: number;
}

export interface SystemStatus {
  version: string;
  data_root: string;
  datasets: number;
  total_images: number;
  total_versions: number;
  models: number;
}

export interface ActivityItem {
  type: string;
  action: string;
  status: string;
  time: string;
}

export interface HealthReport {
  dataset_name: string;
  overall_score: number;
  checks: Record<string, unknown>[];
  issues: string[];
  summary: string;
  recommendations: string[];
}

export interface IngestionRun {
  run_id: string;
  status: string;
  source_count: number;
  total_frames: number;
  kept_frames: number;
  created_at: string;
  pipeline: string;
  report_path: string;
}

export interface ModelDetail {
  model: Model;
  experiment?: Partial<Experiment>;
  file_info: {
    size_bytes: number;
    size_mb: number;
    format: string;
  };
}

export interface MisdataItem {
  filename: string;
  category: string;
  status: string;
  path: string;
  thumbnail?: string;
}

export interface ChangelogEntry {
  timestamp: string;
  action: string;
  detail: string;
  user: string;
  version: string;
  count: number;
  metadata: Record<string, unknown>;
}

export interface FeedbackCategorySummary {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
}

export interface IntelligenceReport {
  dataset_name: string;
  version: string;
  total_images: number;
  total_annotations: number;
  annotated_images: number;
  background_images: number;
  background_ratio: number;
  class_distribution: Record<string, number>;
  camera_distribution: Record<string, number>;
  camera_bias_score: number;
  time_distribution: Record<string, number>;
  object_size_bins: Record<string, number>;
  small_object_ratio: number;
  medium_object_ratio: number;
  large_object_ratio: number;
  bias_warnings: string[];
  recommendations: string[];
}

// ── Dataset Recipe & Builder ──

export interface RecipeSource {
  name: string;
  path?: string;
  include?: string[];
  exclude?: string[];
}

export interface RecipeSplit {
  strategy: "by_video" | "random" | "temporal";
  train: number;
  val: number;
  test: number;
}

export interface RecipeQuality {
  min_blur_score?: number;
  dedup_method?: string;
  hash_distance?: number;
}

export interface RecipeBalance {
  max_per_camera?: number;
  time_balance?: boolean;
}

export interface DatasetRecipe {
  name: string;
  version: string;
  description?: string;
  sources: RecipeSource[];
  include?: string[];
  exclude?: string[];
  sampling: Record<string, any>;
  split: RecipeSplit;
  quality: RecipeQuality;
  balance: RecipeBalance;
  build_target: string;
}

export interface BuildProfile {
  name: string;
  task: string;
  target_format: string;
  include_classes: number[];
  exclude_classes: number[];
  filter: Record<string, any>;
}

export interface WorkspaceBuildResult {
  name: string;
  output_dir: string;
  dataset_yaml: string;
  total_images: number;
  train_count: number;
  val_count: number;
  test_count: number;
  class_count: number;
  class_names: Record<string, string>;
  source_versions: string[];
  skipped_blurry: number;
  skipped_empty: number;
  skipped_duplicate: number;
  duration_sec: number;
  created_at: string;
}

export interface BuildJob {
  job_id: string;
  type: string;
  status: string;
  created_at: string;
  config: Record<string, any>;
}

// ── AI Pre-labeling ──

export interface PreLabelRequest {
  model_path: string;
  conf: number;
  iou: number;
  imgsz: number;
  indices?: number[];
  overwrite: boolean;
}

export interface PreLabelResponse {
  status: string;
  labeled: number;
  skipped: number;
  total_boxes: number;
  model_path: string;
  conf: number;
}

// ── Review Status ──

export type ReviewStatusType = 'pending' | 'approved' | 'rejected';

export interface ReviewStatusItem {
  index: number;
  status: ReviewStatusType;
}

export interface ReviewStatusResult {
  total: number;
  summary: { pending: number; approved: number; rejected: number };
  images: ReviewStatusItem[];
}

export interface ReviewStatusUpdateResult {
  status: string;
  index: number;
  review_status: ReviewStatusType;
  summary: { pending: number; approved: number; rejected: number };
}

export interface BatchReviewStatusResult {
  status: string;
  updated_count: number;
  review_status: ReviewStatusType;
  summary: { pending: number; approved: number; rejected: number };
}

// ── Batch Annotation Operations ──

export type BatchAnnotationOperation = 'delete_by_class' | 'delete_by_area' | 'reassign_class' | 'clear_images';

export interface BatchAnnotationRequest {
  operation: BatchAnnotationOperation;
  indices?: number[];
  class_id?: number;
  min_area?: number;
  max_area?: number;
  new_class_id?: number;
}

export interface BatchAnnotationResponse {
  status: string;
  operation: string;
  images_processed: number;
  boxes_removed: number;
  boxes_modified: number;
  images_cleared: number;
}
