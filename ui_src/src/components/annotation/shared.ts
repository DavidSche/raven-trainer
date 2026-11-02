// Shared types, constants, and helpers for the dataset annotation/browse UI.
// Extracted from DatasetBrowsePage so that the page and its sub-components can
// import the same definitions without creating circular dependencies.

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

export interface BrowseResult {
  total: number;
  annotated: number;
  images: BrowseImage[];
}

export interface YoloBox {
  cls_id: number;
  xc: number; // center x (normalized 0-1)
  yc: number; // center y (normalized 0-1)
  w: number;  // width (normalized 0-1)
  h: number;  // height (normalized 0-1)
}

export const API_BASE = '/api/v1';

// CLASS COLORS
export const CLASS_COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
  '#14b8a6', '#e11d48', '#0ea5e9', '#a855f7', '#65a30d',
];

export function getClassColor(clsId: number): string {
  return CLASS_COLORS[clsId % CLASS_COLORS.length];
}
