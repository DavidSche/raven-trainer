import { api } from "./client";
import type { Model } from "../types";

export async function fetchModels(limit = 50): Promise<Model[]> {
  return api.get<Model[]>("/models?limit=" + limit);
}

export async function fetchModelDetail(id: string): Promise<Record<string, unknown>> {
  return api.get<Record<string, unknown>>("/models/" + id);
}

export async function fetchModelMetadata(id: string): Promise<Record<string, unknown>> {
  return api.get<Record<string, unknown>>("/models/" + id + "/metadata");
}

export async function fetchModelStats(): Promise<Record<string, unknown>> {
  return api.get<Record<string, unknown>>("/models/stats");
}
