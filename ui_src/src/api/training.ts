import { api } from "./client";
import type { Experiment, TrainingProject, ExperimentResults, ExperimentConfig } from "../types";

export async function fetchExperiments(limit = 50): Promise<Experiment[]> {
  return api.get<Experiment[]>("/training/experiments?limit=" + limit);
}

export async function fetchExperiment(id: string): Promise<Experiment> {
  return api.get<Experiment>("/training/experiments/" + id);
}

export async function fetchExperimentConfig(id: string): Promise<ExperimentConfig> {
  return api.get<ExperimentConfig>("/training/experiments/" + id + "/config");
}

export async function fetchExperimentResults(id: string): Promise<ExperimentResults> {
  return api.get<ExperimentResults>("/training/experiments/" + id + "/results");
}

export async function fetchExperimentResultImage(id: string, name: string): Promise<string> {
  return "/api/v1/training/experiments/" + id + "/results-image?name=" + encodeURIComponent(name);
}

export async function fetchTrainingProjects(): Promise<TrainingProject[]> {
  return api.get<TrainingProject[]>("/training/projects");
}

export async function createTrainingProject(config: { name: string; task?: string; backend?: string; model?: string; dataset_path?: string; dataset_yaml?: string }): Promise<{ status: string; name: string; project_dir: string; project_yaml: string; dataset_yaml: string }> {
  return api.post("/training/projects", config);
}
export async function startTraining(config: Record<string, unknown>): Promise<Record<string, unknown>> {
  return api.post<Record<string, unknown>>("/training/start", config);
}

export async function stopTraining(experimentId: string): Promise<Record<string, unknown>> {
  return api.post<Record<string, unknown>>("/training/stop", { experiment_id: experimentId });
}

export async function fetchTrainingStatus(): Promise<Record<string, unknown>> {
  return api.get<Record<string, unknown>>("/training/status");
}

export async function compareExperiments(ids: string[]): Promise<Record<string, unknown>> {
  return api.post<Record<string, unknown>>("/training/experiments/compare", { experiment_ids: ids });
}

export async function deleteExperiment(experimentId: string): Promise<Record<string, unknown>> {
  return api.del<Record<string, unknown>>("/training/experiments/" + experimentId);
}

export async function fetchExperimentLog(experimentId: string): Promise<{ log: string }> {
  return api.get<{ log: string }>("/training/experiments/" + experimentId + "/log");
}

export async function fetchExperimentReport(experimentId: string): Promise<{ html: string }> {
  return api.get<{ html: string }>("/training/experiments/" + experimentId + "/report");
}

/**
 * Connect to WebSocket for real-time training logs.
 * Returns a cleanup function to close the connection.
 */
export function connectTrainingWebSocket(
  experimentId: string,
  onLog: (message: string) => void,
  onStatusChange?: (status: string) => void,
): () => void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/training/${experimentId}`;
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;
  let attempt = 0;
  const MAX_ATTEMPTS = 10;
  const BASE_DELAY = 1000; // 1s initial, exponential backoff up to ~30s

  function connect() {
    if (closed) return;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      attempt = 0; // reset on successful connect
      if (onStatusChange) onStatusChange("connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "log" && data.message) {
          onLog(data.message);
        }
      } catch {
        // ignore non-JSON messages
      }
    };

    ws.onclose = () => {
      if (onStatusChange) onStatusChange("disconnected");
      if (!closed && attempt < MAX_ATTEMPTS) {
        const delay = Math.min(BASE_DELAY * Math.pow(2, attempt), 30000);
        attempt++;
        reconnectTimer = setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  connect();

  return () => {
    closed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (ws) ws.close();
  };
}

