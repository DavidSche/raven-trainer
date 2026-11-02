import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Activity, Plus, Play, Trash2, Loader2, X, Database,
  Image, ChevronRight, ChevronLeft, Save, GitFork, Cpu, Layers,
  Layout, Download, Copy, Check,
} from 'lucide-react';
import ReactFlow, {
  Background, Controls, ReactFlowProvider, useNodesState, useEdgesState,
  useReactFlow, Handle, Position, getNodesBounds,
  type Edge, type Node, type NodeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';
import type {
  PipelineProfileSummary, PipelineProfileDetail,
} from '../api/pipeline';
import {
  listPipelineProfiles, createPipelineProfile, updatePipelineProfile,
  deletePipelineProfile, runPipelineProfile, getPipelineProfile, getPipelineRun,
} from '../api/pipeline';
import { api } from '../api/client';

// ── Default empty profile ──

function EmptyProfile(): PipelineProfileDetail {
  return {
    name: '', version: '1.0', description: '', created_at: '',
    ingestion: {
      enabled: true, mode: 'profile_ref', profile_ref: '',
      raw_video_path: '', pipeline_stages: 'motion,scene_change,burst,balance',
    },
    dataset: { name: '', task: 'detection', source_dirs: [] },
    annotation: { auto_annotate: false, model: 'yolo26n.pt', conf_threshold: 0.25 },
    build: {
      enabled: true, mode: 'recipe_ref', recipe_ref: '',
      recipe_dataset: '', output_dir: '', force: false,
    },
    training: {
      enabled: false, model: 'yolov8m', epochs: 100, batch: 16,
      imgsz: 640, device: 'auto', lr: 0.001, patience: 50,
    },
  };
}

// ── Stage metadata ──

type StageKey = 'ingestion' | 'dataset' | 'annotation' | 'build' | 'training';

const STAGE_ORDER: StageKey[] = ['ingestion', 'dataset', 'annotation', 'build', 'training'];

const STAGE_COLORS: Record<StageKey, {
  border: string; bg: string; header: string; text: string;
  badge: string; handle: string; dot: string;
}> = {
  ingestion:  { border: 'border-sky-300',     bg: 'bg-sky-50',     header: 'border-sky-200 bg-sky-100',     text: 'text-sky-700',     badge: 'bg-sky-200 text-sky-700',           handle: '!border-sky-400 !bg-sky-400',     dot: 'bg-sky-500' },
  dataset:    { border: 'border-violet-300',  bg: 'bg-violet-50',  header: 'border-violet-200 bg-violet-100', text: 'text-violet-700',  badge: 'bg-violet-200 text-violet-700',     handle: '!border-violet-400 !bg-violet-400', dot: 'bg-violet-500' },
  annotation: { border: 'border-amber-300',   bg: 'bg-amber-50',   header: 'border-amber-200 bg-amber-100',   text: 'text-amber-700',   badge: 'bg-amber-200 text-amber-700',       handle: '!border-amber-400 !bg-amber-400',   dot: 'bg-amber-500' },
  build:      { border: 'border-emerald-300', bg: 'bg-emerald-50', header: 'border-emerald-200 bg-emerald-100', text: 'text-emerald-700', badge: 'bg-emerald-200 text-emerald-700',   handle: '!border-emerald-400 !bg-emerald-400', dot: 'bg-emerald-500' },
  training:   { border: 'border-rose-300',    bg: 'bg-rose-50',    header: 'border-rose-200 bg-rose-100',     text: 'text-rose-700',    badge: 'bg-rose-200 text-rose-700',         handle: '!border-rose-400 !bg-rose-400',     dot: 'bg-rose-500' },
};

// ── Custom Flow Node ──

function PipelineFlowNode({ data }: NodeProps) {
  const { t } = useTranslation();
  const stage = data.stage as StageKey;
  const enabled = data.enabled !== false;
  const c = STAGE_COLORS[stage] || STAGE_COLORS.ingestion;
  const label = t(`pipeline.stages.${stage}`);
  return (
    <div className={`min-w-[180px] rounded-xl border shadow-sm ${enabled ? c.border + ' ' + c.bg : 'border-gray-200 bg-gray-50 opacity-60'}`}>
      <div className={`rounded-t-xl border-b px-3 py-1.5 ${enabled ? c.header : 'border-gray-200 bg-gray-100'}`}>
        <span className={`text-[10px] font-bold uppercase tracking-wider ${enabled ? c.text : 'text-gray-400'}`}>
          {label}
        </span>
      </div>
      <div className="px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-gray-800 truncate">{String(data.summary || label)}</span>
          <span className={`text-[8px] px-1.5 py-0.5 rounded font-bold ${enabled ? c.badge : 'bg-gray-200 text-gray-500'}`}>
            {enabled ? 'ON' : 'OFF'}
          </span>
        </div>
        {data.detail && (
          <div className="mt-0.5 text-[9px] text-gray-500 truncate">{String(data.detail)}</div>
        )}
      </div>
      <Handle type="target" position={Position.Left} className={`!size-2.5 ${enabled ? c.handle : '!border-gray-400 !bg-gray-400'}`} />
      <Handle type="source" position={Position.Right} className={`!size-2.5 ${enabled ? c.handle : '!border-gray-400 !bg-gray-400'}`} />
    </div>
  );
}

const nodeTypes = { pipeline: PipelineFlowNode };

// ── Build nodes / edges from profile ──

const STAGE_POSITIONS: Record<StageKey, { x: number; y: number }> = {
  ingestion:  { x: 80,   y: 200 },
  dataset:    { x: 340,  y: 200 },
  annotation: { x: 600,  y: 200 },
  build:      { x: 860,  y: 200 },
  training:   { x: 1120, y: 200 },
};

function stageEnabled(p: PipelineProfileDetail, stage: StageKey): boolean {
  if (stage === 'ingestion') return p.ingestion.enabled;
  if (stage === 'annotation') return p.annotation.auto_annotate;
  if (stage === 'build') return p.build.enabled;
  if (stage === 'training') return p.training.enabled;
  return true; // dataset always enabled
}

function stageSummary(p: PipelineProfileDetail, stage: StageKey, t: (k: string) => string): { summary: string; detail?: string } {
  if (stage === 'ingestion') {
    if (p.ingestion.mode === 'profile_ref') {
      return { summary: p.ingestion.profile_ref || '—', detail: t('pipeline.refProfile') };
    }
    return { summary: p.ingestion.pipeline_stages || 'inline', detail: p.ingestion.raw_video_path || '' };
  }
  if (stage === 'dataset') {
    return { summary: p.dataset.name || '—', detail: p.dataset.task };
  }
  if (stage === 'annotation') {
    return { summary: p.annotation.model || '—', detail: `conf=${p.annotation.conf_threshold}` };
  }
  if (stage === 'build') {
    if (p.build.mode === 'recipe_ref') {
      return { summary: p.build.recipe_ref || '—', detail: p.build.recipe_dataset || '' };
    }
    return { summary: p.build.recipe_ref || 'inline', detail: p.build.output_dir || '' };
  }
  if (stage === 'training') {
    return { summary: p.training.model || '—', detail: `${p.training.epochs}ep / ${p.training.device}` };
  }
  return { summary: '' };
}

function buildNodes(p: PipelineProfileDetail, t: (k: string) => string): Node[] {
  return STAGE_ORDER.map((stage) => {
    const enabled = stageEnabled(p, stage);
    const meta = stageSummary(p, stage, t);
    return {
      id: stage,
      type: 'pipeline',
      position: STAGE_POSITIONS[stage],
      data: { stage, enabled, summary: meta.summary, detail: meta.detail },
      draggable: false,
    };
  });
}

function buildEdges(p: PipelineProfileDetail): Edge[] {
  const edges: Edge[] = [];
  for (let i = 0; i < STAGE_ORDER.length - 1; i++) {
    const from = STAGE_ORDER[i];
    const to = STAGE_ORDER[i + 1];
    if (stageEnabled(p, from) && stageEnabled(p, to)) {
      edges.push({
        id: `e_${from}_${to}`,
        source: from,
        target: to,
        type: 'smoothstep',
        animated: true,
      });
    }
  }
  return edges;
}

// ── Ingestion Profile selector (loads existing profiles) ──

function IngestionProfileSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { t } = useTranslation();
  const [names, setNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    api.get<{ profiles: { name: string }[] }>('/ingestion/profiles')
      .then((res) => { if (alive) setNames((res.profiles || []).map((x) => x.name)); })
      .catch((e) => { console.error('Failed to load ingestion profiles for selector:', e); if (alive) setNames([]); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);
  if (loading) return <div className="text-[10px] text-gray-400">{t('common.loading')}</div>;
  if (names.length === 0) return <div className="text-[10px] text-gray-400">—</div>;
  return (
    <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
      value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{t('pipeline.selectProfile')}</option>
      {names.map((n) => <option key={n} value={n}>{n}</option>)}
    </select>
  );
}

// ── Dataset selector (loads existing datasets) ──

function DatasetSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { t } = useTranslation();
  const [names, setNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    api.get<DatasetLite[]>('/datasets')
      .then((res) => {
        if (alive) {
          const list = (res || []).map((d) => (typeof d === 'string' ? d : d.name)).filter(Boolean) as string[];
          setNames(list);
        }
      })
      .catch((e) => { console.error('Failed to load datasets for selector:', e); if (alive) setNames([]); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);
  if (loading) return <div className="text-[10px] text-gray-400">{t('common.loading')}</div>;
  if (names.length === 0) return <div className="text-[10px] text-gray-400">—</div>;
  return (
    <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
      value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{t('pipeline.selectDataset')}</option>
      {names.map((n) => <option key={n} value={n}>{n}</option>)}
    </select>
  );
}

interface DatasetLite { name: string; task?: string }

// ── Recipe selector (loads recipes for a given dataset) ──

function RecipeSelect({ dataset, value, onChange }: { dataset: string; value: string; onChange: (v: string) => void }) {
  const { t } = useTranslation();
  const [names, setNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!dataset) { setNames([]); return; }
    let alive = true;
    setLoading(true);
    api.get<string[]>(`/datasets/${encodeURIComponent(dataset)}/recipes`)
      .then((res) => {
        if (alive) {
          const list = Array.isArray(res) ? res : ((res as any)?.recipes ?? []);
          setNames(list.map((r: any) => (typeof r === 'string' ? r : r.name)).filter(Boolean) as string[]);
        }
      })
      .catch((e) => { console.error('Failed to load recipes for selector:', e); if (alive) setNames([]); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [dataset]);
  if (!dataset) return <div className="text-[10px] text-gray-400">{t('pipeline.selectDataset')}</div>;
  if (loading) return <div className="text-[10px] text-gray-400">{t('common.loading')}</div>;
  if (names.length === 0) return <div className="text-[10px] text-gray-400">—</div>;
    return (
    <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
      value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{t('pipeline.selectRecipe')}</option>
      {names.map((n) => <option key={n} value={n}>{n}</option>)}
    </select>
  );
}

// ── Inspector Panel ──

interface InspectorProps {
  profile: PipelineProfileDetail;
  selected: StageKey | null;
  set: (field: string, value: any) => void;
  setDeep: (section: string, key: string, value: any) => void;
}

function InspectorPanel({ profile: p, selected, set, setDeep }: InspectorProps) {
  const { t } = useTranslation();

  if (!selected) {
    return (
      <div className="space-y-4">
        <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{t('pipeline.profileInfo')}</div>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.profileName')}</label>
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.name} onChange={(e) => set('name', e.target.value)} placeholder={t('pipeline.profileName')} />
        </div>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.descriptionLabel')}</label>
          <textarea className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none resize-none" rows={3}
            value={p.description} onChange={(e) => set('description', e.target.value)} />
        </div>
        <div className="pt-2 border-t border-gray-100">
          <div className="text-[10px] text-gray-500 mb-2">{t('pipeline.pipelineStages')}</div>
          <div className="space-y-1">
            {STAGE_ORDER.map((s, i) => {
              const en = stageEnabled(p, s);
              const c = STAGE_COLORS[s];
              return (
                <div key={s} className="flex items-center gap-2 text-[10px] text-gray-700">
                  <span className="size-4 rounded-full text-white flex items-center justify-center text-[8px] font-bold shrink-0" style={{ backgroundColor: 'transparent' }}>
                    <span className={`size-2 rounded-full ${c.dot}`} />
                  </span>
                  <span className="truncate">{t(`pipeline.stages.${s}`)}</span>
                  <span className={`ml-auto text-[8px] px-1.5 py-0.5 rounded font-bold ${en ? c.badge : 'bg-gray-200 text-gray-500'}`}>
                    {en ? 'ON' : 'OFF'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
        <div className="pt-2 border-t border-gray-100 text-[9px] text-gray-400">
          {t('pipeline.pipelineFlow')}
        </div>
      </div>
    );
  }

  // Per-stage inspector
  const card = (color: typeof STAGE_COLORS[StageKey], title: string) => (
    <div className={`rounded-lg border px-3 py-2 ${color.header}`}>
      <div className="text-[10px] font-bold uppercase tracking-wider" style={{ color: undefined }}>
        <span className={color.text}>{title}</span>
      </div>
    </div>
  );

  if (selected === 'ingestion') {
    const c = STAGE_COLORS.ingestion;
    return (
      <div className="space-y-3">
        {card(c, t('pipeline.ingestion'))}
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-gray-700">enabled</span>
          <input type="checkbox" checked={p.ingestion.enabled}
            onChange={(e) => setDeep('ingestion', 'enabled', e.target.checked)}
            className="rounded border-gray-300 text-sky-600" />
        </label>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.ingestionMode')}</label>
          <div className="flex gap-2">
            <button onClick={() => setDeep('ingestion', 'mode', 'profile_ref')}
              className={`flex-1 text-[10px] px-2 py-1.5 rounded border font-medium ${p.ingestion.mode === 'profile_ref' ? 'bg-sky-100 border-sky-300 text-sky-700' : 'bg-white border-gray-200 text-gray-500'}`}>
              {t('pipeline.refProfile')}
            </button>
            <button onClick={() => setDeep('ingestion', 'mode', 'inline')}
              className={`flex-1 text-[10px] px-2 py-1.5 rounded border font-medium ${p.ingestion.mode === 'inline' ? 'bg-sky-100 border-sky-300 text-sky-700' : 'bg-white border-gray-200 text-gray-500'}`}>
              {t('pipeline.inlineConfig')}
            </button>
          </div>
        </div>
        {p.ingestion.mode === 'profile_ref' ? (
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.selectProfile')}</label>
            <IngestionProfileSelect value={p.ingestion.profile_ref}
              onChange={(v) => setDeep('ingestion', 'profile_ref', v)} />
          </div>
        ) : (
          <>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500 font-medium">raw_video_path</label>
              <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
                value={p.ingestion.raw_video_path}
                onChange={(e) => setDeep('ingestion', 'raw_video_path', e.target.value)}
                placeholder="data/storage/videos/..." />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500 font-medium">pipeline_stages</label>
              <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
                value={p.ingestion.pipeline_stages}
                onChange={(e) => setDeep('ingestion', 'pipeline_stages', e.target.value)}
                placeholder="motion,scene_change,burst,balance" />
            </div>
          </>
        )}
      </div>
    );
  }

  if (selected === 'dataset') {
    const c = STAGE_COLORS.dataset;
    return (
      <div className="space-y-3">
        {card(c, t('pipeline.datasetSection'))}
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.datasetName')}</label>
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.dataset.name}
            onChange={(e) => setDeep('dataset', 'name', e.target.value)} />
        </div>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.task')}</label>
          <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.dataset.task}
            onChange={(e) => setDeep('dataset', 'task', e.target.value)}>
            <option value="detection">Detection</option>
            <option value="pose">Pose</option>
            <option value="segmentation">Segmentation</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.sourceDirs')}</label>
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.dataset.source_dirs.join(',')}
            onChange={(e) => setDeep('dataset', 'source_dirs',
              e.target.value ? e.target.value.split(',').map((s) => s.trim()) : [])} />
        </div>
      </div>
    );
  }

  if (selected === 'annotation') {
    const c = STAGE_COLORS.annotation;
    return (
      <div className="space-y-3">
        {card(c, t('pipeline.annotationSection'))}
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-gray-700">{t('pipeline.autoAnnotate')}</span>
          <input type="checkbox" checked={p.annotation.auto_annotate}
            onChange={(e) => setDeep('annotation', 'auto_annotate', e.target.checked)}
            className="rounded border-gray-300 text-amber-600" />
        </label>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.model')}</label>
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.annotation.model}
            onChange={(e) => setDeep('annotation', 'model', e.target.value)} />
        </div>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.confThreshold')}</label>
          <div className="flex items-center gap-2">
            <input type="range" min={0} max={1} step={0.01}
              value={p.annotation.conf_threshold}
              onChange={(e) => setDeep('annotation', 'conf_threshold', parseFloat(e.target.value))}
              className="flex-1 h-1.5 accent-amber-500 bg-gray-200 rounded-full appearance-none cursor-pointer" />
            <input type="number" min={0} max={1} step={0.01}
              value={p.annotation.conf_threshold}
              onChange={(e) => setDeep('annotation', 'conf_threshold', parseFloat(e.target.value))}
              className="w-16 border border-gray-300 rounded px-1.5 py-0.5 text-[10px] text-right outline-none" />
          </div>
        </div>
      </div>
    );
  }

  if (selected === 'build') {
    const c = STAGE_COLORS.build;
    return (
      <div className="space-y-3">
        {card(c, t('pipeline.buildWorkspace'))}
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-gray-700">enabled</span>
          <input type="checkbox" checked={p.build.enabled}
            onChange={(e) => setDeep('build', 'enabled', e.target.checked)}
            className="rounded border-gray-300 text-emerald-600" />
        </label>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.buildMode')}</label>
          <div className="flex gap-2">
            <button onClick={() => setDeep('build', 'mode', 'recipe_ref')}
              className={`flex-1 text-[10px] px-2 py-1.5 rounded border font-medium ${p.build.mode === 'recipe_ref' ? 'bg-emerald-100 border-emerald-300 text-emerald-700' : 'bg-white border-gray-200 text-gray-500'}`}>
              {t('pipeline.refRecipe')}
            </button>
            <button onClick={() => setDeep('build', 'mode', 'inline')}
              className={`flex-1 text-[10px] px-2 py-1.5 rounded border font-medium ${p.build.mode === 'inline' ? 'bg-emerald-100 border-emerald-300 text-emerald-700' : 'bg-white border-gray-200 text-gray-500'}`}>
              {t('pipeline.inlineConfig')}
            </button>
          </div>
        </div>
        {p.build.mode === 'recipe_ref' ? (
          <>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.selectDataset')}</label>
              <DatasetSelect value={p.build.recipe_dataset}
                onChange={(v) => { setDeep('build', 'recipe_dataset', v); setDeep('build', 'recipe_ref', ''); }} />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.selectRecipe')}</label>
              <RecipeSelect dataset={p.build.recipe_dataset} value={p.build.recipe_ref}
                onChange={(v) => setDeep('build', 'recipe_ref', v)} />
            </div>
          </>
        ) : (
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">recipe_ref</label>
            <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.build.recipe_ref}
              onChange={(e) => setDeep('build', 'recipe_ref', e.target.value)} />
          </div>
        )}
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">output_dir</label>
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.build.output_dir}
            onChange={(e) => setDeep('build', 'output_dir', e.target.value)} />
        </div>
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-gray-700">force</span>
          <input type="checkbox" checked={p.build.force}
            onChange={(e) => setDeep('build', 'force', e.target.checked)}
            className="rounded border-gray-300 text-emerald-600" />
        </label>
      </div>
    );
  }

  if (selected === 'training') {
    const c = STAGE_COLORS.training;
    return (
      <div className="space-y-3">
        {card(c, t('pipeline.training'))}
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-gray-700">enabled</span>
          <input type="checkbox" checked={p.training.enabled}
            onChange={(e) => setDeep('training', 'enabled', e.target.checked)}
            className="rounded border-gray-300 text-rose-600" />
        </label>
        <div>
          <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.model')}</label>
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
            value={p.training.model}
            onChange={(e) => setDeep('training', 'model', e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.epochs')}</label>
            <input type="number" className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.training.epochs}
              onChange={(e) => setDeep('training', 'epochs', Number(e.target.value))} />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.batch')}</label>
            <input type="number" className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.training.batch}
              onChange={(e) => setDeep('training', 'batch', Number(e.target.value))} />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.imgsz')}</label>
            <input type="number" className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.training.imgsz}
              onChange={(e) => setDeep('training', 'imgsz', Number(e.target.value))} />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.lr')}</label>
            <input type="number" step={0.0001} className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.training.lr}
              onChange={(e) => setDeep('training', 'lr', Number(e.target.value))} />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.patience')}</label>
            <input type="number" className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.training.patience}
              onChange={(e) => setDeep('training', 'patience', Number(e.target.value))} />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.device')}</label>
            <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
              value={p.training.device}
              onChange={(e) => setDeep('training', 'device', e.target.value)}>
              <option value="auto">{t('pipeline.autoDevice')}</option>
              <option value="cpu">cpu</option>
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

// ── Editor (inner, wrapped by ReactFlowProvider) ──

function PipelineEditor({
  profile, setProfile, onBack, onSave, onSaveAs,
}: {
  profile: PipelineProfileDetail;
  setProfile: (p: PipelineProfileDetail) => void;
  onBack: () => void;
  onSave: () => void;
  onSaveAs: (newName: string) => void;
}) {
  const { t } = useTranslation();
  const { fitView, getViewport, setViewport } = useReactFlow();
  const [selected, setSelected] = useState<StageKey | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    setNodes(buildNodes(profile, t));
    setEdges(buildEdges(profile));
  }, [profile, setNodes, setEdges, t]);

  const set = (field: string, value: any) => setProfile({ ...profile, [field]: value });
  const setDeep = (section: string, key: string, value: any) =>
    setProfile({ ...profile, [section]: { ...(profile as any)[section], [key]: value } });

  const handleSave = async () => {
    setSaving(true);
    try { await onSave(); } finally { setSaving(false); }
  };

  const handleSaveAs = async () => {
    if (!saveAsName.trim()) return;
    setSaving(true);
    try { await onSaveAs(saveAsName.trim()); } finally { setSaving(false); }
    setSaveAsOpen(false);
    setSaveAsName('');
  };

  const handleAutoLayout = useCallback(() => {
    const newNodes = nodes.map(n => {
      const stage = n.id as StageKey;
      if (STAGE_POSITIONS[stage]) return { ...n, position: STAGE_POSITIONS[stage] };
      return n;
    });
    setNodes(newNodes);
    setTimeout(() => fitView({ padding: 0.3, duration: 300 }), 50);
  }, [nodes, setNodes, fitView]);

  const handleExportImage = useCallback(() => {
    const flowEl = document.querySelector('.react-flow') as HTMLElement;
    if (!flowEl) return;
    const nodesBounds = getNodesBounds(nodes);
    if (nodesBounds.width === 0 || nodesBounds.height === 0) return;
    const savedViewport = getViewport();
    fitView({ padding: 0.2, duration: 0 });
    requestAnimationFrame(() => {
      import('html-to-image').then(({ toPng }) => {
        toPng(flowEl, { backgroundColor: '#ffffff', pixelRatio: 2 }).then((dataUrl) => {
          const a = document.createElement('a');
          a.href = dataUrl;
          a.download = `${profile.name || 'pipeline-profile'}.png`;
          a.click();
        }).finally(() => setViewport(savedViewport));
      }).catch((err) => {
        console.error('Export failed:', err);
        setViewport(savedViewport);
      });
    });
  }, [nodes, profile, getViewport, setViewport, fitView]);

  return (
    <div className="h-full flex flex-col">
      {/* Top bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-200 bg-white shrink-0">
        <button onClick={onBack} className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700">
          <ChevronLeft className="size-3" /> {t('common.back')}
        </button>
        <span className="text-xs font-medium text-gray-400">|</span>
        <span className="text-xs font-medium text-gray-700">
          {t('pipeline.editProfile')}: {profile.name || t('pipeline.newProfileTitle')}
        </span>
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-medium">
          {t('ingestionProfiles.editing')}
        </span>
        <div className="ml-auto" />
        {/* Toolbar actions (mirroring IngestionProfiles) */}
        <button onClick={handleAutoLayout}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          title={t('ingestionProfiles.autoLayoutTitle')}>
          <Layout className="size-3.5" />
        </button>
        <button onClick={handleExportImage}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          title={t('ingestionProfiles.exportImageTitle')}>
          <Download className="size-3.5" /> {t('ingestionProfiles.exportImage')}
        </button>
        <button onClick={() => { setSaveAsName(profile.name + '_copy'); setSaveAsOpen(true); }}
          disabled={saving || !profile.name}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors">
          <Copy className="size-3.5" /> {t('ingestionProfiles.cloneLabel')}
        </button>
        <button onClick={handleSave} disabled={saving || !profile.name}
          className="flex items-center gap-1.5 px-4 py-1.5 text-xs font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors">
          {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
          {t('pipeline.saveProfile')}
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 flex min-h-0">
        {/* Left: profile info */}
        <div className="w-[220px] shrink-0 border-r border-gray-200 bg-white overflow-y-auto p-3">
          <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2">{t('pipeline.profileInfo')}</div>
          <div className="space-y-2">
            <div>
              <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.profileName')}</label>
              <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none"
                value={profile.name} onChange={(e) => set('name', e.target.value)} />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('pipeline.descriptionLabel')}</label>
              <textarea className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none resize-none" rows={3}
                value={profile.description} onChange={(e) => set('description', e.target.value)} />
            </div>
          </div>
          <div className="mt-4 pt-2 border-t border-gray-100">
            <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2">{t('pipeline.pipelineStages')}</div>
            <div className="space-y-1">
              {STAGE_ORDER.map((s) => {
                const en = stageEnabled(profile, s);
                const c = STAGE_COLORS[s];
                return (
                  <button key={s} onClick={() => setSelected(s)}
                    className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left text-[10px] transition-colors ${selected === s ? 'bg-gray-100' : 'hover:bg-gray-50'}`}>
                    <span className={`size-2 rounded-full ${c.dot}`} />
                    <span className="truncate text-gray-700">{t(`pipeline.stages.${s}`)}</span>
                    <span className={`ml-auto text-[8px] px-1.5 py-0.5 rounded font-bold ${en ? c.badge : 'bg-gray-200 text-gray-500'}`}>
                      {en ? 'ON' : 'OFF'}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Center: ReactFlow canvas */}
        <div className="flex-1 relative" style={{ height: 'calc(100vh - 150px)' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={(_, n) => setSelected(n.id as StageKey)}
            onPaneClick={() => setSelected(null)}
            nodeTypes={nodeTypes}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={true}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={24} size={1} color="#e5e7eb" />
            <Controls showInteractive={false} />
          </ReactFlow>
          <div className="absolute top-2 left-2 text-[10px] text-gray-500 bg-white/80 px-2 py-1 rounded border border-gray-200">
            {t('pipeline.pipelineFlow')}
          </div>
        </div>

        {/* Right: Inspector */}
        <div className="w-[280px] shrink-0 border-l border-gray-200 bg-white overflow-y-auto p-3">
          <InspectorPanel profile={profile} selected={selected} set={set} setDeep={setDeep} />
        </div>
      </div>

      {/* Save As modal */}
      {saveAsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setSaveAsOpen(false)}>
          <div className="bg-white rounded-xl shadow-xl p-5 w-80" onClick={e => e.stopPropagation()}>
            <h3 className="text-sm font-semibold text-gray-800 mb-3">{t('ingestionProfiles.saveAsTitle')}</h3>
            <input autoFocus className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm outline-none mb-3"
              placeholder={t('ingestionProfiles.newNamePlaceholder')} value={saveAsName}
              onChange={e => setSaveAsName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleSaveAs(); if (e.key === 'Escape') setSaveAsOpen(false); }} />
            <div className="flex items-center gap-2 justify-end">
              <button onClick={() => setSaveAsOpen(false)}
                className="px-3 py-1.5 text-xs text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50">{t('common.cancel')}</button>
              <button onClick={handleSaveAs} disabled={saving || !saveAsName.trim()}
                className="px-3 py-1.5 text-xs text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-1">
                {saving ? <Loader2 className="size-3 animate-spin" /> : <Check className="size-3" />} Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Page ──

export default function PipelinePage() {
  const { t } = useTranslation();

  const [profiles, setProfiles] = useState<PipelineProfileSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [editProfile, setEditProfile] = useState<PipelineProfileDetail | null>(null);
  const [showEditor, setShowEditor] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runJobId, setRunJobId] = useState('');
  const [runLog, setRunLog] = useState<string[]>([]);
  const [runStatus, setRunStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadProfiles = useCallback(async () => {
    try { setLoading(true); const res = await listPipelineProfiles(); setProfiles(res.profiles || []); }
    catch (e) { console.error('Failed to load pipeline profiles:', e); } finally { setLoading(false); }
  }, []);

  useEffect(() => { loadProfiles(); }, [loadProfiles]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, []);

  const handleRun = async (name: string) => {
    setRunning(true); setRunJobId(''); setRunLog([]); setRunStatus('starting');
    try {
      const res = await runPipelineProfile(name);
      setRunJobId(res.job_id);
      pollRef.current = setInterval(async () => {
        try {
          const status = await getPipelineRun(res.job_id);
          const lines = status.log.split('\n').filter(Boolean);
          setRunLog(lines); setRunStatus(status.status);
          if (status.status === 'completed' || status.status === 'failed') {
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
            setRunning(false); loadProfiles();
          }
        } catch (e) {
          console.error('Pipeline poll error:', e);
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          setRunning(false);
        }
      }, 2000);
    } catch (e: any) {
      setRunLog(prev => [...prev, 'Error: ' + e.message]);
      setRunStatus('failed'); setRunning(false);
    }
  };

  const openNewProfile = () => { setEditProfile(EmptyProfile()); setShowEditor(true); };

  const openEditProfile = async (name: string) => {
    try { const detail = await getPipelineProfile(name); setEditProfile(detail); setShowEditor(true); }
    catch (e) { console.error('Failed to open pipeline profile for editing:', e); }
  };

  const handleSaveProfile = async () => {
    if (!editProfile || !editProfile.name) return;
    try {
      const existing = profiles.find((p) => p.name === editProfile.name);
      const body = {
        name: editProfile.name, description: editProfile.description, version: editProfile.version,
        ingestion: editProfile.ingestion, dataset: editProfile.dataset,
        annotation: editProfile.annotation, build: editProfile.build, training: editProfile.training,
      };
      if (existing) await updatePipelineProfile(editProfile.name, body);
      else await createPipelineProfile(body);
      setShowEditor(false); setEditProfile(null); setSaveError(null); await loadProfiles();
    } catch (e: any) { console.error('Failed to save pipeline profile:', e); setSaveError(e.message); }
  };

  const handleSaveAsProfile = async (newName: string) => {
    if (!editProfile || !newName) return;
    try {
      const body = {
        name: newName, description: editProfile.description, version: editProfile.version,
        ingestion: editProfile.ingestion, dataset: editProfile.dataset,
        annotation: editProfile.annotation, build: editProfile.build, training: editProfile.training,
      };
      await createPipelineProfile(body);
      setShowEditor(false); setEditProfile(null); setSaveError(null); await loadProfiles();
    } catch (e: any) { console.error('Failed to duplicate pipeline profile:', e); setSaveError(e.message); }
  };

  const handleDelete = async (name: string) => {
    if (!window.confirm(t('pipeline.deleteConfirm') + ' "' + name + '"?')) return;
    try { await deletePipelineProfile(name); await loadProfiles(); } catch (e) { console.error('Failed to delete pipeline profile:', e); }
  };

  // Editor view
  if (showEditor && editProfile) {
    return (
      <ReactFlowProvider>
        <PipelineEditor
          profile={editProfile}
          setProfile={setEditProfile}
          onBack={() => { setShowEditor(false); setEditProfile(null); }}
          onSave={handleSaveProfile}
          onSaveAs={handleSaveAsProfile}
        />
      </ReactFlowProvider>
    );
  }

  // Log viewer
  if (runJobId && runStatus) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <div className="flex items-center gap-3 mb-4">
          <button onClick={() => { setRunJobId(''); setRunLog([]); setRunStatus(''); }}
            className="p-1.5 hover:bg-gray-100 rounded-lg"><X className="w-5 h-5 text-gray-500" /></button>
          <Activity className="w-5 h-5 text-blue-600" />
          <h1 className="text-lg font-semibold">{t('pipeline.pipelineRun')}: {runJobId}</h1>
          <span className={'text-xs px-2 py-0.5 rounded-full ' + (
            runStatus === 'completed' ? 'bg-green-100 text-green-700' :
            runStatus === 'failed' ? 'bg-red-100 text-red-700' :
            'bg-blue-100 text-blue-700'
          )}>{runStatus}</span>
        </div>
        <div className="bg-gray-900 rounded-lg border border-gray-700">
          <div className="px-4 py-2 bg-gray-800 border-b border-gray-700 rounded-t-lg">
            <span className="text-xs text-gray-400 font-mono">{t('pipeline.pipelineLog')}</span>
          </div>
          <div className="p-4 h-80 overflow-y-auto">
            {runLog.length === 0 ? (
              <div className="text-gray-500 text-xs font-mono">{t('pipeline.waitingOutput')}</div>
            ) : (
              runLog.map((line, i) => (
                <div key={i} className={'text-xs font-mono leading-relaxed ' + (
                  line.includes('FAILED') ? 'text-red-400' :
                  line.includes('Complete') ? 'text-green-400' : 'text-green-400/80'
                )}>{line}</div>
              ))
            )}
            {(runStatus === 'running' || runStatus === 'starting') &&
              <Loader2 className="w-3 h-3 animate-spin text-green-400 mt-2" />}
          </div>
        </div>
        <button onClick={() => { setRunJobId(''); setRunLog([]); setRunStatus(''); }}
          className="mt-4 px-4 py-2 border rounded-lg text-sm hover:bg-gray-50">
          {t('pipeline.backToProfiles')}
        </button>
      </div>
    );
  }

  // Profile list
  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-blue-600" />
          <h1 className="text-xl font-semibold">{t('pipeline.title')}</h1>
        </div>
        <button onClick={openNewProfile}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700">
          <Plus className="w-4 h-4" /> {t('pipeline.newProfile')}
        </button>
      </div>
      <p className="text-sm text-gray-500 mb-6">{t('pipeline.description')}</p>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
        </div>
      ) : profiles.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg border border-dashed border-gray-300">
          <Activity className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500 text-sm">{t('pipeline.noProfiles')}</p>
          <p className="text-gray-400 text-xs mt-1">{t('pipeline.noProfilesHint')}</p>
          <button onClick={openNewProfile}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">
            {t('pipeline.createFirst')}
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {profiles.map((p) => {
            const stages: { label: string; cls: string }[] = [];
            stages.push({ label: t('pipeline.stages.ingestion'), cls: p.ingestion_enabled ? 'bg-sky-50 text-sky-600' : 'bg-gray-50 text-gray-400' });
            stages.push({ label: t('pipeline.stages.dataset'), cls: 'bg-violet-50 text-violet-600' });
            if (p.annotation_auto) stages.push({ label: t('pipeline.stages.annotation'), cls: 'bg-amber-50 text-amber-600' });
            if (p.build_enabled) stages.push({ label: t('pipeline.stages.build'), cls: 'bg-emerald-50 text-emerald-600' });
            if (p.training_enabled) stages.push({ label: t('pipeline.stages.training'), cls: 'bg-rose-50 text-rose-600' });

            return (
              <div key={p.name}
                className="bg-white rounded-lg border border-gray-200 p-4 hover:border-blue-200 transition-colors cursor-pointer"
                onClick={() => openEditProfile(p.name)}>
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <GitFork className="w-4 h-4 text-blue-500" />
                      <h3 className="font-medium text-sm">{p.name}</h3>
                      <ChevronRight className="w-3.5 h-3.5 text-gray-300" />
                    </div>
                    {p.description && (
                      <p className="text-xs text-gray-500 mt-0.5 truncate">{p.description}</p>
                    )}
                    <div className="flex items-center gap-2 mt-2 flex-wrap">
                      <span className="text-[10px] text-gray-400 flex items-center gap-1">
                        <Database className="w-3 h-3" />{p.dataset_name || '—'}
                      </span>
                      {stages.map((s) => (
                        <span key={s.label} className={`text-[10px] px-1.5 py-0.5 rounded ${s.cls}`}>{s.label}</span>
                      ))}
                    </div>
                    <div className="flex items-center gap-3 mt-1.5 text-[10px] text-gray-400">
                      {p.ingestion_enabled && (
                        <span className="flex items-center gap-1">
                          <Image className="w-3 h-3" />
                          {p.ingestion_mode === 'profile_ref' ? `ref: ${p.ingestion_profile_ref || '—'}` : 'inline'}
                        </span>
                      )}
                      {p.build_enabled && (
                        <span className="flex items-center gap-1">
                          <Layers className="w-3 h-3" />
                          {p.build_mode === 'recipe_ref' ? `ref: ${p.build_recipe_ref || '—'}` : 'inline'}
                        </span>
                      )}
                      {p.training_enabled && (
                        <span className="flex items-center gap-1">
                          <Cpu className="w-3 h-3" />
                          {p.training_model} / {p.training_device}
                        </span>
                      )}
                    </div>
                    {p.last_run && (
                      <div className="mt-1.5 text-[10px]">
                        <span className="text-gray-400">{t('pipeline.lastRun')}: </span>
                        <span className={'font-medium ' + (
                          p.last_run.status === 'completed' ? 'text-green-600' :
                          p.last_run.status === 'failed' ? 'text-red-600' : 'text-blue-600'
                        )}>
                          {p.last_run.status} ({p.last_run.job_id})
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 ml-4" onClick={(e) => e.stopPropagation()}>
                    <button onClick={() => handleRun(p.name)} disabled={running}
                      className="p-2 hover:bg-green-50 rounded-lg text-green-600 disabled:opacity-50"
                      title={t('pipeline.runPipeline')}>
                      {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                    </button>
                    <button onClick={() => handleDelete(p.name)}
                      className="p-2 hover:bg-red-50 rounded-lg text-red-500" title={t('pipeline.delete')}>
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
