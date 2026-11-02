import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";

import { useTranslation } from "react-i18next";
import { useNavigate, Link } from "react-router-dom";
import {
  RefreshCw, Plus, Trash2, Play, Loader2, Database, Camera, GitFork, FileText, X, Check, Settings, Copy, Edit3,
  ChevronDown, ChevronRight, Sliders, HardDrive, Image, Film, Search, GripVertical,
  ChevronLeft, PanelLeftClose, PanelLeftOpen, FileVideo, Save, AlertTriangle, Layout, Download,
} from "lucide-react";
import ReactFlow, {
  Background, Controls, ReactFlowProvider, addEdge, useEdgesState, useNodesState,
  useReactFlow, Handle, Position, getNodesBounds, getViewportForBounds,
  type Connection, type Edge, type Node, type NodeProps,
} from "reactflow";
import "reactflow/dist/style.css";
import { api } from "../api/client";

interface ProfileSummary {
  name: string; description: string; source_count: number;
  pipeline: string[]; target_dataset: string; created_at: string;
  sources?: { name: string; path: string; source_type: string }[];
  decode?: { mode: string; every_n_frames: number; target_fps: number; jpeg_quality: number };
}
interface Storage { name: string; type: string; path: string; frames?: number; }

interface StageParam {
  key: string; label: string;
  type: "number" | "boolean" | "select";
  default: unknown; min?: number; max?: number; step?: number;
  options?: { label: string; value: string }[];
  description?: string;
}
interface StageDef {
  name: string; label: string; description: string;
  category: string; params: StageParam[];
}

const STAGES: StageDef[] = [
  { name: "frame_extract", label: "Frame Extraction", description: "Decompress video and extract frames", category: "decode", params: [
    { key: "mode", label: "Mode", type: "select", default: "interval", options: [{label:"Interval",value:"interval"},{label:"By FPS",value:"fps"},{label:"Keyframes",value:"keyframe"}] },
    { key: "every_n_frames", label: "Every N frames", type: "number", default: 30, min: 1, max: 10000, step: 1 },
    { key: "max_frames", label: "Max frames", type: "number", default: 0, min: 0, max: 1000000, step: 1 },
    { key: "quality", label: "JPEG Quality", type: "number", default: 90, min: 1, max: 100, step: 1 },
  ]},
  { name: "scene_change", label: "Scene Change", description: "Detect shot transitions", category: "detect", params: [
    { key: "threshold", label: "Threshold", type: "number", default: 0.3, min: 0.01, max: 1, step: 0.01 },
    { key: "method", label: "Method", type: "select", default: "histogram", options: [{label:"Histogram",value:"histogram"},{label:"Hash",value:"hash"}] },
  ]},
  { name: "motion", label: "Motion Detection", description: "Detect motion regions", category: "detect", params: [
    { key: "sensitivity", label: "Sensitivity", type: "number", default: 0.5, min: 0, max: 1, step: 0.05 },
    { key: "min_area", label: "Min area", type: "number", default: 100, min: 1, max: 100000, step: 1 },
  ]},
  { name: "burst", label: "Burst Capture", description: "Capture bursts", category: "capture", params: [
    { key: "max_burst", label: "Max burst", type: "number", default: 5, min: 1, max: 100, step: 1 },
    { key: "interval_ms", label: " (ms)", type: "number", default: 200, min: 10, max: 10000, step: 10 },
  ]},
  { name: "balance", label: "Quality Filter", description: "Remove blurry/duplicate frames", category: "filter", params: [
    { key: "sharpness", label: "Sharpness", type: "number", default: 0.6, min: 0, max: 1, step: 0.01 },
    { key: "dedup_threshold", label: "Dedup", type: "number", default: 0.15, min: 0, max: 1, step: 0.01 },
  ]},
  { name: "object_filter", label: "Object Filter", description: "Filter by detection", category: "filter", params: [
    { key: "conf", label: "Confidence", type: "number", default: 0.5, min: 0, max: 1, step: 0.01 },
    { key: "min_objects", label: "Min objects", type: "number", default: 1, min: 0, max: 1000, step: 1 },
  ]},
  { name: "normalize", label: "Normalize", description: "Normalize dimensions", category: "transform", params: [
    { key: "width", label: "Width", type: "number", default: 1920, min: 64, max: 8192, step: 1 },
    { key: "height", label: "Height", type: "number", default: 1080, min: 64, max: 8192, step: 1 },
    { key: "keep_aspect", label: "Keep aspect ratio", type: "boolean", default: true },
  ]},
  { name: "dedup", label: "Deduplication", description: "Remove near-duplicates", category: "filter", params: [
    { key: "method", label: "Algorithm", type: "select", default: "phash", options: [{label:"Perceptual Hash",value:"phash"},{label:"dHash",value:"dhash"},{label:"Histogram",value:"histogram"}] },
    { key: "threshold", label: "Similarity", type: "number", default: 0.9, min: 0, max: 1, step: 0.01 },
  ]},
];

const STAGE_MAP = new Map(STAGES.map(s => [s.name, s]));

function StageFlowNode({ data }: NodeProps) {
  const { t } = useTranslation();
  const def = STAGE_MAP.get(data.stage as string);
  const on = data.enabled !== false;
  const cat = def?.category as string || "stage";
  const colors: Record<string, {border:string,bg:string,header:string,text:string,badge:string,handle:string}> = {
    decode:    {border:"border-sky-300", bg:"bg-sky-50", header:"border-sky-200 bg-sky-100", text:"text-sky-700", badge:"bg-sky-200 text-sky-700", handle:"!border-sky-400 !bg-sky-400"},
    detect:    {border:"border-amber-300", bg:"bg-amber-50", header:"border-amber-200 bg-amber-100", text:"text-amber-700", badge:"bg-amber-200 text-amber-700", handle:"!border-amber-400 !bg-amber-400"},
    capture:   {border:"border-rose-300", bg:"bg-rose-50", header:"border-rose-200 bg-rose-100", text:"text-rose-700", badge:"bg-rose-200 text-rose-700", handle:"!border-rose-400 !bg-rose-400"},
    filter:    {border:"border-purple-300", bg:"bg-purple-50", header:"border-purple-200 bg-purple-100", text:"text-purple-700", badge:"bg-purple-200 text-purple-700", handle:"!border-purple-400 !bg-purple-400"},
    transform: {border:"border-teal-300", bg:"bg-teal-50", header:"border-teal-200 bg-teal-100", text:"text-teal-700", badge:"bg-teal-200 text-teal-700", handle:"!border-teal-400 !bg-teal-400"},
  };
  const c = colors[cat] || colors.decode;
  return (
    <div className={`min-w-[160px] rounded-xl border shadow-sm ${on ? c.border+" "+c.bg : "border-gray-200 bg-gray-50 opacity-60"}`}>
      <div className={`rounded-t-xl border-b px-3 py-1.5 ${on ? c.header : "border-gray-200 bg-gray-100"}`}>
        <span className={`text-[10px] font-bold uppercase tracking-wider ${on ? c.text : "text-gray-400"}`}>
          {cat}
        </span>
      </div>
      <div className="px-3 py-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-gray-800 truncate">{def?.label || String(data.stage)}</span>
          <span className={`text-[8px] px-1.5 py-0.5 rounded font-bold ${on ? c.badge : "bg-gray-200 text-gray-500"}`}>
            {on ? "ON" : ""}
          </span>
        </div>
        {def && <div className="mt-0.5 text-[8px] text-gray-400">{def.params.length} params</div>}
      </div>
      <Handle type="target" position={Position.Left} className={`!size-2.5 ${on ? c.handle : "!border-gray-400 !bg-gray-400"}`} />
      <Handle type="source" position={Position.Right} className={`!size-2.5 ${on ? c.handle : "!border-gray-400 !bg-gray-400"}`} />
    </div>
  );
}

function FlowNode({ data }: NodeProps) {
  const { t } = useTranslation();
  return (
    <div className="min-w-[160px] rounded-xl border border-violet-300 bg-violet-50 shadow-sm">
      <div className="rounded-t-xl border-b border-violet-200 bg-violet-100 px-3 py-1.5">
        <span className="text-[10px] font-bold text-violet-600 uppercase tracking-wider">{t('ingestionProfiles.sourceHeader')}</span>
      </div>
      <div className="px-3 py-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-800">
          <Database className="size-3.5 text-violet-500" />
          <span className="truncate">{String(data.label)}</span>
        </div>
        <div className="mt-0.5 text-[9px] text-gray-500">{data.mode === "new" ? t('ingestionProfiles.createNew') : t('ingestionProfiles.appendToExisting')}</div>
      </div>
      <Handle type="target" position={Position.Left} className="!border-violet-400 !bg-violet-400 !size-2.5" />
    </div>
  );
}

function Node({ data }: NodeProps) {
  const { t } = useTranslation();
  const isVideo = data.source === "video";
  return (
    <div className="min-w-[160px] rounded-xl border border-sky-300 bg-sky-50 shadow-sm">
      <div className="rounded-t-xl border-b border-sky-200 bg-sky-100 px-3 py-1.5">
        <span className="text-[10px] font-bold text-sky-600 uppercase tracking-wider">{isVideo ? t('ingestionProfiles.videoLabel') : t('ingestionProfiles.imageLabel')}</span>
      </div>
      <div className="px-3 py-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-800">
          {isVideo ? <Film className="size-3.5 text-sky-500" /> : <Image className="size-3.5 text-emerald-500" />}
          <span className="truncate">{String(data.label)}</span>
        </div>
        <div className="mt-0.5 text-[9px] text-gray-500 font-mono truncate">{String(data.path)}</div>
        {data.frames != null && <div className="mt-1 text-[9px] text-gray-400">{Number(data.frames)} frames</div>}
      </div>
      <Handle type="source" position={Position.Right} className="!border-sky-400 !bg-sky-400 !size-2.5" />
    </div>
  );
}


// Register custom node types for ReactFlow
const CUSTOM_NODE_TYPES = {
  source: Node as any,
  stage: StageFlowNode as any,
  dataset: FlowNode as any,
};

// ── Node Inspector ──

function NodeInspector({ node, onUpdate }: { node: Node | null; onUpdate: (id: string, data: Record<string, unknown>) => void }) {
  const { t } = useTranslation();
  if (!node) return null;
  const nt = node.type;
  if (nt === "source") return (
    <div className="space-y-3">
      <div className="bg-sky-50 border border-sky-200 rounded-lg px-3 py-2">
        <div className="text-[10px] font-bold text-sky-600 uppercase">{t('ingestionProfiles.sourceHeader')}</div>
        <div className="mt-1 text-sm font-semibold text-gray-800">{String(node.data.label)}</div>
        <div className="text-[10px] text-gray-500 font-mono mt-0.5">{String(node.data.path)}</div>
      </div>
      <div className="text-[10px] text-gray-600 space-y-1">
        <div className="flex justify-between"><span>{t('ingestionProfiles.source')}</span><span className="text-gray-800">{String(node.data.source)}</span></div>
        {node.data.frames != null && <div className="flex justify-between"><span>{t('ingestionProfiles.frames')}</span><span className="text-gray-800">{(node.data.frames as number).toLocaleString()}</span></div>}
      </div>
    </div>
  );
  if (nt === "dataset") return (
    <div className="space-y-3">
      <div className="bg-violet-50 border border-violet-200 rounded-lg px-3 py-2">
        <div className="text-[10px] font-bold text-violet-600 uppercase">{t('ingestionProfiles.datasetHeader')}</div>
        <div className="mt-1 text-sm font-semibold text-gray-800">{String(node.data.label)}</div>
      </div>
      <div>
        <label className="text-[10px] text-gray-500 font-medium block mb-1">{t('ingestionProfiles.modeLabel')}</label>
        <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none" value={String(node.data.mode)}
          onChange={e => onUpdate(node.id,{...node.data,mode:e.target.value})}>
          <option value="new">{t('ingestionProfiles.createNew')}</option>
          <option value="existing">{t('ingestionProfiles.appendToExisting')}</option>
        </select>
      </div>
      <div>
        <label className="text-[10px] text-gray-500 font-medium block mb-1">{t('ingestionProfiles.dataset')}</label>
        {node.data.mode === "existing" ? (
          <Selector value={String(node.data.dataset||"")} onChange={(val) => onUpdate(node.id,{...node.data,dataset:val})} />
        ) : (
          <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none" value={String(node.data.dataset||"")}
            onChange={e => onUpdate(node.id,{...node.data,dataset:e.target.value})} />
        )}
      </div>
    </div>
  );
  if (nt === "stage") {
    const def = STAGE_MAP.get(node.data.stage as string);
    if (!def) return <div className="text-xs text-gray-500">{t('ingestionProfiles.unknownStage')}</div>;
    const params = (node.data.params as Record<string, unknown>) ?? {};
    const updP = (key: string, val: unknown) => onUpdate(node.id, { ...node.data, params: { ...params, [key]: val } });
    return (
      <div className="space-y-3">
        <div className={`rounded-lg border px-3 py-2 ${node.data.enabled !== false ? "bg-emerald-50 border-emerald-200" : "bg-gray-50 border-gray-200"}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[10px] font-bold text-emerald-600 uppercase">{def.category}</div>
              <div className="text-xs font-semibold text-gray-800 mt-0.5">{def.label}</div>
            </div>
            <button onClick={() => onUpdate(node.id,{...node.data,enabled:node.data.enabled===false})}
              className={`text-[8px] px-2 py-1 rounded font-bold border ${node.data.enabled!==false ? "bg-emerald-200 border-emerald-300 text-emerald-700" : "bg-gray-200 border-gray-300 text-gray-500"}`}>
              {node.data.enabled !== false ? "ON" : t('ingestionProfiles.off')}
            </button>
          </div>
        </div>
        {def.params.map(p => {
          const val = params[p.key] ?? p.default;
          return (
            <div key={p.key}>
              <label className="text-[10px] text-gray-500 font-medium block mb-1">{p.label}</label>
              {p.type === "number" ? (
                <div className="flex items-center gap-2">
                  <input type="range" min={p.min??0} max={p.max??100} step={p.step??0.01}
                    value={typeof val==="number"?val:Number(p.default)}
                    onChange={e => updP(p.key, parseFloat(e.target.value))}
                    className="flex-1 h-1.5 accent-emerald-500 bg-gray-200 rounded-full appearance-none cursor-pointer" />
                  <input type="number" min={p.min} max={p.max} step={p.step}
                    value={typeof val==="number"?val:Number(p.default)}
                    onChange={e => updP(p.key, parseFloat(e.target.value))}
                    className="w-14 border border-gray-300 rounded px-1.5 py-0.5 text-[10px] text-right outline-none" />
                </div>
              ) : p.type === "boolean" ? (
                <button onClick={() => updP(p.key, !val)}
                  className={`text-[10px] px-2.5 py-1.5 rounded border font-medium ${val ? "bg-blue-100 border-blue-300 text-blue-700" : "bg-gray-100 border-gray-300 text-gray-500"}`}>
                  {val ? "ON" : t('ingestionProfiles.off')}
                </button>
              ) : (
                <select className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs outline-none" value={String(val)}
                  onChange={e => updP(p.key, e.target.value)}>
                  {(p.options??[]).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              )}
              {p.description && <p className="mt-0.5 text-[9px] text-gray-400">{p.description}</p>}
            </div>
          );
        })}
      </div>
    );
  }
  return null;
}

// ── Stage Palette ──

function StagePalette({ onAdd }: { onAdd: (name: string) => void }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const cats = useMemo(() => {
    const m = new Map<string, StageDef[]>();
    STAGES.forEach(s => { const a = m.get(s.category) ?? []; a.push(s); m.set(s.category, a); });
    return [...m.entries()];
  }, []);
  const filtered = search.trim() ? STAGES.filter(s => s.name.includes(search.toLowerCase()) || s.label.toLowerCase().includes(search.toLowerCase())) : null;
  return (
    <div className="p-3">
      <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2">{t('ingestionProfiles.stagesHeader')}</div>
      <div className="relative mb-2">
        <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3 text-gray-400" />
        <input className="w-full border border-gray-300 rounded-lg py-1.5 pl-7 pr-2 text-xs outline-none" placeholder={t('ingestionProfiles.searchStages')}
          value={search} onChange={e => setSearch(e.target.value)} />
      </div>
      {filtered ? filtered.map(s => (
        <button key={s.name} onClick={() => { onAdd(s.name); setSearch(""); }}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-gray-100 text-xs text-gray-700">
          <Plus className="size-3 text-gray-400 shrink-0" /> {s.label}
        </button>
      )) : cats.map(([cat, stages]) => (
        <div key={cat}>
          <button onClick={() => setExpanded(expanded===cat?null:cat)}
            className="w-full flex items-center justify-between px-2 py-1 text-[9px] font-bold text-gray-500 uppercase tracking-wider hover:text-gray-700">
            {cat} <ChevronRight className={`size-3 transition-transform ${expanded===cat?"rotate-90":""}`} />
          </button>
          {expanded===cat && stages.map(s => (
            <button key={s.name} onClick={() => onAdd(s.name)}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-gray-100 text-xs text-gray-700 ml-2">
              <Plus className="size-2.5 text-gray-400 shrink-0" /> {s.label}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

// ── Flow or ──

const nidGen = () => "n_" + crypto.randomUUID();

/** Build edges in star topology: all sources feed into the first non-source node,
 *  stages chain sequentially, and the last active node connects to the dataset. */
function computeEdges(nodes: Node[]): Edge[] {
  const sources = nodes.filter(n => n.type === "source");
  const stages = nodes.filter(n => n.type === "stage" && (n.data as any).enabled !== false);
  const datasets = nodes.filter(n => n.type === "dataset");
  const result: Edge[] = [];
  // All sources connect to the first enabled stage (or first dataset if no stages)
  const firstTarget = stages.length > 0 ? stages[0] : (datasets.length > 0 ? datasets[0] : null);
  if (firstTarget) {
    for (const src of sources) {
      result.push({ id: `e_${src.id}_${firstTarget.id}`, source: src.id, target: firstTarget.id, type: "smoothstep", animated: true });
    }
  }
  // Chain stages sequentially
  for (let i = 0; i < stages.length - 1; i++) {
    result.push({ id: `e_${stages[i].id}_${stages[i+1].id}`, source: stages[i].id, target: stages[i+1].id, type: "smoothstep", animated: true });
  }
  // Last active node connects to the dataset
  const lastActive = stages.length > 0 ? stages[stages.length - 1] : (sources.length > 0 ? sources[sources.length - 1] : null);
  if (lastActive && datasets.length > 0 && lastActive.id !== datasets[0].id) {
    result.push({ id: `e_${lastActive.id}_${datasets[0].id}`, source: lastActive.id, target: datasets[0].id, type: "smoothstep", animated: true });
  }
  return result;
}

function Flowor({ sources: storages, editProfile, onSaved }: { sources: Storage[]; editProfile?: ProfileSummary | null; onSaved?: () => void }) {
  const { t } = useTranslation();
  const { screenToFlowPosition, fitView, getViewport, setViewport } = useReactFlow();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [profile, setProfile] = useState("");
  const [profileDesc, setProfileDesc] = useState("");
  const [sideTab, setSideTab] = useState<"sources"|"stages"|"dataset">("sources");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [saveAs, setSaveAs] = useState("");
  const selectedNode = useMemo(() => nodes.find(n => n.id === selectedId) ?? null, [nodes, selectedId]);

  const reconnect = useCallback(() => {
    if (nodes.length > 0) {
      setEdges(computeEdges(nodes));
    }
  }, [nodes]);

  const add = useCallback((src: Storage) => {
    const newId = nidGen();
    const srcCount = nodes.filter(n => n.type === "source").length;
    const newNode = { id: newId, type: "source" as const, position: { x: 60, y: 140 + srcCount * 120 }, data: { label: src.name, path: src.path, source: src.type, frames: src.frames } };
    const updatedNodes: any[] = [...nodes, newNode];
    setNodes([...nodes, newNode]);
    setEdges(computeEdges(updatedNodes));
  }, [nodes, setNodes, setEdges]);

  const addStage = useCallback((name: string) => {
    const def = STAGE_MAP.get(name);
    if (!def) return;
    const params: Record<string, unknown> = {};
    def.params.forEach(p => { params[p.key] = p.default; });
    const newId = nidGen();
    const stgCount = nodes.filter(n => n.type === "stage").length;
    const newNode = { id: newId, type: "stage" as const, position: { x: 280 + stgCount * 220, y: 200 + (stgCount % 2) * 60 }, data: { stage: name, enabled: true, params } };
    const updatedNodes: any[] = [...nodes, newNode];
    setNodes([...nodes, newNode]);
    setEdges(computeEdges(updatedNodes));
    setTimeout(() => fitView({ padding: 0.3, duration: 300 }), 50);
  }, [nodes, setNodes, setEdges, fitView]);

  const addDatasetNode = useCallback(() => {
    const newId = nidGen();
    const stgCount = nodes.filter(n => n.type === "stage").length;
    const dsCount = nodes.filter(n => n.type === "dataset").length;
    const newNode = { id: newId, type: "dataset" as const, position: { x: 280 + Math.max(stgCount, 1) * 220 + dsCount * 180, y: 280 }, data: { label: "Output ", mode: "new", dataset: profile || "dataset" } };
    const updatedNodes: any[] = [...nodes, newNode];
    setNodes([...nodes, newNode]);
    setEdges(computeEdges(updatedNodes));
  }, [nodes, setNodes, setEdges, profile]);

  const updateNode = useCallback((id: string, data: Record<string, unknown>) => {
    setNodes(nds => nds.map(n => n.id === id ? { ...n, data } : n));
  }, [setNodes]);

    const onConnect = useCallback((p: Connection | Edge) => {
    setEdges(eds => addEdge({ ...p, type: "smoothstep", animated: true }, eds));
  }, [setEdges]);
  const onDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect="move"; }, []);

  const handleAutoLayout = useCallback(() => {
    const srcNodes = nodes.filter(n => n.type === "source");
    const stgNodes = nodes.filter(n => n.type === "stage");
    const dsNodes = nodes.filter(n => n.type === "dataset");
    const newNodes = nodes.map(n => {
      if (n.type === "source") {
        const idx = srcNodes.indexOf(n);
        return { ...n, position: { x: 60, y: 100 + idx * 110 } };
      }
      if (n.type === "stage") {
        const idx = stgNodes.indexOf(n);
        return { ...n, position: { x: 320, y: 100 + idx * 110 } };
      }
      if (n.type === "dataset") {
        const idx = dsNodes.indexOf(n);
        return { ...n, position: { x: 580 + idx * 220, y: 150 } };
      }
      return n;
    });
    setNodes(newNodes);
    setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 50);
  }, [nodes, setNodes, fitView]);

  const handleExportImage = useCallback(() => {
    const flowEl = document.querySelector(".react-flow") as HTMLElement;
    if (!flowEl) return;
    const nodesBounds = getNodesBounds(nodes);
    if (nodesBounds.width === 0 || nodesBounds.height === 0) return;
    // Save current viewport, then fit all nodes so edges render in-frame
    const savedViewport = getViewport();
    fitView({ padding: 0.2, duration: 0 });
    // Wait a frame for fitView to propagate to DOM before capturing
    requestAnimationFrame(() => {
      import("html-to-image").then(({ toPng }) => {
        toPng(flowEl, { backgroundColor: "#ffffff", pixelRatio: 2 }).then((dataUrl) => {
          const a = document.createElement("a");
          a.href = dataUrl;
          a.download = `${profile || "ingestion-profile"}.png`;
          a.click();
        }).finally(() => setViewport(savedViewport));
      }).catch((err) => {
        console.error("Export failed:", err);
        setViewport(savedViewport);
      });
    });
  }, [nodes, profile, getViewport, setViewport, fitView]);

  useEffect(() => {
    if (editProfile && nodes.length === 0) {
      setProfileDesc(editProfile.description);
      const pendingNodes: any[] = [];
      const curSrc = storages;
      (editProfile.sources || []).forEach((src: any, si: number) => {
        const full = curSrc.find((s: any) => s.name === src.name);
        pendingNodes.push({ id: nidGen(), type: "source", position: { x: 60, y: 140 + si * 120 }, data: { label: src.name, path: src.path, source: src.source_type || src.source, frames: full?.frames } });
      });
      // Create frame_extract stage from decode config
      const decode = (editProfile as any).decode || {};
      const feDef = STAGE_MAP.get("frame_extract");
      if (feDef) {
        const params: Record<string, any> = {};
        feDef.params.forEach((p: any) => { params[p.key] = decode[p.key] ?? p.default; });
        pendingNodes.push({ id: nidGen(), type: "stage", position: { x: 280, y: 160 }, data: { stage: "frame_extract", enabled: true, params } });
      }
      (editProfile.pipeline || []).forEach((stg: string, si: number) => {
        const def = STAGE_MAP.get(stg);
        if (!def) return;
        const params: Record<string, unknown> = {};
        def.params.forEach((p: any) => { params[p.key] = p.default; });
        // +1 offset because frame_extract already occupies slot 0
        pendingNodes.push({ id: nidGen(), type: "stage", position: { x: 280 + (si + 1) * 220, y: 200 + (si % 2) * 60 }, data: { stage: stg, enabled: true, params } });
      });
      const stgCount = (editProfile.pipeline || []).length;
        // +1 for frame_extract + pipeline stages
        pendingNodes.push({ id: nidGen(), type: "dataset", position: { x: 280 + Math.max(stgCount + 1, 2) * 220, y: 280 }, data: { label: "Output ", mode: "new", dataset: editProfile.target_dataset || editProfile.name || "dataset" } });
        // Compute edges synchronously from pendingNodes (no stale closure issue)
        setNodes(pendingNodes);
        setEdges(computeEdges(pendingNodes));
        setProfile(editProfile.name);
        setTimeout(() => fitView({ padding: 0.3, duration: 300 }), 100);
      }
  }, [editProfile]);

  useEffect(() => { if (profile && !nodes.some(n => n.type === "dataset")) addDatasetNode(); }, [profile]);

  const buildProfileBody = (name: string) => {
    const sNodes = nodes.filter(n => n.type === "source");
    const stNodes = nodes.filter(n => n.type === "stage");
    const dNodes = nodes.filter(n => n.type === "dataset");
    const ds = (dNodes[0]?.data as any)?.dataset as string ?? name;
    // Extract decode config from frame_extract stage (if present)
    const feNode = stNodes.find(n => (n.data as any).stage === "frame_extract");
    const feParams = (feNode?.data as any)?.params as Record<string, any> ?? {};
    const decode = feNode ? {
      mode: feParams.mode || "interval",
      every_n_frames: feParams.every_n_frames ?? 30,
      target_fps: feParams.target_fps ?? 0.0,
      jpeg_quality: feParams.jpeg_quality ?? 80,
    } : {
      mode: "interval", every_n_frames: 30, target_fps: 0.0, jpeg_quality: 80,
    };
    // Exclude frame_extract from pipeline (it's not a sampler, it's decode config)
    const pipeline = stNodes.filter(n => (n.data as any).stage !== "frame_extract").map(n => (n.data as any).stage as string);
    return { name, description: profileDesc, decode, sources: sNodes.map(n => ({ name: (n.data as any).label, path: (n.data as any).path, source_type: (n.data as any).source })), pipeline, sample_count: 500, target_dataset: ds };
  };

  const validateProfile = () => {
    const srcCount = nodes.filter(n => n.type === "source").length;
    const stgCount = nodes.filter(n => n.type === "stage").length;
    const dsCount = nodes.filter(n => n.type === "dataset").length;
    const errors: string[] = [];
    if (srcCount === 0) errors.push("At least one source is required");
    if (dsCount === 0) errors.push("A dataset output node is required");
    if (dsCount > 1) errors.push("Only one dataset output is supported (found " + dsCount + ")");
    // Order validation: frame_extract should come first, dedup should come last among stages
    if (stgCount > 0) {
      const stages = nodes.filter(n => n.type === "stage").map(n => (n.data as any).stage as string);
      if (stages.includes("frame_extract") && stages[0] !== "frame_extract") {
        errors.push(t('ingestionProfiles.frameExtractFirstError'));
      }
      if (stages.includes("dedup") && stages[stages.length - 1] !== "dedup") {
        errors.push("Deduplication should be the last stage");
      }
      // normalize should come after all detect/capture stages
      if (stages.includes("normalize")) {
        const normIdx = stages.indexOf("normalize");
        const laterStages = stages.slice(normIdx + 1);
        const illegalAfter = ["scene_change", "motion", "burst", "object_filter"];
        for (const ill of illegalAfter) {
          if (laterStages.includes(ill)) {
            errors.push("Normalize must come after " + ill + " (currently comes before)");
          }
        }
      }
    }
    return errors;
  };

  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const handleSave = async () => {
    const name = editProfile ? editProfile.name : profile.trim();
    if (!name) return;
    const errors = validateProfile();
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }
    setValidationErrors([]);
    setSaveError('');
    setSaving(true);
    try {
      const body = buildProfileBody(name);
      if (editProfile) {
        await api.put("/ingestion/profiles/" + encodeURIComponent(name), body);
      } else {
        await api.post("/ingestion/profiles", body);
      }
      if (onSaved) onSaved();
    } catch (e: any) {
      console.error('Failed to save profile:', e);
      setSaveError(e?.message || t('common.saveFailed', 'Save failed'));
    }
    setSaving(false);
  };

  const handleSaveAs = async () => {
    const src = editProfile ? editProfile.name : profile.trim();
    if (!src || !saveAs.trim()) return;
    setSaving(true);
    setSaveError('');
    try {
      const body = buildProfileBody(saveAs.trim());
      await api.post("/ingestion/profiles/" + encodeURIComponent(src) + "/clone", body);
      setSaveAsOpen(false);
      setSaveAs("");
      if (onSaved) onSaved();
    } catch (e: any) {
      console.error('Failed to clone profile:', e);
      setSaveError(e?.message || t('common.saveFailed', 'Save failed'));
    }
    setSaving(false);
  };

  const profileInspectorView = (
    <div className="space-y-4 pb-20">
      <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{t('ingestionProfiles.profileHeader')}</div>
      <div>
        <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('ingestionProfiles.nameLabel')}</label>
        <input className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none" value={profile} onChange={e => setProfile(e.target.value)} placeholder={t('ingestionProfiles.profileNamePlaceholder')} />
      </div>
      <div>
        <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('ingestionProfiles.descLabel')}</label>
        <textarea className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none resize-none" rows={3} value={profileDesc} onChange={e => setProfileDesc(e.target.value)} placeholder={t('ingestionProfiles.descPlaceholder')} />
      </div>
      <div className="pt-2 border-t border-gray-100">
        <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('ingestionProfiles.pipelineLabelShort')}</label>
        <div className="space-y-1">
          {(() => {
            const s = nodes.filter(n => n.type === "stage");
            if (s.length === 0) return <p className="text-[9px] text-gray-400 italic">{t('ingestionProfiles.noStagesConfigured')}</p>;
            return s.map((n, i) => (
              <div key={n.id} className="flex items-center gap-2 text-[10px] text-gray-700">
                <span className="size-4 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center text-[8px] font-bold shrink-0">{i+1}</span>
                <span className="truncate">{STAGE_MAP.get(n.data.stage as string)?.label || (n.data.stage as string)}</span>
                {(n.data as any).enabled === false && <span className="text-[8px] text-gray-400 ml-auto">{t('ingestionProfiles.off')}</span>}
              </div>
            ));
          })()}
        </div>
      </div>
      <div>
        <label className="mb-1 block text-[10px] text-gray-500 font-medium">{t('ingestionProfiles.targetDataset')}</label>
        <div className="text-[10px] text-gray-600 font-mono bg-gray-50 rounded-lg px-2.5 py-1.5 border border-gray-200">{editProfile?.target_dataset || profile || t('ingestionProfiles.autoDerive')}</div>
      </div>
    </div>
  );

  const toolbarActions = (
    <div className="flex items-center gap-2">
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
      <button onClick={() => { setSaveAs(profile + "_copy"); setSaveAsOpen(true); }} disabled={saving || !profile.trim()}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors">
        <Copy className="size-3.5" /> {t('ingestionProfiles.cloneLabel')}
      </button>
      <button onClick={handleSave} disabled={saving || !profile.trim() || nodes.filter(n => n.type === "source").length === 0}
        className="flex items-center gap-1.5 px-4 py-1.5 text-xs font-medium bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 transition-colors">
        {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
        {editProfile ? "Save" : "Create"}
      </button>
      {saveError && (
        <div className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg">
          <AlertTriangle className="size-3.5 shrink-0" />
          <span className="truncate max-w-xs">{saveError}</span>
        </div>
      )}
    </div>
  );

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-200 bg-white shrink-0">
        <button onClick={() => { if (onSaved) onSaved(); }} className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700">
          <ChevronLeft className="size-3" /> Back
        </button>
        <span className="text-xs font-medium text-gray-400">|</span>
        <span className="text-xs font-medium text-gray-700">{editProfile ? ": " + editProfile.name : t('ingestionProfiles.newProfile')}</span>
        {editProfile && <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-medium">{t('ingestionProfiles.editing')}</span>}
        <div className="ml-auto" />
        {validationErrors.length > 0 && (
          <div className="flex items-center gap-1.5 text-[10px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-2.5 py-1 max-w-[300px]">
            <AlertTriangle className="size-3 shrink-0" />
            <span className="truncate">{validationErrors[0]}</span>
            <button onClick={() => setValidationErrors([])} className="shrink-0 hover:text-red-800"><X className="size-3" /></button>
          </div>
        )}
        {toolbarActions}
      </div>
      <div className="flex-1 flex min-h-0">
        <div className="w-[220px] shrink-0 border-r border-gray-200 bg-white overflow-y-auto flex flex-col">
          <div className="flex border-b border-gray-200">
            {(["sources","stages","dataset"] as const).map(tab => (
              <button key={tab} onClick={() => setSideTab(tab)}
                className={`flex-1 text-[10px] font-medium py-2 transition-colors ${sideTab===tab ? "text-emerald-600 border-b-2 border-emerald-500 bg-emerald-50/50" : "text-gray-500 hover:text-gray-700"}`}>
                {tab === "sources" ? t('ingestionProfiles.sourcesTab') : tab === "stages" ? t('ingestionProfiles.stagesTab') : t('ingestionProfiles.datasetTab')}
              </button>
            ))}
          </div>
          {sideTab === "sources" && (
            <div className="p-2 space-y-0.5 overflow-y-auto">
              {storages.length === 0 ? (
                <div className="text-[10px] text-gray-400 py-6 text-center"><HardDrive className="size-6 mx-auto mb-2 text-gray-300" /><br /><span className="text-[9px]">{t('ingestionProfiles.addVideosHintShort')}</span></div>
              ) : storages.map(src => (
                <button key={src.name} onClick={() => add(src)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded hover:bg-gray-100 text-left transition-colors">
                  {src.type === "video" ? <Film className="size-3 text-sky-500 shrink-0" /> : <Image className="size-3 text-emerald-500 shrink-0" />}
                  <div className="min-w-0 flex-1"><div className="text-[10px] text-gray-700 truncate">{src.name}</div><div className="text-[8px] text-gray-400 truncate">{src.path}</div></div>
                  <span className="text-[7px] px-1 py-0.5 rounded bg-gray-100 text-gray-500 uppercase">{src.type}</span>
                </button>
              ))}
            </div>
          )}
          {sideTab === "stages" && <StagePalette onAdd={addStage} />}
          {sideTab === "dataset" && (
            <div className="p-3 space-y-3">
              <p className="text-[10px] text-gray-500">{t('ingestionProfiles.outputDataset')}</p>
              <button onClick={() => add({} as any)} className="w-full flex items-center gap-2 px-3 py-2 border border-dashed border-violet-300 rounded-lg text-[10px] text-violet-600 hover:bg-violet-50 transition-colors"><Database className="size-3" /> {t('ingestionProfiles.addOutput')}</button>
            </div>
          )}
        </div>
        <div className="flex-1" style={{ height: "calc(100vh - 110px)" }}>
          <ReactFlow nodeTypes={CUSTOM_NODE_TYPES} nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} onDragOver={onDragOver}
            onNodeClick={(_,n) => setSelectedId(n.id)} onPaneClick={() => setSelectedId(null)}
            onNodesDelete={reconnect}
            fitView deleteKeyCode="Delete" multiSelectionKeyCode="Shift">
            <Background gap={24} size={1} color="#e5e7eb" />
            <Controls />
          </ReactFlow>
        </div>
        <div className="w-[260px] shrink-0 border-l border-gray-200 bg-white overflow-y-auto p-3">
          {selectedNode ? <NodeInspector node={selectedNode} onUpdate={updateNode} /> : profileInspectorView}
        </div>
      </div>
      {saveAsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setSaveAsOpen(false)}>
          <div className="bg-white rounded-xl shadow-xl p-5 w-80" onClick={e => e.stopPropagation()}>
            <h3 className="text-sm font-semibold text-gray-800 mb-3">{t('ingestionProfiles.saveAsTitle')}</h3>
            <input autoFocus className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm outline-none mb-3"
              placeholder={t('ingestionProfiles.newNamePlaceholder')} value={saveAs} onChange={e => setSaveAs(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") handleSaveAs(); if (e.key === "Escape") setSaveAsOpen(false); }} />
            <div className="flex items-center gap-2 justify-end">
              <button onClick={() => setSaveAsOpen(false)} className="px-3 py-1.5 text-xs text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50">{t('common.cancel')}</button>
              <button onClick={handleSaveAs} disabled={saving || !saveAs.trim()}
                className="px-3 py-1.5 text-xs text-white bg-emerald-600 rounded-lg hover:bg-emerald-700 disabled:opacity-50 flex items-center gap-1">
                {saving ? <Loader2 className="size-3 animate-spin" /> : <Check className="size-3" />} Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ──  Selector (fetches available datasets from API) ──

function Selector({ value, onChange }: { value: string; onChange: (name: string) => void }) {
  const { t } = useTranslation();
  const [datasets, sets] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    api.get<any[]>("/datasets")
      .then(data => {
        const arr = Array.isArray(data) ? data : (data as any).datasets || (data as any).items || [];
        const names = arr.map((d: any) => d.name || d);
        sets(names);
      })
      .catch((e) => { console.error('Failed to load datasets for selector:', e); sets([]); })
      .finally(() => setLoading(false));
  }, []);
  if (loading) return <div className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-[10px] text-gray-400">{t('ingestionProfiles.loadingDatasetsShort')}</div>;
  if (datasets.length === 0) return <div className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[10px] text-gray-400">{t('ingestionProfiles.noDatasetsShort')}</div>;
  return (
    <select className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs outline-none" value={value} onChange={e => onChange(e.target.value)}>
      <option value="">{t('ingestionProfiles.selectDataset')}</option>
      {datasets.map(ds => <option key={ds} value={ds}>{ds}</option>)}
    </select>
  );
}

// ── Main Page ──

export default function IngestionProfilesPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);
  const [runMsg, setRunMsg] = useState<string | null>(null);
  const [loadError, setLoadError] = useState('');
  const [sources, sets] = useState<Storage[]>([]);
  const [view, setView] = useState<"list" | "editor">("list");
  const [editProfile, setProfile] = useState<ProfileSummary | null>(null);

  useEffect(() => { loadAll(); }, []);

  const loadAll = async () => {
    setLoading(true);
    try {
      const [p, s] = await Promise.all([
        api.get<any>("/ingestion/profiles"),
        api.get<any>("/storage/sources"),
      ]);
      setProfiles(p.profiles || []);
      sets(s || []);
    } catch (e) { console.error('Failed to load profiles and sources:', e); }
    setLoading(false);
  };

  const handleRun = async (name: string) => {
    setRunning(name); setRunMsg(null);
    try {
      const r = await api.post<any>("/ingestion/profiles/" + encodeURIComponent(name) + "/run");
      if (r.job_id) {
        setRunMsg("Ingestion job " + r.job_id + " started. Redirecting to Jobs page...");
        setTimeout(() => { navigate("/jobs"); }, 1500);
      } else {
        setRunMsg(name + ": started successfully");
      }
    } catch (e) { console.error('Failed to start ingestion job:', e); setRunMsg(name + ": failed to start job"); }
    setRunning(null);
    loadAll();
  };

  const handleDelete = async (name: string) => {
    if (!confirm(t('confirm.deleteProfile', { name }))) return;
    await api.del("/ingestion/profiles/" + encodeURIComponent(name));
    loadAll();
  };

  const handle = async (name: string) => {
    try {
      setLoadError('');
      const p = await api.get<any>("/ingestion/profiles/" + encodeURIComponent(name));
      setProfile({ name: p.name, description: p.description, source_count: (p.sources || []).length, pipeline: p.pipeline, target_dataset: p.target_dataset, created_at: p.created_at, sources: p.sources, decode: p.decode });
      setView("editor");
    } catch (e: any) {
      console.error('Failed to load profile for editing:', e);
      setLoadError(e?.message || t('ingestionProfiles.loadFailed', 'Failed to load profile'));
    }
  };

  const handleNewProfile = () => { setProfile(null); setView("editor"); };

  const handleorBack = () => { setView("list"); setProfile(null); loadAll(); };

  if (view === "editor") {
    return (
      <ReactFlowProvider>
        <Flowor sources={sources} editProfile={editProfile} onSaved={handleorBack} />
      </ReactFlowProvider>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {loadError && (
        <div className="mb-4 flex items-center gap-2 px-4 py-2.5 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
          <AlertTriangle className="size-4 shrink-0" />
          <span>{loadError}</span>
        </div>
      )}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2"><RefreshCw className="size-6 text-emerald-600" /> {t('ingestionProfiles.titleShort')}</h1>
          <p className="text-sm text-gray-500 mt-1">{t('ingestionProfiles.descShort')}</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={loadAll} className="flex items-center gap-1 px-3 py-1.5 border border-gray-200 rounded-lg text-xs text-gray-600 hover:bg-gray-50"><RefreshCw className="size-3" /> {t('ingestionProfiles.refresh')}</button>
          <button onClick={handleNewProfile} className="flex items-center gap-1.5 px-4 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-medium hover:bg-emerald-700"><Plus className="size-3.5" /> {t('ingestionProfiles.newProfileShort')}</button>
        </div>
      </div>
      <div className="text-xs bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-emerald-700 mb-6 leading-relaxed">
        <strong>:</strong> Raw Data (storage/) &rarr; <strong>{t('ingestionProfiles.conceptName')}</strong> &rarr; Asset  (datasets/)
      </div>
      {runMsg && (
        <div className="flex items-center gap-2 px-4 py-2.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-700 mb-4"><FileText className="size-3.5" /> {runMsg}</div>
      )}
      {loading ? (
        <div className="flex items-center justify-center py-20"><Loader2 className="size-5 animate-spin text-gray-400" /></div>
      ) : profiles.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <RefreshCw className="size-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-sm font-medium text-gray-500 mb-1">{t('ingestionProfiles.noProfiles')}</h3>
          <p className="text-xs text-gray-400 mb-6">{t('ingestionProfiles.noProfilesHint')}</p>
          <button onClick={handleNewProfile} className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 transition-colors"><Plus className="size-4" /> {t('ingestionProfiles.createProfileShort')}</button>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {profiles.map(p => (
            <div key={p.name} className="bg-white rounded-lg border border-gray-200 p-4 hover:border-emerald-200 transition-all">
              <div className="flex items-start gap-3">
                <div className="size-10 rounded-lg bg-emerald-100 text-emerald-600 flex items-center justify-center"><GitFork className="size-5" /></div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium truncate flex items-center gap-1.5">
                    {p.name}
                    {(p.name === "default_video" || p.name === "default_image") && <span className="text-[8px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-600 font-medium uppercase tracking-wider">{t('ingestionProfiles.defaultBadge')}</span>}
                  </h3>
                  <p className="text-[10px] text-gray-400 truncate mt-0.5">{p.description || "No description"}</p>
                </div>
              </div>
              <div className="mt-3 space-y-1.5">
                <div className="flex items-center gap-2 text-[10px] text-gray-500"><Camera className="size-3" /><span>{p.source_count} source(s): {p.sources?.map(s => s.name).join(", ") || "-"}</span></div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500"><GitFork className="size-3" /><span>{t('ingestionProfiles.pipelinePrefix')}: {p.pipeline?.join(", ") || "-"}</span></div>
                {p.target_dataset && <div className="flex items-center gap-2 text-[10px] text-gray-500"><Database className="size-3" /><span>Target: {p.target_dataset}</span></div>}
              </div>
              <div className="mt-3 pt-3 border-t border-gray-100 flex items-center justify-between">
                <button onClick={() => handleRun(p.name)} disabled={running === p.name}
                  className="flex items-center gap-1 px-2.5 py-1 text-[10px] bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors">
                  {running === p.name ? <Loader2 className="size-3 animate-spin" /> : <Play className="size-3" />} Run
                </button>
                <div className="flex items-center gap-1">
                  <button onClick={() => handle(p.name)} className="flex items-center gap-1 px-2 py-1 text-[10px] text-gray-600 hover:bg-gray-100 rounded transition-colors"><Edit3 className="size-3" /> {t('ingestionProfiles.editShort')}</button>
                  <button onClick={() => handleDelete(p.name)} className="flex items-center gap-1 px-2 py-1 text-[10px] text-red-500 hover:bg-red-50 rounded transition-colors" aria-label={t('ingestionProfiles.delete')}><Trash2 className="size-3" /></button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
