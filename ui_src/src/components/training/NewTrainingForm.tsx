import React, { useState, useEffect } from 'react';
import { HelpCircle, ChevronDown, ChevronRight, Play } from 'lucide-react';
import { startTraining, fetchTrainingProjects, createTrainingProject } from '../../api/training';
import type { TrainingProject } from '../../types';

type PresetId = 'quick_test' | 'standard' | 'high_accuracy' | 'small_gpu' | 'edge' | 'custom';

interface Preset {
  label: string;
  description: string;
  values: Partial<TrainingFormValues>;
}

interface TrainingFormValues {
  project: string;
  backend: string;
  model: string;
  dataset_version: string;
  device: string;
  epochs: number;
  batch: number;
  imgsz: number;
  workers: number;
  seed: number;
  optimizer: string;
  lr: number;
  lrf: number;
  warmup_epochs: number;
  warmup_momentum: number;
  warmup_bias_lr: number;
  momentum: number;
  patience: number;
  amp: boolean;
  resume: boolean;
  freeze: number;
  nbs: number;
  label_smoothing: number;
  close_mosaic: number;
  dropout: number;
  fliplr: number;
  flipud: number;
  scale: number;
  translate: number;
  perspective: number;
  mosaic: number;
  mixup: number;
  copy_paste: number;
  copy_paste_mode: string;
  auto_augment: string;
  erasing: number;
  crop_fraction: number;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  val_split: string;
  conf: number;
  iou: number;
  save_best: boolean;
  save_period: number;
  plots: boolean;
  confusion_matrix: boolean;
  cache: string;
  pin_memory: boolean;
  deterministic: boolean;
  compile: boolean;
  box: number;
  cls: number;
  dfl: number;
  multi_scale: boolean;
  rect: boolean;
  cos_lr: boolean;
  overlap_mask: boolean;
  single_cls: boolean;
  // rf-detr (TrainConfig whitelist) — see raven_rfdetr.config._TRAIN_WHITELIST
  lr_encoder: number;
  weight_decay: number;
  grad_accum_steps: number;
  use_ema: boolean;
  ema_decay: number;
  early_stopping: boolean;
  early_stopping_patience: number;
  checkpoint_interval: number;
  eval_interval: number;
  export_format: string;
  export_opset: number;
  export_half: boolean;
  export_dynamic: boolean;
  export_simplify: boolean;
}

const PRESETS: Record<PresetId, Preset> = {
  quick_test: {
    label: 'Quick Test',
    description: '5 epochs, minimal augment, fast validation',
    values: { epochs: 5, batch: 4, imgsz: 320, amp: true, mosaic: 0.0, mixup: 0, copy_paste: 0, auto_augment: '', erasing: 0, hsv_h: 0, hsv_s: 0, hsv_v: 0, fliplr: 0, scale: 0, translate: 0, workers: 2, save_period: 5 },
  },
  standard: {
    label: 'Standard',
    description: '100 epochs, balanced augment, good starting point',
    values: { epochs: 100, batch: 16, imgsz: 640, amp: true, optimizer: 'AdamW', lr: 0.001, lrf: 0.01, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, momentum: 0.937, patience: 20, mosaic: 1.0, mixup: 0.1, copy_paste: 0, copy_paste_mode: 'flip', auto_augment: 'randaugment', erasing: 0.4, crop_fraction: 1.0, hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, fliplr: 0.5, translate: 0.1, workers: 8, save_period: 10, close_mosaic: 10, cos_lr: true, box: 7.5, cls: 0.5, dfl: 1.5, plots: true },
  },
  high_accuracy: {
    label: 'High Accuracy',
    description: '300 epochs, strong augment, EMA, best for production',
    values: { epochs: 300, batch: 8, imgsz: 960, amp: true, optimizer: 'AdamW', lr: 0.001, lrf: 0.01, warmup_epochs: 5, warmup_momentum: 0.8, warmup_bias_lr: 0.1, momentum: 0.937, patience: 50, mosaic: 1.0, mixup: 0.3, copy_paste: 0.3, copy_paste_mode: 'mix', auto_augment: 'randaugment', erasing: 0.4, crop_fraction: 1.0, hsv_h: 0.02, hsv_s: 0.8, hsv_v: 0.5, fliplr: 0.5, translate: 0.2, perspective: 0.001, workers: 8, save_period: 20, close_mosaic: 15, cos_lr: true, box: 7.5, cls: 0.5, dfl: 1.5, plots: true },
  },
  small_gpu: {
    label: 'Small GPU',
    description: 'Optimized for limited VRAM (4-6GB)',
    values: { epochs: 100, batch: 4, imgsz: 416, amp: true, optimizer: 'AdamW', lr: 0.001, lrf: 0.01, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, momentum: 0.937, patience: 30, mosaic: 0.5, mixup: 0, copy_paste: 0, close_mosaic: 5, hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, fliplr: 0.5, workers: 4, save_period: 10 },
  },
  edge: {
    label: 'Edge Deployment',
    description: 'Small model, FP16, tuned for edge devices',
    values: { epochs: 100, batch: 16, imgsz: 640, amp: true, optimizer: 'SGD', lr: 0.01, lrf: 0.01, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, momentum: 0.937, patience: 20, mosaic: 0.0, mixup: 0, copy_paste: 0, hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, fliplr: 0.5, close_mosaic: 0, workers: 4, save_best: true, save_period: 10 },
  },
  custom: {
    label: 'Custom',
    description: 'Start from scratch, manually configure all',
    values: {},
  },
};

// Backend catalog — mirrors raven_trainer.cli.new.SUPPORTED_BACKENDS.
const BACKEND_OPTIONS = [
  { label: 'Ultralytics', value: 'ultralytics' },
  { label: 'MMDetection', value: 'mmdetection' },
  { label: 'Anomalib', value: 'anomalib' },
  { label: 'RF-DETR', value: 'rfdetr' },
];

// Per-backend model catalogs — mirror raven_trainer.cli.new.SUPPORTED_TASKS and the
// adapters' supported_models (yolo26 + yolo11 families + rtdetr-l; rf-detr det/seg variants).
// YOLO26/YOLO11 sizes follow raven-onnxruntime models/yolo26|yolo11 (n/s/m/l/x per task).
const MODEL_CATALOG: Record<string, { label: string; models: string[] }[]> = {
  ultralytics: [
    { label: 'Detection (YOLO26)', models: ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'] },
    { label: 'Segmentation (YOLO26)', models: ['yolo26n-seg', 'yolo26s-seg', 'yolo26m-seg', 'yolo26l-seg', 'yolo26x-seg'] },
    { label: 'Pose (YOLO26)', models: ['yolo26n-pose', 'yolo26s-pose', 'yolo26m-pose', 'yolo26l-pose', 'yolo26x-pose'] },
    { label: 'Classification (YOLO26)', models: ['yolo26n-cls', 'yolo26s-cls', 'yolo26m-cls', 'yolo26l-cls', 'yolo26x-cls'] },
    { label: 'OBB (YOLO26)', models: ['yolo26n-obb', 'yolo26s-obb', 'yolo26m-obb', 'yolo26l-obb', 'yolo26x-obb'] },
    { label: 'Detection (YOLO11)', models: ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'] },
    { label: 'Segmentation (YOLO11)', models: ['yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg'] },
    { label: 'Pose (YOLO11)', models: ['yolo11n-pose', 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose', 'yolo11x-pose'] },
    { label: 'Classification (YOLO11)', models: ['yolo11n-cls', 'yolo11s-cls', 'yolo11m-cls', 'yolo11l-cls', 'yolo11x-cls'] },
    { label: 'OBB (YOLO11)', models: ['yolo11n-obb', 'yolo11s-obb', 'yolo11m-obb', 'yolo11l-obb', 'yolo11x-obb'] },
    { label: 'RT-DETR (Detection)', models: ['rtdetr-l'] },
  ],
  rfdetr: [
    { label: 'Detection', models: ['rf-detr-nano', 'rf-detr-small', 'rf-detr-medium', 'rf-detr-large'] },
    { label: 'Segmentation', models: ['rf-detr-seg-nano', 'rf-detr-seg-small', 'rf-detr-seg-medium', 'rf-detr-seg-large'] },
  ],
};

// Sentinel for the "Custom model…" escape hatch in the model dropdown.
const CUSTOM_MODEL = '__custom__';

const DEFAULT_MODEL: Record<string, string> = {
  ultralytics: 'yolo11n',
  rfdetr: 'rf-detr-nano',
  mmdetection: 'rtdetr-l',
  anomalib: 'patchcore',
};

function modelInCatalog(backend: string, model: string): boolean {
  const groups = MODEL_CATALOG[backend];
  if (!groups) return false;
  return groups.some((g) => g.models.includes(model));
}

const DEFAULTS: TrainingFormValues = {
  project: '', backend: 'ultralytics', model: 'yolo11n',
  dataset_version: '', device: 'cuda:0',
  epochs: 100, batch: 16, imgsz: 640, workers: 8, seed: 42,
  optimizer: 'AdamW', lr: 0.001, lrf: 0.01,
  warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, momentum: 0.937,
  patience: 20, amp: true, resume: false, freeze: 0, nbs: 64,
  label_smoothing: 0, close_mosaic: 10, dropout: 0,
  fliplr: 0.5, flipud: 0, scale: 0.5, translate: 0.1, perspective: 0,
  mosaic: 1.0, mixup: 0.1, copy_paste: 0, copy_paste_mode: 'flip',
  auto_augment: 'randaugment', erasing: 0.4, crop_fraction: 1.0,
  hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4,
  val_split: 'val', conf: 0.25, iou: 0.7,
  save_best: true, save_period: 10, plots: true, confusion_matrix: true,
  cache: 'disk', pin_memory: true, deterministic: false, compile: false,
  box: 7.5, cls: 0.5, dfl: 1.5,
  multi_scale: false, rect: false, cos_lr: true, overlap_mask: true,
  single_cls: false,
  lr_encoder: 0.00015, weight_decay: 0.0001, grad_accum_steps: 4,
  use_ema: true, ema_decay: 0.993, early_stopping: false,
  early_stopping_patience: 10, checkpoint_interval: 10, eval_interval: 1,
  export_format: 'onnx', export_opset: 17,
  export_half: false, export_dynamic: false, export_simplify: true,
};

function Section({ title, defaultOpen = true, children, count }: {
  title: string; defaultOpen?: boolean; children: React.ReactNode; count?: number;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button type="button" onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-4 py-2.5 bg-gray-50 hover:bg-gray-100 text-sm font-medium">
        <span>{title}</span>
        <div className="flex items-center gap-2">
          {count !== undefined && <span className="text-xs text-gray-400">{count} params</span>}
          {open ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
        </div>
      </button>
      {open && <div className="p-4 space-y-3">{children}</div>}
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1 flex items-center gap-1">
        {label}
        {hint && <span title={hint}><HelpCircle className="w-3 h-3 text-gray-300" /></span>}
      </label>
      {children}
    </div>
  );
}

function Input({ value, onChange, type = 'number', min, max, step, className, placeholder }: {
  value: number | string; onChange: (v: string) => void;
  type?: string; min?: number; max?: number; step?: number; className?: string; placeholder?: string;
}) {
  return (
    <input type={type} value={value} onChange={(e) => onChange(e.target.value)}
      min={min} max={max} step={step} placeholder={placeholder}
      className={`w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${className || ''}`} />
  );
}

function Select({ value, onChange, options, groups }: {
  value: string; onChange: (v: string) => void; options: { label: string; value: string }[];
  groups?: { label: string; models: string[] }[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}
      className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
      {groups && groups.map((g) => (
        <optgroup key={g.label} label={g.label}>
          {g.models.map((m) => <option key={m} value={m}>{m}</option>)}
        </optgroup>
      ))}
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

function Toggle({ value, onChange, label }: { value: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)}
        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
      {label && <span className="text-xs text-gray-600">{label}</span>}
    </label>
  );
}

interface Props {
  onTrainingStarted: (experimentId: string) => void;
  datasetYaml?: string;
  initialProject?: string;
}

// Per-backend hyperparameter payloads.
//
// The server (_do_train) merges `hyperparameters` into train_cfg, and each adapter
// whitelists the kwargs it accepts (raven_ultralytics._YOLO_TRAIN_KWARGS / raven_rfdetr
// filter_train_kwargs). Sending backend-specific keys only prevents wrong fills:
// e.g. a YOLO-only mosaic/mixup value must never reach rf-detr's TrainConfig
// (extra="forbid"), and rf-detr-only lr_encoder/eval_interval must never reach YOLO.
function buildHyperparameters(values: TrainingFormValues): Record<string, unknown> {
  if (values.backend === 'ultralytics') {
    // Keys must match raven_ultralytics._YOLO_TRAIN_KWARGS exactly (unknown keys are
    // silently dropped); non-train kwargs (conf/iou/lr_scheduler/copy_paste_mode/...) are omitted.
    return {
      optimizer: values.optimizer,
      lr0: values.lr,
      lrf: values.lrf,
      momentum: values.momentum,
      warmup_epochs: values.warmup_epochs,
      warmup_momentum: values.warmup_momentum,
      warmup_bias_lr: values.warmup_bias_lr,
      patience: values.patience,
      nbs: values.nbs,
      label_smoothing: values.label_smoothing,
      close_mosaic: values.close_mosaic,
      dropout: values.dropout,
      fliplr: values.fliplr,
      flipud: values.flipud,
      scale: values.scale,
      translate: values.translate,
      perspective: values.perspective,
      mosaic: values.mosaic,
      mixup: values.mixup,
      copy_paste: values.copy_paste,
      auto_augment: values.auto_augment,
      erasing: values.erasing,
      crop_fraction: values.crop_fraction,
      hsv_h: values.hsv_h,
      hsv_s: values.hsv_s,
      hsv_v: values.hsv_v,
      box: values.box,
      cls: values.cls,
      dfl: values.dfl,
      multi_scale: values.multi_scale,
      rect: values.rect,
      cos_lr: values.cos_lr,
      overlap_mask: values.overlap_mask,
      single_cls: values.single_cls,
      amp: values.amp,
      resume: values.resume,
      deterministic: values.deterministic,
      compile: values.compile,
      freeze: values.freeze,
      cache: values.cache,
      pin_memory: values.pin_memory,
      save: values.save_best,
      save_period: values.save_period,
      plots: values.plots,
      seed: values.seed,
      workers: values.workers,
    };
  }
  if (values.backend === 'rfdetr') {
    return {
      lr: values.lr,
      lr_encoder: values.lr_encoder,
      weight_decay: values.weight_decay,
      warmup_epochs: values.warmup_epochs,
      grad_accum_steps: values.grad_accum_steps,
      use_ema: values.use_ema,
      ema_decay: values.ema_decay,
      early_stopping: values.early_stopping,
      early_stopping_patience: values.early_stopping_patience,
      checkpoint_interval: values.checkpoint_interval,
      eval_interval: values.eval_interval,
      batch_size: values.batch,
      num_workers: values.workers,
      seed: values.seed,
    };
  }
  return {};
}

export function NewTrainingForm({ onTrainingStarted, datasetYaml, initialProject }: Props) {
  const [preset, setPreset] = useState<PresetId>('standard');
  const [datasetPath, setDatasetPath] = useState(datasetYaml || '');

  useEffect(() => {
    if (datasetYaml) {
      setDatasetPath(datasetYaml);
    }
    if (initialProject) {
      setValues(prev => ({ ...prev, project: initialProject }));
    }
  }, [datasetYaml, initialProject]);
  const [values, setValues] = useState<TrainingFormValues>({ ...DEFAULTS, ...PRESETS.standard.values });
  const [projects, setProjects] = useState<TrainingProject[]>([]);
  const [starting, setStarting] = useState(false);
    const [showCreateProject, setShowCreateProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);

  useEffect(() => {
    fetchTrainingProjects().then(setProjects).catch((e) => console.error('Failed to load training projects:', e));
  }, []);

  const update = <K extends keyof TrainingFormValues>(key: K, val: TrainingFormValues[K]) => {
    setValues((prev) => ({ ...prev, [key]: val }));
  };

  const applyPreset = (id: PresetId) => {
    setPreset(id);
    if (id !== 'custom') {
      setValues((prev) => ({ ...prev, ...PRESETS[id].values }));
    }
  };
  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    setCreatingProject(true);
    try {
      await createTrainingProject({
        name: newProjectName.trim(),
        task: 'detection',
        backend: values.backend,
        model: values.model,
      });
      setShowCreateProject(false);
      setNewProjectName('');
      const updated = await fetchTrainingProjects();
      setProjects(updated);
      if (updated.length > 0) {
        const p = updated[0];
        setValues(prev => ({ ...prev, project: p.name, backend: p.backend, model: p.model }));
      }
    } catch (e) {
      console.error('Failed to create project:', e);
    }
    setCreatingProject(false);
  };

  const handleSubmit = async () => {
    setStarting(true);
    try {
      const config = {
        project: values.project,
        model: values.model,
        backend: values.backend,
        epochs: values.epochs,
        batch: values.batch,
        imgsz: values.imgsz,
        device: values.device,
        export_format: values.export_format,
        export_opset: values.export_opset,
        export_half: values.export_half,
        export_dynamic: values.export_dynamic,
        export_simplify: values.export_simplify,
        // 后端专属超参数：服务端 merge 进 train_cfg 后由各适配器 whitelist 过滤。
        // 只提交当前后端认识的 key，防止 YOLO 专有参数污染 rf-detr 训练（反之亦然）。
        hyperparameters: buildHyperparameters(values),
      };
      const result = await startTraining(config);
      const expId = result.experiment_id as string;
      onTrainingStarted(expId);
    } catch (e) {
      console.error('Failed to start training:', e);
    }
    setStarting(false);
  };

  const estHours = (values.epochs * values.batch > 0)
    ? Math.round(values.epochs * (values.imgsz / 320) * (16 / values.batch) * 0.012 * 10) / 10
    : 0;

  return (
    <div className="flex gap-6">
      <div className="flex-1 space-y-4 max-w-2xl">

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="text-sm font-medium text-blue-800 mb-2">Training Preset</div>
          <div className="grid grid-cols-3 gap-2">
            {(Object.entries(PRESETS) as [PresetId, Preset][]).map(([id, p]) => (
              <button key={id} type="button" onClick={() => applyPreset(id)}
                className={`text-left px-3 py-2 rounded-lg text-xs border transition-colors ${preset === id
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300'}`}>
                <div className="font-medium text-sm">{p.label}</div>
                <div className={`mt-0.5 ${preset === id ? 'text-blue-100' : 'text-gray-400'}`}>{p.description}</div>
              </button>
            ))}
          </div>
        </div>

        <Section title="Basic" count={10}>
          <Field label="Project" hint="Training project created by `raven-train new`">
            <select className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
              value={values.project}
              onChange={(e) => {
                const p = projects.find((pr) => pr.name === e.target.value);
                update('project', e.target.value);
                if (p) { update('backend', p.backend); update('model', p.model); }
              }}>
              <option value="">Select project...</option>
              {projects.map((p) => <option key={p.name} value={p.name}>{p.display_name || p.name}</option>)}
            </select>
            {projects.length === 0 && (
              <div className="text-xs text-yellow-600 mt-1 flex items-center gap-2">
              <span>No training projects found.</span>
              <button onClick={() => setShowCreateProject(true)} className="text-blue-600 hover:underline">Create one</button>
            </div>
            )}
          </Field>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Backend">
              <Select value={values.backend} onChange={(v) => {
                update('backend', v);
                // 切后端时收敛 model / export_format，避免残留另一后端的模型名：
                // backend=ultralytics + model=rf-detr-nano（或反之）会导致适配器加载失败。
                if (!modelInCatalog(v, values.model)) {
                  update('model', DEFAULT_MODEL[v] || '');
                }
                // rfdetr 导出器只支持 ONNX
                if (v === 'rfdetr' && values.export_format !== 'onnx') update('export_format', 'onnx');
              }}
                options={BACKEND_OPTIONS} />
            </Field>
            <Field label="Model">
              {(values.backend === 'ultralytics' || values.backend === 'rfdetr') ? (
                <div className="space-y-1">
                  <Select
                    value={modelInCatalog(values.backend, values.model) ? values.model : CUSTOM_MODEL}
                    onChange={(v) => {
                      if (v === CUSTOM_MODEL) update('model', '');
                      else update('model', v);
                    }}
                    groups={MODEL_CATALOG[values.backend]}
                    options={[{ label: 'Custom…', value: CUSTOM_MODEL }]} />
                  {(!modelInCatalog(values.backend, values.model) || values.model === '') && (
                    <Input value={values.model} onChange={(v) => update('model', v)} type="text"
                      placeholder="Custom model name (e.g. yolo26s)" />
                  )}
                </div>
              ) : (
                <Input value={values.model} onChange={(v) => update('model', v)} type="text"
                  placeholder="e.g. rtdetr-l / patchcore" />
              )}
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Dataset" hint="Dataset and version from Data Assets">
              <Input value={values.dataset_version} onChange={(v) => update('dataset_version', v)} type="text" placeholder="person:v1" />
            </Field>
            <Field label="Device" hint="Training device">
              <Select value={values.device} onChange={(v) => update('device', v)}
                options={[{ label: 'Auto', value: 'auto' }, { label: 'CUDA:0', value: 'cuda:0' }, { label: 'CPU', value: 'cpu' }, { label: 'MPS', value: 'mps' }]} />
            </Field>
          </div>

          <div className="grid grid-cols-4 gap-3">
            <Field label="Epochs"><Input value={values.epochs} onChange={(v) => update('epochs', Number(v))} min={1} max={1000} /></Field>
            <Field label="Batch"><Input value={values.batch} onChange={(v) => update('batch', Number(v))} min={1} max={256} /></Field>
            <Field label="Image Size"><Input value={values.imgsz} onChange={(v) => update('imgsz', Number(v))} step={32} /></Field>
            <Field label="Workers"><Input value={values.workers} onChange={(v) => update('workers', Number(v))} min={0} max={32} /></Field>
          </div>

          <Field label="Seed"><Input value={values.seed} onChange={(v) => update('seed', Number(v))} /></Field>
        </Section>

        {(values.backend === 'ultralytics') && (
        <Section title="Optimization" count={12}>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Optimizer">
              <Select value={values.optimizer} onChange={(v) => update('optimizer', v)}
                options={[{ label: 'Auto', value: 'auto' }, { label: 'SGD', value: 'SGD' }, { label: 'Adam', value: 'Adam' }, { label: 'AdamW', value: 'AdamW' }]} />
            </Field>
            <Field label="Learning Rate (lr0)">
              <Input value={values.lr} onChange={(v) => update('lr', Number(v))} step={0.0001} />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Final LR (lrf)"><Input value={values.lrf} onChange={(v) => update('lrf', Number(v))} step={0.001} /></Field>
            <Field label="Momentum"><Input value={values.momentum} onChange={(v) => update('momentum', Number(v))} step={0.01} /></Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Warmup Epochs">
              <Input value={values.warmup_epochs} onChange={(v) => update('warmup_epochs', Number(v))} min={0} max={50} />
            </Field>
            <Field label="Cos LR" hint="Cosine LR schedule (ultralytics cos_lr)">
              <Toggle value={values.cos_lr} onChange={(v) => update('cos_lr', v)} label="" />
            </Field>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Warmup Momentum"><Input value={values.warmup_momentum} onChange={(v) => update('warmup_momentum', Number(v))} step={0.1} /></Field>
            <Field label="Warmup Bias LR"><Input value={values.warmup_bias_lr} onChange={(v) => update('warmup_bias_lr', Number(v))} step={0.1} /></Field>
            <Field label="Patience"><Input value={values.patience} onChange={(v) => update('patience', Number(v))} min={0} max={500} /></Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Freeze Backbone">
              <Select value={String(values.freeze)} onChange={(v) => update('freeze', v === 'all' ? 10 : Number(v))}
                options={[{ label: 'None (0)', value: '0' }, { label: '10 layers', value: '10' }, { label: 'All except head (10)', value: 'all' }]} />
            </Field>
            <Field label="NBS (Nominal Batch)"><Input value={values.nbs} onChange={(v) => update('nbs', Number(v))} min={1} /></Field>
          </div>
          <div className="flex gap-6">
            <Toggle value={values.amp} onChange={(v) => update('amp', v)} label="AMP (Mixed Precision)" />
            <Toggle value={values.resume} onChange={(v) => update('resume', v)} label="Resume" />
            <Toggle value={values.deterministic} onChange={(v) => update('deterministic', v)} label="Deterministic" />
            <Toggle value={values.compile} onChange={(v) => update('compile', v)} label="Torch Compile" />
          </div>
        </Section>
        )}

        {values.backend === 'rfdetr' && (
        <Section title="RF-DETR Training" count={12}>
          <div className="text-xs text-gray-500 mb-2">rf-detr 训练参数（TrainConfig whitelist）。仅提交 rf-detr 认识的 key，避免 YOLO 专有参数进入训练。</div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Learning Rate (lr)">
              <Input value={values.lr} onChange={(v) => update('lr', Number(v))} step={0.0001} />
            </Field>
            <Field label="Encoder LR">
              <Input value={values.lr_encoder} onChange={(v) => update('lr_encoder', Number(v))} step={0.0001} />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Weight Decay"><Input value={values.weight_decay} onChange={(v) => update('weight_decay', Number(v))} step={0.0001} /></Field>
            <Field label="Warmup Epochs"><Input value={values.warmup_epochs} onChange={(v) => update('warmup_epochs', Number(v))} min={0} max={50} /></Field>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Grad Accum Steps"><Input value={values.grad_accum_steps} onChange={(v) => update('grad_accum_steps', Number(v))} min={1} /></Field>
            <Field label="EMA Decay"><Input value={values.ema_decay} onChange={(v) => update('ema_decay', Number(v))} step={0.001} /></Field>
            <Field label="Eval Interval"><Input value={values.eval_interval} onChange={(v) => update('eval_interval', Number(v))} min={1} /></Field>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Checkpoint Interval"><Input value={values.checkpoint_interval} onChange={(v) => update('checkpoint_interval', Number(v))} min={1} /></Field>
            <Field label="Early Stop Patience"><Input value={values.early_stopping_patience} onChange={(v) => update('early_stopping_patience', Number(v))} min={1} /></Field>
          </div>
          <div className="flex gap-6">
            <Toggle value={values.use_ema} onChange={(v) => update('use_ema', v)} label="EMA" />
            <Toggle value={values.early_stopping} onChange={(v) => update('early_stopping', v)} label="Early Stopping" />
          </div>
        </Section>
        )}

        {(values.backend === 'ultralytics') && (
        <Section title="Loss Weights" count={5}>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Box Loss"><Input value={values.box} onChange={(v) => update('box', Number(v))} step={0.5} /></Field>
            <Field label="Cls Loss"><Input value={values.cls} onChange={(v) => update('cls', Number(v))} step={0.1} /></Field>
            <Field label="DFL Loss"><Input value={values.dfl} onChange={(v) => update('dfl', Number(v))} step={0.1} /></Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Label Smoothing"><Input value={values.label_smoothing} onChange={(v) => update('label_smoothing', Number(v))} step={0.01} /></Field>
          </div>
        </Section>
        )}

        {(values.backend === 'ultralytics') && (
        <Section title="Augmentation — Color &amp; Geometry" count={10}>
          <div className="text-xs text-gray-500 mb-2 font-medium">Color (HSV)</div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="HSV-Hue"><Input value={values.hsv_h} onChange={(v) => update('hsv_h', Number(v))} step={0.001} /></Field>
            <Field label="HSV-Saturation"><Input value={values.hsv_s} onChange={(v) => update('hsv_s', Number(v))} step={0.01} /></Field>
            <Field label="HSV-Value"><Input value={values.hsv_v} onChange={(v) => update('hsv_v', Number(v))} step={0.01} /></Field>
          </div>
          <div className="text-xs text-gray-500 mb-2 mt-3 font-medium">Geometry</div>
          <div className="grid grid-cols-4 gap-3">
            <Field label="Flip LR"><Input value={values.fliplr} onChange={(v) => update('fliplr', Number(v))} step={0.1} min={0} max={1} /></Field>
            <Field label="Flip UD"><Input value={values.flipud} onChange={(v) => update('flipud', Number(v))} step={0.1} min={0} max={1} /></Field>
            <Field label="Scale"><Input value={values.scale} onChange={(v) => update('scale', Number(v))} step={0.1} /></Field>
            <Field label="Translate"><Input value={values.translate} onChange={(v) => update('translate', Number(v))} step={0.05} /></Field>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Perspective"><Input value={values.perspective} onChange={(v) => update('perspective', Number(v))} step={0.0001} /></Field>
          </div>
        </Section>
        )}

        {(values.backend === 'ultralytics') && (
        <Section title="Augmentation — Advanced" count={10}>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Mosaic"><Input value={values.mosaic} onChange={(v) => update('mosaic', Number(v))} step={0.1} min={0} max={1} /></Field>
            <Field label="MixUp"><Input value={values.mixup} onChange={(v) => update('mixup', Number(v))} step={0.1} min={0} max={1} /></Field>
            <Field label="CopyPaste"><Input value={values.copy_paste} onChange={(v) => update('copy_paste', Number(v))} step={0.1} min={0} max={1} /></Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="CopyPaste Mode">
              <Select value={values.copy_paste_mode} onChange={(v) => update('copy_paste_mode', v)}
                options={[{ label: 'Flip', value: 'flip' }, { label: 'Mix', value: 'mix' }]} />
            </Field>
            <Field label="Auto Augment">
              <Select value={values.auto_augment} onChange={(v) => update('auto_augment', v)}
                options={[{ label: 'RandAugment', value: 'randaugment' }, { label: 'AutoAugment', value: 'autoaugment' }, { label: 'None', value: '' }]} />
            </Field>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Erasing"><Input value={values.erasing} onChange={(v) => update('erasing', Number(v))} step={0.1} min={0} max={1} /></Field>
            <Field label="Crop Fraction"><Input value={values.crop_fraction} onChange={(v) => update('crop_fraction', Number(v))} step={0.1} /></Field>
          </div>
        </Section>
        )}

        {(values.backend === 'ultralytics') && (
        <Section title="Validation &amp; Runtime" count={10}>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Close Mosaic (epoch)"><Input value={values.close_mosaic} onChange={(v) => update('close_mosaic', Number(v))} min={0} /></Field>
            <Field label="Dropout"><Input value={values.dropout} onChange={(v) => update('dropout', Number(v))} step={0.05} min={0} max={0.9} /></Field>
            <Field label="Confidence"><Input value={values.conf} onChange={(v) => update('conf', Number(v))} step={0.05} min={0} max={1} /></Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Cache">
              <Select value={values.cache} onChange={(v) => update('cache', v)}
                options={[{ label: 'Disk', value: 'disk' }, { label: 'RAM', value: 'ram' }, { label: 'None', value: 'false' }]} />
            </Field>
            <Field label="Save Every (epochs)"><Input value={values.save_period} onChange={(v) => update('save_period', Number(v))} min={-1} /></Field>
          </div>
          <div className="flex gap-6 flex-wrap">
            <Toggle value={values.save_best} onChange={(v) => update('save_best', v)} label="Save Best Only" />
            <Toggle value={values.plots} onChange={(v) => update('plots', v)} label="Generate Plots" />
            <Toggle value={values.pin_memory} onChange={(v) => update('pin_memory', v)} label="Pin Memory" />
          </div>
          <div className="flex gap-6 flex-wrap">
            <Toggle value={values.multi_scale} onChange={(v) => update('multi_scale', v)} label="Multi Scale" />
            <Toggle value={values.rect} onChange={(v) => update('rect', v)} label="Rect Training" />
            <Toggle value={values.overlap_mask} onChange={(v) => update('overlap_mask', v)} label="Overlap Mask" />
            <Toggle value={values.single_cls} onChange={(v) => update('single_cls', v)} label="Single Class" />
          </div>
        </Section>
        )}

        <Section title="Model Export" count={5}>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Export Format">
              {values.backend === 'rfdetr' ? (
                // rfdetr 适配器只支持 ONNX（export.py 对 fmt!=='onnx' 抛错）
                <select className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm bg-gray-50 text-gray-500"
                  value="onnx" disabled>
                  <option value="onnx">ONNX (with verification)</option>
                </select>
              ) : (
                <select className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
                  value={values.export_format}
                  onChange={(e) => update('export_format', e.target.value)}>
                  <option value="onnx">ONNX (with verification)</option>
                  <option value="torchscript">TorchScript</option>
                  <option value="openvino">OpenVINO</option>
                  <option value="tflite">TFLite</option>
                </select>
              )}
            </Field>
            <Field label="Opset Version">
              <Input value={values.export_opset} onChange={(v) => update('export_opset', Number(v))} min={11} max={20} />
            </Field>
          </div>
          <div className="flex gap-6">
            <Toggle value={values.export_half} onChange={(v) => update('export_half', v)} label="FP16 (Half)" />
            <Toggle value={values.export_dynamic} onChange={(v) => update('export_dynamic', v)} label="Dynamic Axes" />
            <Toggle value={values.export_simplify} onChange={(v) => update('export_simplify', v)} label="Simplify (onnxsim)" />
          </div>
        </Section>

        <button onClick={handleSubmit} disabled={!values.project || starting}
          className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed">
          <Play className="w-4 h-4" />
          {starting ? 'Starting Training...' : 'Start Training'}
        </button>
      </div>

      <div className="w-72 shrink-0">
        <div className="bg-white rounded-lg border border-gray-200 p-4 sticky top-6">
          <h3 className="text-sm font-semibold mb-3">Training Summary</h3>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Backend</span><span className="font-medium">{values.backend || '-'}</span></div>
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Model</span><span className="font-medium">{values.model || '-'}</span></div>
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Epochs</span><span className="font-medium">{values.epochs}</span></div>
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Batch</span><span className="font-medium">{values.batch}</span></div>
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Image Size</span><span className="font-medium">{values.imgsz}</span></div>
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Device</span><span className="font-medium">{values.device}</span></div>
            <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">AMP</span><span className="font-medium">{values.amp ? 'On' : 'Off'}</span></div>
            {values.backend === 'ultralytics' ? (
              <>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Optimizer</span><span className="font-medium">{values.optimizer}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">LR / LRF</span><span className="font-medium">{values.lr}/{values.lrf}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Mosaic / MixUp</span><span className="font-medium">{values.mosaic}/{values.mixup}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Close Mosaic</span><span className="font-medium">{values.close_mosaic}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Loss: box/cls/dfl</span><span className="font-medium">{values.box}/{values.cls}/{values.dfl}</span></div>
              </>
            ) : values.backend === 'rfdetr' ? (
              <>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">LR / Encoder LR</span><span className="font-medium">{values.lr}/{values.lr_encoder}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Weight Decay</span><span className="font-medium">{values.weight_decay}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Grad Accum / EMA</span><span className="font-medium">{values.grad_accum_steps}/{values.use_ema ? 'on' : 'off'}</span></div>
                <div className="flex justify-between py-1.5 border-b border-gray-50"><span className="text-gray-400">Early Stop / Eval</span><span className="font-medium">{values.early_stopping ? 'on' : 'off'}/{values.eval_interval}</span></div>
              </>
            ) : null}
            <div className="flex justify-between py-1.5 mt-2 pt-2 border-t border-gray-200">
              <span className="text-gray-500">Estimated Time</span>
              <span className="font-semibold text-blue-700">{estHours}h</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-50">
              <span className="text-gray-400">Export</span>
              <span className="font-medium">{values.export_format.toUpperCase()}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-gray-500">Output</span>
              <span className="font-mono text-xs truncate ml-2">projects/{values.project || '?'}/</span>
            </div>
          </div>
        </div>
      </div>
          {/* Create Project Modal */}
      {showCreateProject && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center" onClick={() => setShowCreateProject(false)}>
          <div className="bg-white rounded-xl p-6 w-96 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="text-lg font-semibold mb-4">Create Training Project</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
                <input type="text" value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" placeholder="e.g. person_detection" />
              </div>
              <div className="text-xs text-gray-500">
                This will create projects/{newProjectName || '...'}/ with project.yaml, dataset.yaml, and train.yaml.
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <button onClick={() => setShowCreateProject(false)} className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50">Cancel</button>
              <button onClick={handleCreateProject} disabled={!newProjectName.trim() || creatingProject}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
                {creatingProject ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </div>
        </div>
      )}</div>
  );
}


