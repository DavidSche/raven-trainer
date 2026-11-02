import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FileOutput, Plus, Trash2, Save, Play, Loader2, Check,
  AlertTriangle, Settings, SplitSquareHorizontal, Filter,
  Camera, Hash, Image, Download, ExternalLink,
} from 'lucide-react';
import {
  fetchRecipes, fetchRecipe, saveRecipe, createDefaultRecipe,
  buildFromRecipe, fetchBuildJobs, fetchVersions, fetchDatasets,
} from '../../api/datasets';
import { api } from '../../api/client';
import type {
  DatasetRecipe, RecipeSource,
  RecipeSplit, WorkspaceBuildResult, BuildJob,
} from '../../types';

interface AvailableSource {
  name: string;
  display_name: string;
  task: string;
  current_version: string;
  versions: string[];
  total_images: number;
}

interface Props {
  recipeName: string;
}

const DEFAULT_RECIPE: DatasetRecipe = {
  name: '',
  version: '1.0',
  description: '',
  sources: [],
  include: [], exclude: [],
  sampling: { strategy: 'motion,scene_change', motion_threshold: 0.4, scene_threshold: 0.3 },
  split: { strategy: 'by_video', train: 0.7, val: 0.2, test: 0.1 },
  quality: { min_blur_score: 100, dedup_method: 'dhash', hash_distance: 5 },
  balance: { max_per_camera: 5000, time_balance: true },
  build_target: 'yolo',
};

export default function RecipeBuilderTab({ recipeName }: Props) {
  const { t } = useTranslation();
  const [recipeNames, setRecipeNames] = useState<string[]>([]);
  const [activeRecipe, setActiveRecipe] = useState('default');
  const [recipe, setRecipe] = useState<DatasetRecipe>(DEFAULT_RECIPE);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [building, setBuilding] = useState(false);
  const [buildResult, setBuildResult] = useState<WorkspaceBuildResult | null>(null);
  const [buildLog, setBuildLog] = useState<string[]>([]);
  const [forceRebuild, setForceRebuild] = useState(false);
  const [buildJobs, setBuildJobs] = useState<BuildJob[]>([]);
  const [error, setError] = useState('');
  const [modified, setModified] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [availableSources, setAvailableSources] = useState<AvailableSource[]>([]);

  useEffect(() => { loadRecipes(); loadBuildJobs(); loadAvailableSources(); }, [recipeName]);

  const loadRecipes = async () => {
    setLoading(true);
    try {
      const data = await api.get<{ description?: string; sources?: any[] }>('/recipes/' + encodeURIComponent(recipeName));
      // API returns RecipeResponse with sources as array of {name, path}
      setRecipeNames([recipeName]);
      setRecipe(prev => ({
        ...prev,
        description: data.description || '',
        sources: (data.sources || []).map((s: any) => ({
          name: typeof s === 'string' ? s : (s.name || ''),
          path: typeof s === 'object' ? (s.path || null) : null,
          include: [],
          exclude: [],
        })),
      }));
    } catch (e) {
      console.error('Failed to load recipe for', recipeName, e);
    }
    setLoading(false);
  };

  const loadBuildJobs = async () => {
    try { setBuildJobs(await fetchBuildJobs(recipeName)); } catch (e) { console.error('Failed to load build jobs for', recipeName, e); }
  };

  const loadAvailableSources = async () => {
    try {
      const assets = await fetchDatasets();
      const sources: AvailableSource[] = [];
      for (const a of assets) {
        try {
          const v = await fetchVersions(a.name);
          sources.push({
            name: a.name, display_name: a.display_name || a.name,
            task: a.task, current_version: a.current_version || 'v1',
            versions: v.map(x => x.version), total_images: a.total_images,
          });
        } catch (e) {
          console.error('Failed to fetch versions for', a.name, e);
          sources.push({
            name: a.name, display_name: a.display_name || a.name,
            task: a.task, current_version: 'v1',
            versions: ['v1'], total_images: a.total_images,
          });
        }
      }
      setAvailableSources(sources);
    } catch (e) {
      console.error('Failed to load available sources for recipe builder:', e);
    }
  };

  const switchRecipe = async (name: string) => {
    if (modified && !confirm(t('recipes.unsavedConfirm'))) return;
    setActiveRecipe(name);
    try { setRecipe(await fetchRecipe(recipeName, name)); setModified(false); } catch (e) { console.error('Failed to switch recipe to', name, e); }
  };

  const handleSave = async () => {
    setSaving(true); setError('');
    try {
      await api.put('/recipes/' + encodeURIComponent(recipeName), {
        name: recipeName,
        description: recipe.description,
        sources: recipe.sources,
      });
      setModified(false);
    } catch (e: any) { setError(e.message || 'Save failed'); }
    setSaving(false);
  };

  const handleCreateDefault = async () => {
    try { await createDefaultRecipe(recipeName); await loadRecipes(); await switchRecipe('default'); }
    catch (e: any) { setError(e.message); }
  };

  const handleBuild = async () => {
    setBuilding(true); setBuildResult(null); setBuildLog([]); setError('');
    try {
      const path = '/recipes/' + encodeURIComponent(recipeName) + '/build?force=' + (forceRebuild ? 'true' : 'false');
      const result = await api.post<WorkspaceBuildResult>(path);
      setBuildResult(result);
      setBuildLog(p => [...p, 'Complete: ' + result.train_count + ' train, ' + result.val_count + ' val, ' + result.test_count + ' test (' + result.duration_sec + 's)']);
    } catch (e: any) {
      // 409 Conflict: workspace already exists — preserve the dedicated user-facing hint.
      if (e.message && e.message.includes('API 409')) {
        setError('Workspace already exists. Enable "Force rebuild" to overwrite.');
        setBuildLog(p => [...p, 'WARN: Workspace exists, use Force rebuild to overwrite']);
      } else {
        setError(e.message || 'Build failed');
        setBuildLog(p => [...p, 'ERROR: ' + (e.message || 'Build failed')]);
      }
    }
    setBuilding(false);
  };

  const addSource = () => {
    const firstAvailable = availableSources[0]?.name || '';
    const defaultName = firstAvailable ? `${firstAvailable}@${availableSources[0].current_version}` : 'new_source';
    setRecipe(p => ({ ...p, sources: [...p.sources, { name: defaultName, include: [], exclude: [] }] }));
    setModified(true);
  };

  const removeSource = (idx: number) => {
    setRecipe(p => ({ ...p, sources: p.sources.filter((_, i) => i !== idx) }));
    setModified(true);
  };

  const updateSourceDataset = (idx: number, dsName: string) => {
    const ds = availableSources.find(s => s.name === dsName);
    const version = ds?.current_version || 'v1';
    setRecipe(p => {
      const sources = [...p.sources];
      sources[idx] = { ...sources[idx], name: `${dsName}@${version}` };
      return { ...p, sources };
    });
    setModified(true);
  };

  const updateSourceVersion = (idx: number, ver: string) => {
    setRecipe(p => {
      const sources = [...p.sources];
      const base = sources[idx].name.split('@')[0];
      sources[idx] = { ...sources[idx], name: `${base}@${ver}` };
      return { ...p, sources };
    });
    setModified(true);
  };

  const updateSplit = (field: keyof RecipeSplit, value: number | string) => {
    setRecipe(p => ({ ...p, split: { ...p.split, [field]: value } }));
    setModified(true);
  };

  const updateList = (field: 'include' | 'exclude', value: string) => {
    setRecipe(p => ({ ...p, [field]: value.split('\n').map(s => s.trim()).filter(Boolean) }));
    setModified(true);
  };

  const parseSource = (name: string) => {
    const parts = name.split('@');
    return { dataset: parts[0], version: parts[1] || '' };
  };

  const splitTotal = recipe.split.train + recipe.split.val + recipe.split.test;
  if (loading) return <div className="p-8 text-center text-gray-400"><Loader2 className="w-5 h-5 animate-spin mx-auto" /></div>;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <select value={activeRecipe} onChange={e => switchRecipe(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm bg-white">
            {recipeNames.length > 0 ? recipeNames.map(n => <option key={n} value={n}>{n}</option>)
              : <option value="">No recipes</option>}
          </select>
          {recipeNames.length === 0 && (
            <button onClick={handleCreateDefault}
              className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs hover:bg-blue-700">
              <Plus className="w-3.5 h-3.5" /> Create Default Recipe
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          {modified && <span className="text-xs text-amber-600 font-medium">Unsaved changes</span>}
          <button onClick={handleSave} disabled={saving || !modified}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Recipe
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      <div className="text-xs bg-blue-50 border border-blue-200 rounded-lg px-4 py-2.5 text-blue-700 leading-relaxed">
        <strong>三层概念</strong>：
        资产数据集（Asset Dataset, <code className="bg-blue-100 px-1 rounded">data/datasets/</code>）
        → Recipe 构建配方（<code className="bg-blue-100 px-1 rounded">data/recipes/</code>）
        → 训练工作区（Training Workspace, <code className="bg-blue-100 px-1 rounded">data/workspace/</code>）。
        Recipe 是数据资产和训练工作区之间的多对多桥梁。
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <div className="xl:col-span-2 space-y-4">
          {/* Sources — SELECTION based */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Camera className="w-4 h-4 text-blue-600" /> Sources（来源资产数据集）
              </h3>
              <button onClick={addSource}
                className="flex items-center gap-1 px-2.5 py-1 text-xs bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100">
                <Plus className="w-3 h-3" /> Add Source
              </button>
            </div>
            <div className="text-[10px] text-gray-400 mb-3">
              从已注册的资产数据集选择来源。一个 Recipe 可以引用多个数据集，实现多对多映射。
            </div>
            {recipe.sources.length === 0 ? (
              <div className="text-xs text-gray-400 py-4 text-center">
                No sources configured. Click "Add Source" to select from available asset datasets.
              </div>
            ) : (
              <div className="space-y-2">
                {recipe.sources.map((src, i) => {
                  const { dataset: dsName, version: verName } = parseSource(src.name);
                  const ds = availableSources.find(s => s.name === dsName);
                  return (
                    <div key={i} className="flex items-center gap-2 bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">
                      <select value={dsName}
                        onChange={e => updateSourceDataset(i, e.target.value)}
                        className="flex-1 bg-white border border-gray-200 rounded px-2 py-1.5 text-xs font-medium focus:border-blue-400 focus:outline-none">
                        <option value="">-- Select Asset Dataset --</option>
                        {availableSources.map(s => (
                          <option key={s.name} value={s.name}>
                            {s.display_name} ({s.task}, {s.total_images} imgs)
                          </option>
                        ))}
                      </select>
                      {ds && ds.versions.length > 0 && (
                        <select value={verName}
                          onChange={e => updateSourceVersion(i, e.target.value)}
                          className="bg-white border border-gray-200 rounded px-2 py-1.5 text-xs font-mono focus:border-blue-400 focus:outline-none">
                          {ds.versions.map(v => <option key={v} value={v}>{v}</option>)}
                        </select>
                      )}
                      <span className="text-[10px] text-gray-400 w-12 text-right">{ds?.total_images ?? '?'} img</span>
                      <button onClick={() => removeSource(i)}
                        className="p-1 text-red-400 hover:text-red-600 hover:bg-red-50 rounded">
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Split */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <SplitSquareHorizontal className="w-4 h-4 text-blue-600" /> Split Strategy
            </h3>
            <div className="grid grid-cols-4 gap-3">
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Strategy</label>
                <select value={recipe.split.strategy} onChange={e => updateSplit('strategy', e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs">
                  <option value="by_video">by_video - per camera random (default)</option>
                  <option value="random">random - simple shuffle</option>
                  <option value="temporal">temporal - time ordered</option>
              <option value="stratified">stratified - class balanced</option>
              <option value="group">group - anti leakage</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Train</label>
                <input type="number" value={recipe.split.train}
                  onChange={e => updateSplit('train', parseFloat(e.target.value) || 0)}
                  min={0} max={1} step={0.05}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs" />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Val</label>
                <input type="number" value={recipe.split.val}
                  onChange={e => updateSplit('val', parseFloat(e.target.value) || 0)}
                  min={0} max={1} step={0.05}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs" />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Test</label>
                <input type="number" value={recipe.split.test}
                  onChange={e => updateSplit('test', parseFloat(e.target.value) || 0)}
                  min={0} max={1} step={0.05}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs" />
              </div>
            </div>
            {Math.abs(splitTotal - 1.0) > 0.01 && (
              <div className="mt-2 text-xs text-amber-600">Split ratios must sum to 1.0 (currently {splitTotal.toFixed(2)})</div>
            )}
          </div>

          {/* Filters */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Filter className="w-4 h-4 text-blue-600" /> Filters & Quality
            </h3>
            <div className="grid grid-cols-3 gap-3 mb-3">
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Min Blur Score</label>
                <input type="number" value={recipe.quality.min_blur_score || 100}
                  onChange={e => setRecipe(p => ({ ...p, quality: { ...p.quality, min_blur_score: parseInt(e.target.value) || 0 } }))}
                  min={0} max={500} className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs" />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Dedup Method</label>
                <select value={recipe.quality.dedup_method || 'dhash'}
                  onChange={e => setRecipe(p => ({ ...p, quality: { ...p.quality, dedup_method: e.target.value } }))}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs">
                  <option value="dhash">dHash</option>
                  <option value="phash">pHash</option>
                  <option value="ssim">SSIM</option>
                  <option value="none">None</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Hash Distance</label>
                <input type="number" value={recipe.quality.hash_distance || 5}
                  onChange={e => setRecipe(p => ({ ...p, quality: { ...p.quality, hash_distance: parseInt(e.target.value) || 5 } }))}
                  min={1} max={64} className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Max per Camera</label>
                <input type="number" value={recipe.balance.max_per_camera || 5000}
                  onChange={e => setRecipe(p => ({ ...p, balance: { ...p.balance, max_per_camera: parseInt(e.target.value) || 0 } }))}
                  min={0} step={100} className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs" />
              </div>
              <div className="flex items-end pb-1.5">
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                  <input type="checkbox" checked={recipe.balance.time_balance !== false}
                    onChange={e => setRecipe(p => ({ ...p, balance: { ...p.balance, time_balance: e.target.checked } }))}
                    className="rounded border-gray-300 text-blue-600" />
                  Time balance
                </label>
              </div>
            </div>
          </div>

          {/* Filename patterns */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Hash className="w-4 h-4 text-blue-600" /> Filename Patterns
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Include</label>
                <textarea value={(recipe.include || []).join('\n')}
                  onChange={e => updateList('include', e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs font-mono h-16 resize-none"
                  placeholder="daytime&#10;camera01" />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Exclude</label>
                <textarea value={(recipe.exclude || []).join('\n')}
                  onChange={e => updateList('exclude', e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-xs font-mono h-16 resize-none"
                  placeholder="blurry&#10;night" />
              </div>
            </div>
          </div>
        </div>

        {/* Right: Build panel */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Settings className="w-4 h-4 text-purple-600" /> Build
            </h3>
            <button onClick={handleBuild} disabled={building || recipe.sources.length === 0}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 transition-colors">
              {building ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {building ? 'Building...' : 'Build Workspace'}
            </button>
            <label className="flex items-center gap-1.5 mt-2 text-[10px] text-gray-500 cursor-pointer select-none justify-center">
              <input type="checkbox" checked={forceRebuild} onChange={e => setForceRebuild(e.target.checked)}
                className="rounded border-gray-300 text-purple-600 w-3 h-3" />
              Force rebuild (overwrite existing workspace)
            </label>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
            <button onClick={() => setShowLog(!showLog)}
              className="w-full flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
              <span className="text-xs text-gray-400 font-mono">Build Log</span>
              <span className="text-[10px] text-gray-500">{buildJobs.length > 0 ? buildJobs[0].status : 'ready'}</span>
            </button>
            {showLog && (
              <div className="h-48 overflow-y-auto p-3 space-y-1">
                {buildLog.length === 0
                  ? <div className="text-gray-500 text-xs font-mono">Ready to build...</div>
                  : buildLog.map((line, i) => (
                      <div key={i} className={`text-xs font-mono leading-relaxed ${line.includes("ERROR") ? "text-red-400" : line.includes("Complete") ? "text-green-400" : "text-green-400/80"}`}>{line}</div>
                    ))}
                {building && <Loader2 className="w-3 h-3 animate-spin text-green-400 mt-1" />}
              </div>
            )}
          </div>

          {buildResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 space-y-3">
              <h3 className="text-sm font-semibold text-green-800 flex items-center gap-2">
                <Check className="w-4 h-4" /> Build Complete
              </h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-white rounded px-2 py-1.5 border border-green-100">
                  <div className="text-green-700 font-medium">{buildResult.train_count}</div>
                  <div className="text-green-500">Train</div>
                </div>
                <div className="bg-white rounded px-2 py-1.5 border border-green-100">
                  <div className="text-green-700 font-medium">{buildResult.val_count}</div>
                  <div className="text-green-500">Val</div>
                </div>
                <div className="bg-white rounded px-2 py-1.5 border border-green-100">
                  <div className="text-green-700 font-medium">{buildResult.test_count}</div>
                  <div className="text-green-500">Test</div>
                </div>
                <div className="bg-white rounded px-2 py-1.5 border border-green-100">
                  <div className="text-green-700 font-medium">{buildResult.class_count}</div>
                  <div className="text-green-500">Classes</div>
                </div>
              </div>
              <div className="text-[10px] text-green-600 space-y-1">
                {buildResult.skipped_duplicate > 0 && <div className="flex items-center gap-1"><Image className="w-3 h-3" /> {buildResult.skipped_duplicate} duplicates removed</div>}
                {buildResult.skipped_blurry > 0 && <div className="flex items-center gap-1"><Image className="w-3 h-3" /> {buildResult.skipped_blurry} blurry removed</div>}
                <div className="flex items-center gap-1"><Download className="w-3 h-3" /> <code className="bg-green-100 px-1 rounded">{buildResult.dataset_yaml}</code></div>
              </div>
              {buildResult.output_dir && (
                <a href={`/training?dataset_yaml=${encodeURIComponent(buildResult.dataset_yaml)}`}
                  className="flex items-center justify-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs hover:bg-blue-700 w-full">
                  <ExternalLink className="w-3 h-3" /> Start Training
                </a>
              )}
            </div>
          )}

          {buildJobs.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 px-4 py-3">
              <h4 className="text-xs font-medium text-gray-500 mb-2">Recent Builds</h4>
              <div className="space-y-1">
                {buildJobs.slice(0, 5).map(job => (
                  <div key={job.job_id} className="flex items-center gap-2 text-xs">
                    <span className={`w-1.5 h-1.5 rounded-full ${job.status === "completed" ? "bg-green-500" : job.status === "failed" ? "bg-red-500" : job.status === "running" ? "bg-blue-500 animate-pulse" : "bg-gray-400"}`} />
                    <span className="text-gray-600 font-mono">{job.job_id}</span>
                    <span className="text-gray-400">{job.status}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
