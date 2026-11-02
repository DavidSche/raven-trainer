import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Plus, BookTemplate, Trash2, Loader2, ExternalLink, Database } from 'lucide-react';
import { fetchDatasets } from '../api/datasets';
import { api } from '../api/client';
import type { Dataset } from '../types';

interface RecipeItem {
  name: string;
  description: string;
  sourceCount: number;
  sources: ({name: string; path?: string})[];
  createdAt: string;
}

export default function RecipeListPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [recipes, setRecipes] = useState<RecipeItem[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    loadRecipes();
    fetchDatasets().then(setDatasets).catch((e) => console.error('Failed to load datasets for recipes:', e));
  }, []);

  const loadRecipes = async () => {
    setLoading(true);
    try {
      const data = await api.get<{ recipes: RecipeItem[] }>('/recipes');
      setRecipes(data.recipes || []);
    } catch (e) {
      console.error('Failed to load recipes:', e);
      setRecipes([]);
    }
    setLoading(false);
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const body = {
        name: newName.trim(),
        description: '',
        sources: selectedSources.map(s => ({ name: s, path: null, include: [], exclude: [] })),
      };
      await api.post('/recipes', body);
      await loadRecipes();
      navigate('/recipes/' + newName.trim());
    } catch (e) {
      console.error('Failed to create recipe:', newName.trim(), e);
    }
    setCreating(false);
    setShowCreate(false);
  };

  const toggleSource = (name: string) => {
    setSelectedSources(prev => prev.includes(name) ? prev.filter(s => s !== name) : [...prev, name]);
  };

  const handleDelete = async (name: string) => {
    if (!confirm(t('confirm.deleteRecipe', { name }))) return;
    try { await api.del('/recipes/' + name); await loadRecipes(); } catch (e) { console.error('Failed to delete recipe:', name, e); }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <BookTemplate className="w-6 h-6 text-purple-600" /> {t('nav.recipes') || 'Recipes'}
          </h1>
          <p className="text-sm text-gray-500 mt-1">{t('recipes.standaloneDesc')}</p>
        </div>
        <button onClick={() => { setNewName(''); setSelectedSources([]); setShowCreate(true); }}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700">
          <Plus className="w-4 h-4" /> New Recipe
        </button>
      </div>

      <div className="text-xs bg-purple-50 border border-purple-200 rounded-lg px-4 py-3 text-purple-700 mb-6 leading-relaxed">
        <strong>Three-layer concept:</strong> Asset Dataset (data/datasets/)
        &rarr; <strong>Recipe (data/recipes/, standalone)</strong>
        &rarr; Training Workspace (data/workspace/). A Recipe is a many-to-many bridge.
      </div>

      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setShowCreate(false)}>
          <div className="bg-white rounded-xl border border-gray-200 p-6 w-[500px] max-w-[90vw] shadow-2xl" onClick={e => e.stopPropagation()}>
            <h2 className="text-sm font-semibold mb-4">New Recipe</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Recipe Name</label>
                <input type="text" value={newName} onChange={e => setNewName(e.target.value)}
                  placeholder="e.g. person_detection_v3"
                  className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:border-purple-400 outline-none" />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-1">Source Asset Datasets</label>
                <div className="border border-gray-200 rounded-lg max-h-40 overflow-y-auto space-y-1 p-1">
                  {datasets.length === 0
                    ? <div className="text-xs text-gray-400 p-3 text-center">No asset datasets found</div>
                    : datasets.map(ds => (
                        <label key={ds.name} className="flex items-center gap-2 px-2 py-1.5 text-xs rounded hover:bg-gray-50 cursor-pointer">
                          <input type="checkbox" checked={selectedSources.includes(ds.name)}
                            onChange={() => toggleSource(ds.name)} className="rounded border-gray-300 text-purple-600" />
                          <Database className="w-3.5 h-3.5 text-gray-400" />
                          <span className="flex-1">{ds.display_name || ds.name}</span>
                          <span className="text-gray-400">{ds.total_images} imgs</span>
                        </label>
                      ))}
                </div>
              </div>
              <div className="flex gap-2 pt-2">
                <button onClick={handleCreate} disabled={creating || !newName.trim() || selectedSources.length === 0}
                  className="flex items-center gap-1 px-4 py-1.5 bg-purple-600 text-white rounded-lg text-xs hover:bg-purple-700 disabled:opacity-50">
                  {creating ? <Loader2 className="w-3 h-3 animate-spin" /> : <Plus className="w-3 h-3" />}
                  Create Recipe
                </button>
                <button onClick={() => setShowCreate(false)} className="px-4 py-1.5 border border-gray-200 rounded-lg text-xs text-gray-500 hover:bg-gray-50">{t('common.cancel')}</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20"><Loader2 className="w-5 h-5 animate-spin text-gray-400" /></div>
      ) : recipes.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <BookTemplate className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-sm font-medium text-gray-500 mb-1">{t('recipes.noRecipes')}</h3>
          <p className="text-xs text-gray-400">{t('recipes.noRecipesHint')}</p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {recipes.map(recipe => (
            <div key={recipe.name} className="bg-white rounded-lg border border-gray-200 p-4 hover:border-purple-200 hover:shadow-sm transition-all cursor-pointer"
              onClick={() => navigate('/recipes/' + recipe.name)}>
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-purple-100 text-purple-600 flex items-center justify-center">
                  <BookTemplate className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium truncate">{recipe.name}</h3>
                  <p className="text-[10px] text-gray-400 mt-0.5">{recipe.description || 'No description'}</p>
                </div>
              </div>
              <div className="mt-3 space-y-1.5">
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Database className="w-3 h-3" /><span>{recipe.sourceCount || (recipe.sources?.length || 0)} source(s)</span>
                </div>
                {recipe.sources && recipe.sources.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {recipe.sources.slice(0, 3).map(s => (<span key={s.name} className="text-[9px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">{s.name}</span>))}
                    {recipe.sources.length > 3 && <span className="text-[9px] text-gray-400">+{recipe.sources.length - 3}</span>}
                  </div>
                )}
              </div>
              <div className="mt-3 pt-3 border-t border-gray-100 flex items-center justify-between">
                <span className="text-[10px] text-gray-400">{recipe.createdAt ? new Date(recipe.createdAt).toLocaleDateString() : ''}</span>
                <div className="flex gap-1">
                  <button onClick={e => { e.stopPropagation(); navigate('/recipes/' + recipe.name); }}
                    className="px-2 py-1 text-[10px] text-purple-600 hover:bg-purple-50 rounded transition-colors">
                    <ExternalLink className="w-3 h-3 inline mr-0.5" />Open
                  </button>
                  <button onClick={e => { e.stopPropagation(); handleDelete(recipe.name); }}
                    className="px-2 py-1 text-[10px] text-red-500 hover:bg-red-50 rounded transition-colors">
                    <Trash2 className="w-3 h-3 inline mr-0.5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
