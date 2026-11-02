import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { FolderOpen, Trash2, Loader2, ExternalLink, HardDrive } from 'lucide-react';
import { api } from '../api/client';

interface WorkspaceInfo {
  name: string;
  path: string;
  has_dataset_yaml: boolean;
  train_count: number;
  val_count: number;
  test_count: number;
  nc: number;
}

export default function WorkspacesPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => { loadWorkspaces(); }, []);

  const loadWorkspaces = async () => {
    setLoading(true);
    try {
      const data = await api.get<{ workspaces: WorkspaceInfo[] }>('/workspaces');
      setWorkspaces(data.workspaces || []);
    } catch (err) {
      console.error('Failed to load workspaces:', err);
    }
    setLoading(false);
  };

  const handleDelete = async (name: string) => {
    if (!window.confirm(t('confirm.deleteWorkspace', { name }))) return;
    setDeleting(name);
    try {
      await api.del('/workspaces/' + name);
      await loadWorkspaces();
    } catch (err) {
      console.error('Failed to delete workspace "' + name + '":', err);
    }
    setDeleting(null);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <FolderOpen className="w-6 h-6 text-amber-600" /> {t('nav.workspaces')}
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            {t('workspaces.description')}
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20"><Loader2 className="w-5 h-5 animate-spin text-gray-400" /></div>
      ) : workspaces.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <FolderOpen className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-sm font-medium text-gray-500 mb-1">{t('workspaces.empty')}</h3>
          <p className="text-xs text-gray-400">{t('workspaces.emptyHint')}</p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {workspaces.map(ws => (
            <div key={ws.name} className="bg-white rounded-lg border border-gray-200 p-4 hover:border-amber-200 transition-all">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-amber-100 text-amber-600 flex items-center justify-center">
                  <HardDrive className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium truncate">{ws.name}</h3>
                  <p className="text-[10px] text-gray-400 truncate mt-0.5 font-mono">{ws.path}</p>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-[10px]">
                <div className="bg-gray-50 rounded px-2 py-1.5">
                  <div className="font-medium text-gray-700">{ws.train_count}</div>
                  <div className="text-gray-400">{t('workspaces.train')}</div>
                </div>
                <div className="bg-gray-50 rounded px-2 py-1.5">
                  <div className="font-medium text-gray-700">{ws.val_count}</div>
                  <div className="text-gray-400">{t('workspaces.val')}</div>
                </div>
                <div className="bg-gray-50 rounded px-2 py-1.5">
                  <div className="font-medium text-gray-700">{ws.nc || '?'}</div>
                  <div className="text-gray-400">{t('workspaces.classes')}</div>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-gray-100 flex items-center justify-between">
                <button onClick={() => {
                    const dsPath = ws.path + '/dataset.yaml';
                    // Navigate to training with workspace info (SPA navigation)
                    navigate('/training?dataset_yaml=' + encodeURIComponent(dsPath) + '&project=' + encodeURIComponent(ws.name));
                  }}
                  className="flex items-center gap-1 px-2.5 py-1 text-[10px] bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
                  <ExternalLink className="w-3 h-3" /> Train
                </button>
                <button onClick={() => handleDelete(ws.name)} disabled={deleting === ws.name}
                  className="flex items-center gap-1 px-2.5 py-1 text-[10px] text-red-500 hover:bg-red-50 rounded transition-colors">
                  {deleting === ws.name ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
