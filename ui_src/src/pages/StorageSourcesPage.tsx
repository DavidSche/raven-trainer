import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import {
  FolderOpen, HardDrive, Database, ChevronRight,
  Film, Image as ImageIcon, Calendar,
} from 'lucide-react';
import { api } from '../api/client';

interface StorageSource {
  name: string;
  display_name: string;
  media_type: string;
  file_count: number;
  total_size: number;
  total_size_display: string;
  datasets: { dataset_name: string; version: string }[];
  preview: Record<string, unknown>;
}

interface MediaTypeGroup {
  media_type: string;
  label: string;
  path: string;
  source_count: number;
  file_count: number;
  total_size: number;
  total_size_display: string;
  sources: StorageSource[];
}

interface StorageResponse {
  root_path: string;
  media_types: MediaTypeGroup[];
  summary: {
    total_images: number;
    total_videos: number;
    total_size: number;
    total_size_display: string;
    total_sources: number;
  };
}

function formatDate(ts: number) {
  const d = new Date(ts * 1000);
  return d.toISOString().replace('T', ' ').slice(0, 19);
}

const TYPE_ICONS: Record<string, React.ReactNode> = {
  images: <ImageIcon className="w-5 h-5 text-green-500" />,
  videos: <Film className="w-5 h-5 text-purple-500" />,
};

export default function StorageSourcesPage() {
  const { t } = useTranslation();
  const [data, setData] = useState<StorageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get<StorageResponse>('/storage')
      .then((d: StorageResponse) => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) {
    return <div className="p-6 flex items-center justify-center h-64">
      <div className="text-gray-400 flex items-center gap-2">
        <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        {t('common.loading')}
      </div>
    </div>;
  }

  if (error) {
    return <div className="p-6 max-w-6xl">
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
        <HardDrive className="w-10 h-10 mx-auto mb-3 text-red-300" />
        <p className="text-red-700 font-medium">{t('storage.loadError')}</p>
        <p className="text-red-500 text-sm mt-1">{error}</p>
        <button onClick={() => window.location.reload()}
          className="mt-3 px-4 py-1.5 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700">{t('common.retry')}</button>
      </div>
    </div>;
  }

  if (!data) return null;

  return (
    <div className="p-6 max-w-6xl">
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <HardDrive className="w-5 h-5 text-gray-400" />
        <h1 className="text-lg font-semibold">Storage</h1>
        <span className="text-sm text-gray-400 ml-2">{data.root_path}</span>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-2xl font-bold text-gray-800">{data.media_types.length}</div>
          <div className="text-xs text-gray-500 mt-1">Media Types</div>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-2xl font-bold text-gray-800">{data.summary.total_sources}</div>
          <div className="text-xs text-gray-500 mt-1">Sources</div>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-2xl font-bold text-gray-800">{data.summary.total_images.toLocaleString()}</div>
          <div className="text-xs text-gray-500 mt-1">Images</div>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-2xl font-bold text-gray-800">{data.summary.total_videos}</div>
          <div className="text-xs text-gray-500 mt-1">Videos</div>
        </div>
      </div>

      {/* Media type sections */}
      {data.media_types.map((mt) => (
        <div key={mt.media_type} className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            {TYPE_ICONS[mt.media_type]}
            <h2 className="font-medium text-sm text-gray-700">{mt.label}</h2>
            <span className="text-xs text-gray-400">
              {mt.source_count} sources, {mt.file_count.toLocaleString()} files, {mt.total_size_display}
            </span>
            <code className="text-[10px] text-gray-300 ml-auto">{mt.path}</code>
          </div>

          {mt.sources.length === 0 ? (
            <div className="bg-gray-50 rounded-lg border border-gray-200 p-6 text-center text-sm text-gray-400">
              No {mt.media_type} sources. Add directories to <code className="bg-gray-100 px-1 rounded">{mt.path}/</code>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {mt.sources.map((src) => (
                <Link
                  key={src.name}
                  to={`/storage/${mt.media_type}/${src.name}`}
                  className="bg-white rounded-lg border border-gray-200 p-4 hover:border-blue-300 hover:shadow-sm transition-all block"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                        mt.media_type === 'videos' ? 'bg-purple-50' : 'bg-green-50'
                      }`}>
                        {mt.media_type === 'videos'
                          ? <Film className="w-5 h-5 text-purple-500" />
                          : <ImageIcon className="w-5 h-5 text-green-500" />
                        }
                      </div>
                      <div className="min-w-0">
                        <h3 className="font-medium text-sm truncate">{src.display_name}</h3>
                        <code className="text-xs text-gray-400">{src.name}</code>
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-gray-300 shrink-0" />
                  </div>

                  <div className="mt-3 flex items-center gap-3 text-xs text-gray-500">
                    <span className="font-medium text-gray-700">{src.file_count.toLocaleString()}</span>
                    <span>{mt.media_type === 'videos' ? 'videos' : 'images'}</span>
                    <span className="text-gray-300">|</span>
                    <span>{src.total_size_display}</span>
                  </div>

                  {src.datasets.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {src.datasets.map((ds) => (
                        <Link key={ds.dataset_name}
                          to={`/datasets/${ds.dataset_name}`}
                          onClick={(e) => e.stopPropagation()}
                          className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded text-xs text-gray-600 hover:bg-blue-50 hover:text-blue-600 border border-gray-100"
                        >
                          <Database className="w-3 h-3" />
                          {ds.dataset_name}
                        </Link>
                      ))}
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* System closed-loop diagram */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100 p-4 mt-6">
        <h3 className="text-xs font-semibold text-blue-700 uppercase tracking-wider mb-2">Closed Loop</h3>
        <div className="flex items-center gap-2 text-xs text-blue-600 flex-wrap">
          <span className="px-2 py-1 bg-white rounded border border-blue-200">Storage</span>
          <span className="text-blue-300">→</span>
          <span className="px-2 py-1 bg-white rounded border border-blue-200">{t('storage.flowExtract')}</span>
          <span className="text-blue-300">→</span>
          <span className="px-2 py-1 bg-white rounded border border-blue-200">{t('storage.flowDataset')}</span>
          <span className="text-blue-300">→</span>
          <span className="px-2 py-1 bg-white rounded border border-blue-200">{t('storage.flowTrain')}</span>
          <span className="text-blue-300">→</span>
          <span className="px-2 py-1 bg-white rounded border border-blue-200">{t('storage.flowModel')}</span>
          <span className="text-blue-300">→</span>
          <span className="px-2 py-1 bg-white rounded border border-blue-200">{t('storage.flowDeploy')}</span>
          <span className="text-blue-300">→</span>
          <span className="px-2 py-1 bg-white rounded border border-blue-200">{t('storage.flowFeedback')}</span>
          <span className="text-blue-300">↻</span>
        </div>
      </div>
    </div>
  );
}
