import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { fetchModels, fetchModelStats, fetchModelDetail } from '../api/models';
import type { Model } from '../types';
import { Box, Download, X, BarChart3, HardDrive } from 'lucide-react';

export default function ModelsPage() {
  const { t } = useTranslation();
  const [models, setModels] = useState<Model[]>([]);
  const [stats, setStats] = useState<Record<string, unknown>>({});
  const [selectedModel, setSelectedModel] = useState<any>(null);

  useEffect(() => {
    fetchModels().then(setModels).catch((e) => console.error('Failed to load models:', e));
    fetchModelStats().then(setStats).catch((e) => console.error('Failed to load model stats:', e));
  }, []);

  const handleShowDetail = async (modelId: string) => {
    try {
      const detail = await fetchModelDetail(modelId);
      setSelectedModel(detail);
    } catch (e) {
      console.error('Failed to fetch model detail for', modelId, e);
      setSelectedModel({ model: models.find((m) => m.id === modelId) });
    }
  };

  return (
    <div className="p-6 max-w-6xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">{t('models.title')}</h1>
      </div>

      {Object.keys(stats).length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">{stats.total as unknown as number ?? 0}</span>
              <Box className="w-5 h-5 text-blue-500" />
            </div>
            <div className="text-sm text-gray-500 mt-1">{t('models.totalModels')}</div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">{stats.total_size_mb as unknown as number ?? 0} MB</span>
              <HardDrive className="w-5 h-5 text-green-500" />
            </div>
            <div className="text-sm text-gray-500 mt-1">{t('models.totalSize')}</div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">{Object.keys(stats.by_project || {}).length}</span>
              <BarChart3 className="w-5 h-5 text-purple-500" />
            </div>
            <div className="text-sm text-gray-500 mt-1">{t('models.projects')}</div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">{Object.keys(stats.by_format || {}).length}</span>
              <Download className="w-5 h-5 text-orange-500" />
            </div>
            <div className="text-sm text-gray-500 mt-1">{t('models.formats')}</div>
          </div>
        </div>
      )}

      {models.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-400">
          <Box className="w-8 h-8 mx-auto mb-2" />
          {t('models.noModels')}
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="text-left px-4 py-3 font-medium">{t('models.tableId')}</th>
                <th className="text-left px-4 py-3 font-medium">{t('models.tableProject')}</th>
                <th className="text-left px-4 py-3 font-medium">{t('models.tableFormat')}</th>
                <th className="text-left px-4 py-3 font-medium">{t('models.tableType')}</th>
                <th className="text-right px-4 py-3 font-medium">{t('models.tableCreated')}</th>
                <th className="text-right px-4 py-3 font-medium">{t('models.tableActions')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {models.map((m) => (
                <tr key={m.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-mono text-xs">{m.id}</td>
                  <td className="px-4 py-3">{m.project}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      m.format === 'pytorch' ? 'bg-blue-100 text-blue-700' :
                      m.format === 'onnx' ? 'bg-green-100 text-green-700' :
                      'bg-gray-100 text-gray-600'
                    }`}>
                      {m.format}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-500">{m.type}</td>
                  <td className="px-4 py-3 text-right text-gray-400">{m.created_at.slice(0, 10)}</td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => handleShowDetail(m.id)}
                      className="text-blue-600 hover:underline text-xs mr-3"
                    >
                      {t('models.details')}
                    </button>
                    <a
                      href={'/api/v1/models/' + m.id + '/download'}
                      className="text-green-600 hover:underline text-xs"
                      download
                    >
                      <Download className="w-3 h-3 inline" />
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedModel && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedModel(null)}>
          <div className="bg-white rounded-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-semibold">
                {selectedModel.model?.id || t('models.modelDetails')}
              </h2>
              <button onClick={() => setSelectedModel(null)} className="text-gray-400 hover:text-gray-600">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">{t('models.modelInfo')}</h3>
                <pre className="bg-gray-50 rounded-lg p-3 text-xs font-mono">
                  {JSON.stringify(selectedModel.model, null, 2)}
                </pre>
              </div>
              {selectedModel.file_info && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500 mb-2">{t('models.fileInfo')}</h3>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-xs text-gray-500">{t('models.size')}</div>
                      <div className="font-medium">{selectedModel.file_info?.size_mb ?? 0} MB</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-xs text-gray-500">{t('models.formatLabel')}</div>
                      <div className="font-medium">{selectedModel.file_info?.format || '-'}</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-xs text-gray-500">{t('models.bytes')}</div>
                      <div className="font-medium text-xs">{selectedModel.file_info?.size_bytes?.toLocaleString() || 0}</div>
                    </div>
                  </div>
                </div>
              )}
              {selectedModel.experiment && Object.keys(selectedModel.experiment).length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500 mb-2">{t('models.experiment')}</h3>
                  <pre className="bg-gray-50 rounded-lg p-3 text-xs font-mono">
                    {JSON.stringify(selectedModel.experiment, null, 2)}
                  </pre>
                </div>
              )}
              <div className="flex gap-2 pt-2">
                <a
                  href={'/api/v1/models/' + (selectedModel.model?.id || '') + '/download'}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
                  download
                >
                  <Download className="w-4 h-4" /> {t('models.download')}
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
