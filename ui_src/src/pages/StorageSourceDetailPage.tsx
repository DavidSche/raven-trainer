import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, Link } from 'react-router-dom';
import {
  ChevronRight, Image as ImageIcon, Info, HardDrive,
  Database, Calendar, Film, Play, Settings, Loader2,
} from 'lucide-react';
import { api } from '../api/client';

const API_BASE = '/api/v1';

interface MediaFile {
  filename: string;
  size_bytes: number;
  size_display: string;
  width: number;
  height: number;
  duration_seconds?: number;
  fps?: number;
  codec?: string;
  format_name?: string;
}

interface SourceDetail {
  name: string;
  display_name: string;
  file_count: number;
  total_size: number;
  total_size_display: string;
  datasets: { dataset_name: string; version: string }[];
  previews: MediaFile[];
}

interface VideoInfo {
  source: string;
  display_name: string;
  file_count: number;
  total_duration_seconds: number;
  total_duration_display: string;
  total_size: number;
  total_size_display: string;
  datasets: { dataset_name: string; version: string }[];
  files: MediaFile[];
}

function formatDuration(seconds: number) {
  if (seconds <= 0) return '0:00';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function StorageSourceDetailPage() {
  const { t } = useTranslation();
  const { media_type, source } = useParams<{ media_type: string; source: string }>();
  const [detail, setDetail] = useState<SourceDetail | VideoInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [showInfo, setShowInfo] = useState(false);
  const [previewFile, setPreviewFile] = useState<string | null>(null);

  // Extraction state (video only)
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<any>(null);
  const [extractError, setExtractError] = useState<string | null>(null);
  const [samplingMethod, setSamplingMethod] = useState('interval');
  const [intervalSec, setIntervalSec] = useState(5);
  const [maxFrames, setMaxFrames] = useState(500);
  const [datasetName, setDatasetName] = useState('');
  const [datasetMode, setDatasetMode] = useState<'new' | 'existing'>('new');
  const [existingDatasets, setExistingDatasets] = useState<{ name: string; version_count: number; total_images: number }[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);

  const isVideo = media_type === 'videos';

  useEffect(() => {
    if (!media_type || !source) return;
    setLoading(true);
    const path = isVideo
      ? `/storage/videos/${source}/info`
      : `/storage/${media_type}/${source}/browse?offset=0&limit=50`;
    api.get<any>(path)
      .then((d: any) => {
        if (isVideo) {
          setDetail(d as VideoInfo);
        } else {
          setDetail({
            name: source,
            display_name: (source || '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            file_count: d.total || 0,
            total_size: 0,
            total_size_display: '',
            datasets: [],
            previews: d.files || [],
          } as SourceDetail);
        }
        setLoading(false);
      })
      .catch((e) => { console.error('Failed to load source detail:', e); setLoading(false); });
  }, [media_type, source]);

  // Fetch existing datasets for the selector
  useEffect(() => {
    setDatasetsLoading(true);
    api.get<any[]>('/datasets')
      .then(data => { setExistingDatasets(data || []); setDatasetsLoading(false); })
      .catch((e) => { console.error('Failed to load datasets for selector:', e); setDatasetsLoading(false); });
  }, []);

  const handleExtract = async () => {
    if (!source) return;
    setExtracting(true);
    setExtractResult(null);
    setExtractError(null);
    try {
      const result = await api.post<any>(`/storage/videos/${source}/extract`, {
        dataset_name: datasetName || undefined,
        sampling: samplingMethod,
        interval_seconds: intervalSec,
        max_frames: maxFrames,
      });
      setExtractResult(result);
    } catch (e: any) {
      setExtractError(e.message);
    }
    setExtracting(false);
  };

  if (loading) {
    return <div className="p-6 flex items-center justify-center h-64">
      <div className="text-gray-400 flex items-center gap-2">
        <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        {t('common.loading')}
      </div>
    </div>;
  }

  if (!detail) {
    return <div className="p-6 text-gray-400">{t('storage.sourceNotFound')}</div>;
  }

  return (
    <div className="p-6 max-w-6xl">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
        <Link to="/storage" className="hover:text-blue-600">{t('nav.storage')}</Link>
        <span>/</span>
        <span className="text-blue-600">{media_type}</span>
        <span>/</span>
        <span className="text-gray-900 font-medium">{source}</span>
      </div>

      {/* Header */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              isVideo ? 'bg-purple-50' : 'bg-green-50'
            }`}>
              {isVideo ? <Film className="w-5 h-5 text-purple-500" /> : <ImageIcon className="w-5 h-5 text-green-500" />}
            </div>
            <div>
              <h1 className="text-lg font-semibold">{source}</h1>
              <div className="flex items-center gap-3 text-xs text-gray-500 mt-0.5">
                <span>{detail.file_count} {isVideo ? t('storage.videos') : t('storage.images')}</span>
                {'total_size_display' in detail && detail.total_size_display && (
                  <><span className="text-gray-300">|</span><span>{detail.total_size_display}</span></>
                )}
                {'total_duration_display' in detail && detail.total_duration_display && (
                  <><span className="text-gray-300">|</span><span>{detail.total_duration_display}</span></>
                )}
              </div>
            </div>
          </div>
          <button onClick={() => setShowInfo(!showInfo)}
            className={`p-1.5 rounded ${showInfo ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:bg-gray-100'}`}
            aria-label={t('datasetDetail.imageInfo')}
            aria-expanded={showInfo}
          ><Info className="w-4 h-4" /></button>
        </div>

        {/* Dataset associations */}
        {detail.datasets && detail.datasets.length > 0 && (
          <div className="mt-3 flex items-center gap-2">
            <Database className="w-3.5 h-3.5 text-gray-400" />
            <span className="text-xs text-gray-500">{t('storage.usedBy')}</span>
            {detail.datasets.map((ds: any) => (
              <Link key={ds.dataset_name} to={`/datasets/${ds.dataset_name}`}
                className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-50 rounded text-xs text-blue-600 hover:bg-blue-100 border border-blue-100"
              >{ds.dataset_name} <span className="text-blue-300">({ds.version})</span></Link>
            ))}
          </div>
        )}
      </div>

      {/* Video extraction panel */}
      {isVideo && (
        <div className="bg-white rounded-lg border border-blue-200 p-4 mb-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-4 h-4 text-blue-500" />
            <h3 className="font-medium text-sm text-blue-700">{t('storage.extractFrames')}</h3>
          </div>

          {/* Dataset mode selector */}
          <div className="flex items-center gap-2 mb-3">
            <button onClick={() => { setDatasetMode('new'); setDatasetName(''); }}
              className={`px-3 py-1 rounded text-xs border transition-colors ${
                datasetMode === 'new'
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300'
              }`}>{t('storage.newDataset')}</button>
            <button onClick={() => setDatasetMode('existing')}
              className={`px-3 py-1 rounded text-xs border transition-colors ${
                datasetMode === 'existing'
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300'
              }`}>{t('storage.addToExisting')}</button>
            <span className="text-[10px] text-gray-400 ml-auto">{t('storage.sourceToDataset')}</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">{t('storage.samplingMethod')}</label>
              <select value={samplingMethod} onChange={e => setSamplingMethod(e.target.value)}
                className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs">
                <option value="interval">{t('storage.intervalEveryN')}</option>
                <option value="scene_change">{t('storage.sceneChange')}</option>
                <option value="motion">{t('storage.motionDetection')}</option>
                <option value="burst">{t('storage.burst')}</option>
                <option value="all">{t('storage.allMethods')}</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">{t('storage.intervalSec')}</label>
              <input type="number" value={intervalSec} min={1} max={300}
                onChange={e => setIntervalSec(Number(e.target.value))}
                className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs" />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">{t('storage.maxFrames')}</label>
              <input type="number" value={maxFrames} min={10} max={10000}
                onChange={e => setMaxFrames(Number(e.target.value))}
                className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs" />
            </div>
            {datasetMode === 'new' ? (
              <div>
                <label className="block text-xs text-gray-500 mb-1">{t('storage.newDatasetName')}</label>
                <input type="text" value={datasetName} placeholder={`extracted_${source}`}
                  onChange={e => setDatasetName(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs" />
              </div>
            ) : (
              <div>
                <label className="block text-xs text-gray-500 mb-1">{t('storage.existingDataset')}</label>
                <select value={datasetName} onChange={e => setDatasetName(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white">
                  <option value="">{t('storage.selectDatasetPlaceholder')}</option>
                  {existingDatasets.map((ds: any) => (
                    <option key={ds.name} value={ds.name}>
                      {ds.name} (v{ds.version_count}, {ds.total_images} {t('storage.images')})
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button onClick={handleExtract} disabled={extracting || (datasetMode === 'existing' && !datasetName)}
              className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">
              {extracting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {extracting ? t('storage.extracting') : datasetMode === 'existing' ? t('storage.addToDataset', { name: datasetName || '...' }) : t('storage.startExtraction')}
            </button>

            {/* Source→Dataset relationship hint */}
            {detail && 'datasets' in detail && (detail as any).datasets && (detail as any).datasets.length > 0 && (
              <span className="text-[10px] text-gray-400">
                {t('storage.sourceUsedBy', { count: (detail as any).datasets.length })}
              </span>
            )}
          </div>

          {/* Extraction result */}
          {extractResult && (
            <div className="mt-3 bg-green-50 border border-green-200 rounded-lg p-3 text-sm">
              <p className="text-green-700 font-medium flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full" />
                {t('storage.extractionCompleted')}
              </p>
              <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                <div className="bg-white rounded p-2 border border-green-100">
                  <span className="text-gray-400">{t('storage.datasetLabel')}</span>
                  <div className="font-medium text-green-700">
                    <Link to={`/datasets/${extractResult.dataset_name}`}
                      className="underline hover:text-green-800">{extractResult.dataset_name}</Link>
                    <span className="text-green-400">/{extractResult.version}</span>
                  </div>
                </div>
                <div className="bg-white rounded p-2 border border-green-100">
                  <span className="text-gray-400">{t('storage.rawFrames')}</span>
                  <div className="font-medium">{extractResult.total_raw_frames || extractResult.total_sampled || 0}</div>
                </div>
                <div className="bg-white rounded p-2 border border-green-100">
                  <span className="text-gray-400">{t('storage.afterQuality')}</span>
                  <div className="font-medium">{extractResult.after_quality || 0}</div>
                </div>
                <div className="bg-white rounded p-2 border border-green-100">
                  <span className="text-gray-400">{t('storage.keptFinal')}</span>
                  <div className="font-medium">{extractResult.after_dedup || extractResult.after_quality || extractResult.total_sampled || 0}</div>
                </div>
              </div>

              {/* Detailed quality report */}
              {extractResult.report && (
                <div className="mt-2 bg-white rounded p-2 border border-green-100 text-xs">
                  <p className="font-medium text-gray-600 mb-1">{t('storage.qualityFilter')}</p>
                  <div className="grid grid-cols-4 gap-2 text-gray-500">
                    <div>{t('storage.blurry')}: <span className="font-medium text-gray-700">{extractResult.report.removed_blurry || 0}</span></div>
                    <div>{t('storage.black')}: <span className="font-medium text-gray-700">{extractResult.report.removed_black || 0}</span></div>
                    <div>{t('storage.overexposed')}: <span className="font-medium text-gray-700">{extractResult.report.removed_overexposed || 0}</span></div>
                    <div>{t('storage.small')}: <span className="font-medium text-gray-700">{extractResult.report.removed_small || 0}</span></div>
                  </div>
                  <div className="mt-1 text-gray-500">
                    {t('storage.dedupRemoved')}: <span className="font-medium text-gray-700">{extractResult.report.dedup_removed || 0}</span>
                  </div>
                  {/* Report summary */}
                  {extractResult.report_summary && (
                    <pre className="mt-2 text-[10px] text-gray-400 bg-gray-50 p-2 rounded overflow-x-auto">
                      {extractResult.report_summary}
                    </pre>
                  )}
                </div>
              )}
            </div>
          )}
          {extractError && (
            <div className="mt-3 bg-red-50 border border-red-200 rounded-lg p-3 text-sm">
              <p className="text-red-700 font-medium">{t('storage.extractionFailed')}</p>
              <p className="text-xs text-red-500 mt-1">{extractError}</p>
            </div>
          )}
        </div>
      )}

      {/* File previews */}
      {'previews' in detail && detail.previews.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {detail.previews.map((file: MediaFile) => (
            <div key={file.filename}
              onClick={() => setPreviewFile(previewFile === file.filename ? null : file.filename)}
              className="bg-white rounded-lg border border-gray-200 overflow-hidden cursor-pointer hover:border-gray-300 transition-all"
            >
              {isVideo ? (
                <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
                  <Film className="w-8 h-8 text-gray-500" />
                  <div className="absolute bottom-1 right-1 bg-black/70 text-white text-[10px] px-1 rounded">
                    {file.duration_seconds ? formatDuration(file.duration_seconds) : ''}
                  </div>
                </div>
              ) : (
                <div className="aspect-square bg-gray-100">
                  <img src={`${API_BASE}/storage/images/${source}/file/${file.filename}`}
                    alt={file.filename} className="w-full h-full object-cover" loading="lazy" />
                </div>
              )}
              <div className="p-2">
                <p className="text-xs text-gray-600 truncate" title={file.filename}>{file.filename}</p>
                <p className="text-[10px] text-gray-400 mt-0.5">
                  {file.width > 0 && `${file.width} × ${file.height}`}
                  {file.width > 0 && ' · '}
                  {file.size_display}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Video files table */}
      {'files' in detail && detail.files && detail.files.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('storage.file')}</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('storage.resolution')}</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('storage.duration')}</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">FPS</th>
                <th className="text-left px-4 py-2 font-medium text-gray-500">{t('storage.codec')}</th>
                <th className="text-right px-4 py-2 font-medium text-gray-500">{t('storage.size')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {detail.files.map((file: MediaFile) => (
                <tr key={file.filename} className="hover:bg-gray-50">
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      <Film className="w-4 h-4 text-purple-300 shrink-0" />
                      <span className="text-xs font-mono">{file.filename}</span>
                    </div>
                  </td>
                  <td className="px-4 py-2 text-xs text-gray-500">
                    {file.width > 0 ? `${file.width} × ${file.height}` : '-'}
                  </td>
                  <td className="px-4 py-2 text-xs text-gray-500">
                    {file.duration_seconds ? formatDuration(file.duration_seconds) : '-'}
                  </td>
                  <td className="px-4 py-2 text-xs text-gray-500">
                    {file.fps ? `${file.fps.toFixed(1)}` : '-'}
                  </td>
                  <td className="px-4 py-2 text-xs text-gray-500">{file.codec || '-'}</td>
                  <td className="px-4 py-2 text-xs text-gray-500 text-right">{file.size_display}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Image preview modal */}
      {previewFile && !isVideo && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={() => setPreviewFile(null)}>
          <div className="relative max-w-5xl max-h-[90vh]" onClick={e => e.stopPropagation()}>
            <img src={`${API_BASE}/storage/images/${source}/file/${previewFile}`}
              alt={previewFile} className="max-w-full max-h-[85vh]" />
          </div>
        </div>
      )}
    </div>
  );
}
