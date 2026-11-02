import React, { useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useParams } from 'react-router-dom';
import {
  fetchExperiment, fetchExperimentConfig, fetchExperimentLog,
  fetchExperimentReport, fetchExperimentResults,
  connectTrainingWebSocket,
} from '../api/training';
import { StatusBadge } from '../components/common/StatusBadge';
import { TrainingMetricsChart, type MetricsDataPoint } from '../components/charts/TrainingMetricsChart';
import { TrainingProgress } from '../components/training/TrainingProgress';
import type { Experiment, ExperimentResults, OnnxVerification, ExperimentConfig } from '../types';
import {
  Settings, TrendingUp, Box, FileText, Download, Clock, GitCommit,
  Image, Table, ShieldCheck, AlertTriangle, CheckCircle, XCircle, Maximize2,
} from 'lucide-react';

type Tab = 'metrics' | 'images' | 'config' | 'model' | 'report' | 'onnx';

export default function TrainingDetailPage() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [config, setConfig] = useState<ExperimentConfig | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('metrics');
  const [logContent, setLogContent] = useState('');
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [liveLogs, setLiveLogs] = useState<string[]>([]);

  const [experimentError, setExperimentError] = useState<string | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    fetchExperiment(id).then(setExperiment).catch((e) => setExperimentError(e.message));
    fetchExperimentConfig(id).then(setConfig).catch((e) => setConfigError(e.message));
    fetchExperimentResults(id).then(setResults).catch((e) => setResultsError(e.message));
  }, [id]);

  useEffect(() => {
    if (!id) return;
    fetchExperimentLog(id).then((data) => {
      if (data.log) setLogContent(data.log || '');
    }).catch((e) => console.error('Failed to load experiment log:', e.message));
  }, [id]);

  useEffect(() => {
    if (!id) return;
    fetchExperimentReport(id).then((data) => {
      if (data.html) { setReportHtml(data.html); setReportError(null); }
    }).catch((e) => {
      setReportError(e.message); setReportHtml(null);
    });
  }, [id]);

  useEffect(() => {
    if (!id || experiment?.status !== 'running') return;
    const cleanup = connectTrainingWebSocket(id, (msg) => {
      setLiveLogs((prev) => [...prev, msg]);
    });
    return cleanup;
  }, [id, experiment?.status]);

  if (!experiment) {
    if (experimentError) {
      return (
        <div className="p-6 max-w-7xl">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-3" />
            <p className="text-sm font-medium text-red-600 mb-1">{t('trainingDetail.loadError')}</p>
            <p className="text-xs text-red-400">{experimentError}</p>
          </div>
        </div>
      );
    }
    return <div className="p-6 text-gray-400">{t('trainingDetail.loading')}</div>;
  }

  const isRunning = experiment.status === 'running';
  const metrics = experiment.metrics || {};

  const metricsData: MetricsDataPoint[] = [];
  if (results?.csv?.rows && results.csv.rows.length > 0) {
    results.csv.rows.forEach((row: Record<string, unknown>) => {
      const ep = Number(row['epoch'] ?? 0);
      if (!ep) return;
      const dp: MetricsDataPoint = { epoch: ep };
      const h = results.csv!.headers;
      h.forEach((header: string) => {
        const val = row[header];
        if (typeof val === 'number') {
          const hLower = header.toLowerCase();
          if (hLower.includes('map50') && !hLower.includes('95')) dp.mAP50 = val;
          else if (hLower.includes('map50') && hLower.includes('95')) dp.mAP50_95 = val;
          else if (hLower.includes('precision')) dp.precision = val;
          else if (hLower.includes('recall')) dp.recall = val;
          else if (hLower.includes('loss') || hLower.includes('box') || hLower.includes('cls') || hLower.includes('dfl')) {
            const lossKey = 'loss' as keyof MetricsDataPoint;
            dp[lossKey] = (dp[lossKey] || 0) + val;
          }
        }
      });
      metricsData.push(dp);
    });
  }

  // NOTE: Do NOT synthesize fake training curves when CSV is missing.
  // Showing interpolated/imaginary metrics misleads users into thinking
  // they are viewing real training progress. Show an empty-state instead.

  const reproducibility = (config as unknown as Record<string, unknown>)?.reproducibility as Record<string, unknown> | undefined;
  const expCfg = experiment.config as Record<string, unknown> | undefined;
  const onnxVerification = expCfg?.onnx_verification as OnnxVerification | undefined;

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'metrics', label: t('trainingDetail.tabMetrics'), icon: <TrendingUp className="w-4 h-4" /> },
    { key: 'images', label: t('trainingDetail.tabImages'), icon: <Image className="w-4 h-4" /> },
    { key: 'config', label: t('trainingDetail.tabConfig'), icon: <Settings className="w-4 h-4" /> },
    { key: 'model', label: t('trainingDetail.tabModel'), icon: <Box className="w-4 h-4" /> },
    { key: 'onnx', label: t('trainingDetail.tabOnnx'), icon: <ShieldCheck className="w-4 h-4" /> },
    { key: 'report', label: t('trainingDetail.tabReport'), icon: <FileText className="w-4 h-4" /> },
  ];

  return (
    <div className="p-6 max-w-7xl">
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
        <Link to="/training" className="hover:text-blue-600">{t('trainingDetail.breadcrumb')}</Link>
        <span>/</span>
        <span className="text-gray-900 font-medium">{experiment.id}</span>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold">{experiment.id}</h1>
            <StatusBadge status={experiment.status} />
          </div>
          <div className="text-xs text-gray-400 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {experiment.created_at.slice(0, 19).replace('T', ' ')}
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldProject')}</div>
            <div className="font-medium">{experiment.project}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldModel')}</div>
            <div className="font-medium">{experiment.model}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldBackend')}</div>
            <div className="font-medium">{experiment.backend}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldEpochs')}</div>
            <div className="font-medium">{experiment.epochs}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldBatch')}</div>
            <div className="font-medium">{experiment.batch}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldImageSize')}</div>
            <div className="font-medium">{experiment.imgsz}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldDevice')}</div>
            <div className="font-medium">{experiment.device || 'cuda:0'}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">{t('trainingDetail.fieldDataset')}</div>
            <div className="font-medium">{experiment.dataset || 'N/A'}</div>
          </div>
        </div>
      </div>

      {isRunning && (
        <div className="mb-6">
          <TrainingProgress experimentId={experiment.id} isRunning={true} />
        </div>
      )}

      <div className="flex gap-1 mb-4 border-b">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-3 py-2 text-sm border-b-2 transition-colors ${
              activeTab === tab.key
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="font-medium mb-4">{t('trainingDetail.trainingMetrics')}</h2>
            {metricsData.length > 0 ? (
              <TrainingMetricsChart data={metricsData} />
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                <AlertTriangle className="w-8 h-8 mb-3 text-amber-400" />
                <p className="text-sm font-medium text-gray-500">{t('trainingDetail.noEpochData')}</p>
                <p className="text-xs text-gray-400 mt-1">{t('trainingDetail.noEpochDataHint')}</p>
              </div>
            )}
          </div>

          {results?.csv && results.csv.rows.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="font-medium mb-2 flex items-center gap-2">
                <Table className="w-4 h-4" /> {t('trainingDetail.resultsCsv')}
              </h2>
              <div className="overflow-x-auto max-h-64 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-50">
                      {results.csv.headers.map((h: string) => (
                        <th key={h} className="px-2 py-1 text-left font-medium text-gray-500 whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.csv.rows.map((row: Record<string, unknown>, i: number) => (
                      <tr key={i} className="hover:bg-gray-50">
                        {results.csv!.headers.map((h: string) => (
                          <td key={h} className="px-2 py-1 border-t border-gray-100 whitespace-nowrap">
                            {typeof row[h] === 'number' ? (row[h] as number).toFixed(4) : String(row[h] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {Object.keys(metrics).length > 0 && (
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
              {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="text-xs text-gray-500 truncate">{key}</div>
                  <div className="text-lg font-semibold mt-1">
                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
                  </div>
                </div>
              ))}
            </div>
          )}

          {results?.args_yaml && (
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="font-medium mb-2">{t('trainingDetail.trainingArgs')}</h2>
              <pre className="bg-gray-50 rounded-lg p-4 text-xs font-mono overflow-x-auto">
                {results.args_yaml}
              </pre>
            </div>
          )}
        </div>
      )}

      {activeTab === 'images' && (
        <div className="space-y-4">
          <h2 className="font-medium">{t('trainingDetail.trainingResultImages')}</h2>
          {results?.images && results.images.length > 0 ? (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
              {results.images.map((img) => {
                const imgUrl = `/api/v1/training/experiments/${experiment.id}/results-image?name=${encodeURIComponent(img.name)}`;
                return (
                  <div key={img.name} className="bg-white rounded-lg border border-gray-200 overflow-hidden group">
                    <div className="relative aspect-video bg-gray-100">
                      <img
                        src={imgUrl}
                        alt={img.name}
                        className="w-full h-full object-contain cursor-pointer"
                        onClick={() => setSelectedImage(imgUrl)}
                      />
                      <button
                        onClick={() => setSelectedImage(imgUrl)}
                        className="absolute top-2 right-2 bg-black/50 text-white p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Maximize2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <div className="px-3 py-2">
                      <div className="text-xs font-mono truncate">{img.name}</div>
                      <div className="text-xs text-gray-400">{(img.size / 1024).toFixed(1)} KB</div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-400">
              <Image className="w-8 h-8 mx-auto mb-2" />
              {resultsError
                ? t('trainingDetail.noResults')
                : t('trainingDetail.noImages')
              }
            </div>
          )}

          {selectedImage && (
            <div
              className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center cursor-pointer"
              onClick={() => setSelectedImage(null)}
            >
              <img src={selectedImage} className="max-w-[90vw] max-h-[90vh] object-contain" alt="Enlarged" />
            </div>
          )}
        </div>
      )}

      {activeTab === 'config' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="font-medium mb-4">{t('trainingDetail.configuration')}</h2>
          {config ? (
            <pre className="bg-gray-50 rounded-lg p-4 text-xs font-mono overflow-x-auto max-h-96 overflow-y-auto">
              {JSON.stringify(config, null, 2)}
            </pre>
          ) : (
            <div className="text-gray-400 text-sm">{t('trainingDetail.noConfig')}</div>
          )}

          {reproducibility && Object.keys(reproducibility).length > 0 && (
            <div className="mt-6">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <GitCommit className="w-4 h-4" /> {t('trainingDetail.reproducibility')}
              </h3>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(reproducibility).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 rounded-lg p-3">
                    <div className="text-xs text-gray-500">{key}</div>
                    <div className="text-xs font-mono mt-1 break-all">{String(value)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'model' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="font-medium mb-4">{t('trainingDetail.modelFiles')}</h2>
          {experiment.status === 'completed' ? (
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                {t('trainingDetail.modelFilesDesc')} <code className="bg-gray-100 px-1 rounded">projects/{experiment.project}/models/</code>
              </p>
              <div className="grid grid-cols-2 gap-3 max-w-md">
                <a
                  href={`/api/v1/training/experiments/${experiment.id}/download`}
                  className="flex items-center gap-2 px-4 py-3 bg-blue-50 text-blue-700 rounded-lg text-sm hover:bg-blue-100 border border-blue-200"
                >
                  <Download className="w-4 h-4" /> {t('trainingDetail.downloadPt')}
                </a>
                <a
                  href={`/api/v1/training/experiments/${experiment.id}/download?format=onnx`}
                  className="flex items-center gap-2 px-4 py-3 bg-green-50 text-green-700 rounded-lg text-sm hover:bg-green-100 border border-green-200"
                >
                  <Download className="w-4 h-4" /> {t('trainingDetail.downloadOnnx')}
                </a>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-sm">
              <p>{t('trainingDetail.modelNotReady')}</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'onnx' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="font-medium mb-4 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4" /> {t('trainingDetail.onnxExportVerification')}
            </h2>

            {experiment.status !== 'completed' ? (
              <div className="text-gray-400 text-sm">{t('trainingDetail.onnxNotReady')}</div>
            ) : (
              <div className="space-y-4">
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium mb-2">{t('trainingDetail.ptVsOnnxConsistency')}</h3>
                  {onnxVerification?.pt_vs_onnx ? (
                    <div className="space-y-2 text-xs">
                      <div className="flex items-center gap-2">
                        {onnxVerification.pt_vs_onnx.passed ? (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-500" />
                        )}
                        <span className={onnxVerification.pt_vs_onnx.passed ? 'text-green-700' : 'text-red-700'}>
                          {onnxVerification.pt_vs_onnx.passed ? t('trainingDetail.passed') : t('trainingDetail.failed')}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-3">
                        <div className="bg-gray-50 rounded p-2">
                          <div className="text-gray-500">{t('trainingDetail.imagesCompared')}</div>
                          <div className="font-medium">{onnxVerification.pt_vs_onnx.images_compared}</div>
                        </div>
                        <div className="bg-gray-50 rounded p-2">
                          <div className="text-gray-500">{t('trainingDetail.maxIouDiff')}</div>
                          <div className="font-medium">{(onnxVerification.pt_vs_onnx.max_iou_diff).toFixed(6)}</div>
                        </div>
                        <div className="bg-gray-50 rounded p-2">
                          <div className="text-gray-500">{t('trainingDetail.maxScoreDiff')}</div>
                          <div className="font-medium">{(onnxVerification.pt_vs_onnx.max_score_diff).toFixed(6)}</div>
                        </div>
                        <div className="bg-gray-50 rounded p-2">
                          <div className="text-gray-500">{t('trainingDetail.classDiff')}</div>
                          <div className="font-medium">{onnxVerification.pt_vs_onnx.max_class_diff}</div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-400 text-xs">{t('trainingDetail.notAvailable')}</div>
                  )}
                </div>

                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium mb-2">{t('trainingDetail.onnxValidation')}</h3>
                  {onnxVerification?.onnx_val ? (
                    <div className="space-y-2">
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                        <div className="bg-gray-50 rounded p-3">
                          <div className="text-xs text-gray-500">mAP50</div>
                          <div className="text-lg font-semibold">
                            {onnxVerification.onnx_val.map50?.toFixed(4) ?? '-'}
                          </div>
                        </div>
                        <div className="bg-gray-50 rounded p-3">
                          <div className="text-xs text-gray-500">mAP50-95</div>
                          <div className="text-lg font-semibold">
                            {onnxVerification.onnx_val.map50_95?.toFixed(4) ?? '-'}
                          </div>
                        </div>
                      </div>
                      {onnxVerification.onnx_val.per_class_map50_95 && (
                        <div className="mt-3">
                          <div className="text-xs text-gray-500 mb-1">{t('trainingDetail.perClassMap')}</div>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(onnxVerification.onnx_val.per_class_map50_95).map(([cls, val]) => (
                              <span key={cls} className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-xs">
                                Class {cls}: {val.toFixed(4)}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-gray-400 text-xs">{t('trainingDetail.onnxValNotPerformed')}</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'report' && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          {reportHtml ? (
            <iframe
              srcDoc={reportHtml}
              className="w-full border-0"
              style={{ height: '80vh' }}
              title="Training Report"
              sandbox="allow-scripts"
            />
          ) : reportError ? (
            <div className="p-6">
              <h2 className="font-medium mb-4">{t('trainingDetail.trainingReport')}</h2>
              <div className="text-amber-600 text-sm bg-amber-50 rounded-lg p-4 border border-amber-200">
                <AlertTriangle className="w-4 h-4 inline mr-1" />
                {t('trainingDetail.reportNotAvailable')} {experiment.status === 'completed' ? t('trainingDetail.reportRunAgain') : t('trainingDetail.reportWillGenerate')}
              </div>
            </div>
          ) : (
            <div className="p-6">
              <h2 className="font-medium mb-4">{t('trainingDetail.trainingReport')}</h2>
              <div className="text-gray-400 text-sm">{t('trainingDetail.loadingReport')}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
