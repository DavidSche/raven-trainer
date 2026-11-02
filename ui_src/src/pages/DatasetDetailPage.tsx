import { useTranslation } from 'react-i18next';
import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  BarChart3, GitBranch, Tags, Stethoscope, AlertTriangle,
  Plus, Trash2, Check, X, RefreshCw, FileOutput, Loader2, Layers, HardDrive, Clock, History, BookTemplate,
} from 'lucide-react';
import { fetchDataset, fetchIntelligence, fetchVersions, createVersion,
  fetchTags, addTag, removeTag, runHealthCheck,
  fetchMisdata, approveMisdata, rejectMisdata, fetchFeedbackStats,
  fetchChangelog } from '../api/datasets';
import type { Dataset, IntelligenceReport, Version, HealthReport, MisdataItem, ChangelogEntry } from '../types';
import { createTrainingProject } from '../api/training';
import { useNavigate } from 'react-router-dom';
import { ClassDistributionChart } from '../components/charts/ClassDistributionChart';
import { CameraDistributionChart } from '../components/charts/CameraDistributionChart';
import { ObjectSizeChart } from '../components/charts/ObjectSizeChart';
import { VersionTimeline } from '../components/dataset/VersionTimeline';
import { LoadingState, ErrorState, EmptyState } from '../components/common/StateViews';

type Tab = 'analyze' | 'versions' | 'tags' | 'doctor' | 'misdata' | 'build' | 'history';

const COLOR_MAP: Record<string, string> = {
  approved: 'text-green-600 bg-green-50',
  pending: 'text-yellow-600 bg-yellow-50',
  rejected: 'text-red-600 bg-red-50',
};

export default function DatasetDetailPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { name } = useParams<{ name: string }>();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [report, setReport] = useState<IntelligenceReport | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>('analyze');
  const [newVersionSource, setNewVersionSource] = useState('manual');
  const [creatingVersion, setCreatingVersion] = useState(false);
  const [tags, setTags] = useState<{ name: string; color: string; description: string }[]>([]);
  const [newTagName, setNewTagName] = useState('');
  const [newTagColor, setNewTagColor] = useState('#3b82f6');
  const [healthReport, setHealthReport] = useState<HealthReport | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [misdataItems, setMisdataItems] = useState<MisdataItem[]>([]);
  const [feedbackStats, setFeedbackStats] = useState<Record<string, number>>({});
  const [misdataFilter, setMisdataFilter] = useState('');
  const [changelog, setChangelog] = useState<ChangelogEntry[]>([]);
  const [changelogLoading, setChangelogLoading] = useState(false);
  const [changelogTotal, setChangelogTotal] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!name) return;
    setLoadError(null);
    fetchDataset(name).then(setDataset).catch((e) => setLoadError(e.message));
    fetchIntelligence(name).then(setReport).catch((e) => console.error('Failed to load intelligence:', e.message));
    fetchVersions(name).then(setVersions).catch((e) => console.error('Failed to load versions:', e.message));
    fetchTags(name).then(setTags).catch((e) => console.error('Failed to load tags:', e.message));
    fetchMisdata(name).then(setMisdataItems).catch((e) => console.error('Failed to load misdata:', e.message));
    fetchFeedbackStats(name).then(setFeedbackStats).catch((e) => console.error('Failed to load feedback stats:', e.message));
  }, [name]);

  // Fetch changelog
  useEffect(() => {
    if (!name) return;
    setChangelogLoading(true);
    fetchChangelog(name)
      .then(data => { setChangelog(data.entries || []); setChangelogTotal(data.total || 0); setChangelogLoading(false); })
      .catch((e) => { console.error('Failed to load changelog:', e.message); setChangelogLoading(false); });
  }, [name]);

  if (!dataset) {
    if (loadError) {
      return (
        <div className='p-6'>
          <ErrorState error={loadError} />
        </div>
      );
    }
    return <div className='p-6'><LoadingState /></div>;
  }

  const handleCreateVersion = async () => {
    if (!name) return;
    setCreatingVersion(true);
    try {
      await createVersion(name, newVersionSource);
      setVersions(await fetchVersions(name));
    } catch (e) { console.error(e); }
    setCreatingVersion(false);
  };

  const handleAddTag = async () => {
    if (!name || !newTagName.trim()) return;
    try {
      await addTag(name, newTagName.trim(), newTagColor);
      setTags(await fetchTags(name));
      setNewTagName('');
    } catch (e) { console.error(e); }
  };

  const handleRemoveTag = async (tagName: string) => {
    if (!name) return;
    try { await removeTag(name, tagName); setTags(await fetchTags(name)); }
    catch (e) { console.error(e); }
  };

  const handleRunHealth = async () => {
    if (!name) return;
    setHealthLoading(true);
    try { setHealthReport(await runHealthCheck(name)); }
    catch (e) { console.error(e); }
    setHealthLoading(false);
  };

  const handleApprove = async (item: MisdataItem) => {
    if (!name) return;
    try {
      await approveMisdata(name, item.filename, item.category);
      setMisdataItems(await fetchMisdata(name, misdataFilter || undefined));
    } catch (e) { console.error(e); }
  };

  const handleReject = async (item: MisdataItem) => {
    if (!name) return;
    try {
      await rejectMisdata(name, item.filename, item.category);
      setMisdataItems(await fetchMisdata(name, misdataFilter || undefined));
    } catch (e) { console.error(e); }
  };

  const TABS: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'analyze', label: t('datasetDetail.tabAnalysis'), icon: <BarChart3 className="w-4 h-4" /> },
    { key: 'versions', label: t('datasetDetail.tabVersions'), icon: <GitBranch className="w-4 h-4" /> },
    { key: 'tags', label: t('datasetDetail.tabTags'), icon: <Tags className="w-4 h-4" /> },
    { key: 'doctor', label: t('datasetDetail.tabHealth'), icon: <Stethoscope className="w-4 h-4" /> },
    { key: 'misdata', label: t('datasetDetail.tabFeedback'), icon: <AlertTriangle className="w-4 h-4" /> },
    { key: 'build', label: t('datasetDetail.tabBuild'), icon: <FileOutput className="w-4 h-4" /> },
    { key: 'history', label: t('datasetDetail.tabHistory'), icon: <Clock className="w-4 h-4" /> },
  ];

  return (
    <div className="p-6 max-w-6xl">
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
        <Link to="/datasets" className="hover:text-blue-600">{t("datasets.title")}</Link>
        <span>/</span>
        <span className="text-gray-900 font-medium">{dataset.display_name || dataset.name}</span>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-semibold">{dataset.display_name || dataset.name}</h1>
            <p className="text-sm text-gray-500 mt-1">{dataset.name}</p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-gray-500">{dataset.task}</span>
            <span className={'px-2 py-0.5 rounded-full text-xs ' + (dataset.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600')}>{dataset.status}</span>
            <Link
              to={`/datasets/${name}/browse/${dataset.current_version || 'v1'}`}
              className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs hover:bg-blue-700"
            >
              <Layers className="w-3.5 h-3.5" /> {t('datasetDetail.browse')}
            </Link>
          </div>
        </div>
        <div className="flex gap-4 mt-3 text-sm text-gray-500">
          <span>{dataset.total_images.toLocaleString()} {t('datasetDetail.images')}</span>
          <span>{dataset.total_annotations.toLocaleString()} {t('datasetDetail.annotations')}</span>
          <span>{t('datasetDetail.versionsCount', { count: dataset.version_count })}</span>
        </div>
      </div>

      <div className="flex gap-1 mb-4 border-b overflow-x-auto">
        {TABS.map((tab) => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            className={'flex items-center gap-1.5 px-3 py-2 text-sm border-b-2 transition-colors whitespace-nowrap ' + (activeTab === tab.key ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700')}>
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Analysis Tab */}
      {activeTab === 'analyze' && report && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-medium mb-3">{t('datasetDetail.classDistribution')}</h3>
              <ClassDistributionChart data={report.class_distribution} />
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-medium mb-3">{t('datasetDetail.cameraDistribution')}</h3>
              <CameraDistributionChart data={report.camera_distribution} />
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-medium mb-3">{t('datasetDetail.objectSize')}</h3>
              <ObjectSizeChart data={report.object_size_bins} />
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-medium mb-3">{t('datasetDetail.timeDistribution')}</h3>
              <ClassDistributionChart data={report.time_distribution} />
            </div>
          </div>
          {report.bias_warnings?.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h3 className="text-sm font-medium text-yellow-800 mb-2">{t('datasetDetail.biasWarnings')}</h3>
              <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">{report.bias_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
              {report.recommendations?.length > 0 && (
                <><h4 className="text-sm font-medium text-yellow-800 mt-3 mb-1">{t('datasetDetail.recommendations')}</h4>
                <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">{report.recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul></>
              )}
            </div>
          )}
        </div>
      )}
      {activeTab === 'analyze' && !report && (
        <EmptyState message={t('datasetDetail.noAnalysis')} />
      )}

      {/* Versions Tab */}
      {activeTab === 'versions' && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center gap-3">
              <input type="text" value={newVersionSource} onChange={(e) => setNewVersionSource(e.target.value)}
                className="flex-1 border border-gray-300 rounded-lg px-3 py-1.5 text-sm" placeholder={t('datasetDetail.versionSourcePlaceholder')} />
              <button onClick={handleCreateVersion} disabled={creatingVersion || !newVersionSource.trim()}
                className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">
                <Plus className="w-4 h-4" /> {creatingVersion ? t('datasetDetail.creating') : t('datasetDetail.newVersion')}
              </button>
            </div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4"><VersionTimeline versions={versions} /></div>
        </div>
      )}

      {/* Tags Tab */}
      {activeTab === 'tags' && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center gap-2">
              <input type="text" value={newTagName} onChange={(e) => setNewTagName(e.target.value)}
                className="flex-1 border border-gray-300 rounded-lg px-3 py-1.5 text-sm" placeholder={t('datasetDetail.tagNamePlaceholder')}
                onKeyDown={(e) => e.key === 'Enter' && handleAddTag()} />
              <input type="color" value={newTagColor} onChange={(e) => setNewTagColor(e.target.value)} className="w-8 h-8 rounded cursor-pointer border border-gray-300" />
              <button onClick={handleAddTag} disabled={!newTagName.trim()}
                className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">
                <Plus className="w-4 h-4" /> {t('datasetDetail.add')}
              </button>
            </div>
          </div>
          {tags.length === 0 ? (
            <EmptyState message={t('datasetDetail.noTags')} />
          ) : (
            <div className="flex flex-wrap gap-2">
              {tags.map((tag) => (
                <span key={tag.name} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm"
                  style={{ backgroundColor: tag.color + '20', color: tag.color, border: '1px solid ' + tag.color + '40' }}>
                  {tag.name}
                  <button onClick={() => handleRemoveTag(tag.name)} className="hover:opacity-60"><X className="w-3 h-3" /></button>
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Health Tab */}
      {activeTab === 'doctor' && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-sm">{t('datasetDetail.datasetHealthCheck')}</h3>
              <button onClick={handleRunHealth} disabled={healthLoading}
                className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">
                {healthLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                {healthLoading ? t('datasetDetail.running') : t('datasetDetail.runCheck')}
              </button>
            </div>
          </div>
          {healthReport ? (
            <div className="space-y-4">
              {/* Score card */}
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-center gap-4 mb-3">
                  <div className="relative w-20 h-20">
                    <svg className="w-20 h-20 -rotate-90" viewBox="0 0 36 36">
                      <circle cx="18" cy="18" r="15.5" fill="none" stroke="#e5e7eb" strokeWidth="3" />
                      <circle cx="18" cy="18" r="15.5" fill="none"
                        stroke={healthReport.overall_score >= 70 ? '#22c55e' : healthReport.overall_score >= 40 ? '#eab308' : '#ef4444'}
                        strokeWidth="3" strokeDasharray={`${healthReport.overall_score}, 100`}
                        strokeLinecap="round" />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-xl font-bold">{healthReport.overall_score.toFixed(0)}</span>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">{t('datasetDetail.overallHealth')}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {t('datasetDetail.checksSummary', {
                        checks: healthReport.checks?.length || 0,
                        pass: healthReport.checks?.filter((c: any) => c.status === "pass").length || 0,
                        issues: healthReport.checks?.filter((c: any) => c.status !== "pass").length || 0,
                      })}
                    </div>
                    {healthReport.summary && (
                      <div className="text-xs text-gray-500 mt-1">{healthReport.summary}</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Issues */}
              {healthReport.issues?.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-red-800 mb-2">{t('datasetDetail.issues')}</h3>
                  <ul className="space-y-1">{healthReport.issues.map((issue: string, i: number) => (
                    <li key={i} className="text-sm text-red-700 flex items-start gap-2"><AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />{issue}</li>
                  ))}</ul>
                </div>
              )}

              {/* Recommendations */}
              {healthReport.recommendations?.length > 0 && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-amber-800 mb-2">{t('datasetDetail.recommendations')}</h3>
                  <ul className="space-y-1">{healthReport.recommendations.map((r: string, i: number) => (
                    <li key={i} className="text-sm text-amber-700 flex items-start gap-2"><AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />{r}</li>
                  ))}</ul>
                </div>
              )}

              {/* Per-check detail */}
              {healthReport.checks?.length > 0 && (
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-sm font-medium mb-3">{t('datasetDetail.checkDetails')}</h3>
                  <div className="space-y-3">
                    {healthReport.checks.map((check: any, i: number) => {
                      const labels: Record<string, string> = {
                        dataset_size: t('datasetDetail.checkDatasetSize'),
                        camera_balance: t('datasetDetail.checkCameraDiversity'),
                        camera_coverage: t('datasetDetail.checkCameraCoverage'),
                        manifest_integrity: t('datasetDetail.checkManifestIntegrity'),
                        empty_labels: t('datasetDetail.checkEmptyLabels'),
                        invalid_boxes: t('datasetDetail.checkInvalidBoxes'),
                      };
                      const label = labels[check.name] || check.name || t('datasetDetail.checkN', { n: i + 1 });
                      return (
                        <div key={i} className={`rounded-lg border p-3 ${
                          check.status === "pass" ? "bg-green-50 border-green-200" :
                          check.status === "warn" ? "bg-amber-50 border-amber-200" :
                          "bg-red-50 border-red-200"
                        }`}>
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              {check.status === "pass"
                                ? <Check className="w-4 h-4 text-green-600" />
                                : <X className="w-4 h-4 text-red-600" />}
                              <span className="font-medium text-sm">{label}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full ${
                                  check.score >= 70 ? "bg-green-500" : check.score >= 40 ? "bg-yellow-500" : "bg-red-500"
                                }`} style={{ width: `${check.score}%` }} />
                              </div>
                              <span className="text-xs text-gray-500 w-6 text-right">{check.score.toFixed(0)}</span>
                            </div>
                          </div>
                          <p className="text-xs text-gray-600">{check.message}</p>
                          {check.detail && (
                            <code className="text-[10px] text-gray-400 mt-1 block">{check.detail}</code>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <EmptyState
              icon={<Stethoscope className="w-8 h-8 text-gray-300 mb-3" />}
              message={t('datasetDetail.clickRunCheck')}
            />
          )}
        </div>
      )}

      {/* Feedback Tab */}
      {activeTab === 'misdata' && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-3">
            {Object.entries(feedbackStats).length > 0 ? Object.entries(feedbackStats).map(([cat, count]) => (
              <div key={cat} className="bg-white rounded-lg border border-gray-200 p-3 text-center">
                <div className="text-lg font-semibold">{count}</div>
                <div className="text-xs text-gray-500">{cat.replace(/_/g, ' ')}</div>
              </div>
            )) : <div className="col-span-4 text-center text-gray-400 text-sm py-4">{t('datasetDetail.noFeedback')}</div>}
          </div>
          <div className="flex gap-2">
            {['', 'false_positive', 'false_negative', 'hard_case', 'customer_case'].map((cat) => (
              <button key={cat} onClick={() => setMisdataFilter(cat)}
                className={'px-3 py-1 rounded-lg text-xs border transition-colors ' + (misdataFilter === cat ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300')}>
                {cat.replace(/_/g, ' ') || t('datasetDetail.all')}
              </button>
            ))}
          </div>
          {misdataItems.filter(i => !misdataFilter || i.category === misdataFilter).length === 0 ? (
            <EmptyState message={t('datasetDetail.noFeedbackItems')} />
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50"><tr>
                  <th className="text-left px-4 py-2 font-medium text-gray-500">{t('datasetDetail.filename')}</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-500">{t('datasetDetail.category')}</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-500">{t('datasetDetail.status')}</th>
                  <th className="text-right px-4 py-2 font-medium text-gray-500">{t('datasetDetail.actions')}</th>
                </tr></thead>
                <tbody className="divide-y divide-gray-100">
                  {misdataItems.filter(i => !misdataFilter || i.category === misdataFilter).map((item) => (
                    <tr key={item.filename} className="hover:bg-gray-50">
                      <td className="px-4 py-2 text-xs font-mono">{item.filename}</td>
                      <td className="px-4 py-2 text-xs">{item.category}</td>
                      <td className="px-4 py-2">
                        <span className={'px-2 py-0.5 rounded text-xs ' + (COLOR_MAP[item.status] || 'text-gray-600 bg-gray-50')}>{item.status}</span>
                      </td>
                      <td className="px-4 py-2 text-right">
                        {item.status === 'pending' && (
                          <div className="flex items-center justify-end gap-1">
                            <button onClick={() => handleApprove(item)} className="p-1 text-green-600 hover:bg-green-50 rounded" title={t('common.approve')} aria-label={t('common.approve')}><Check className="w-4 h-4" /></button>
                            <button onClick={() => handleReject(item)} className="p-1 text-red-600 hover:bg-red-50 rounded" title={t('common.reject')} aria-label={t('common.reject')}><X className="w-4 h-4" /></button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="w-4 h-4 text-gray-400" />
              <h3 className="font-medium text-sm">{t('datasetDetail.operationHistory')}</h3>
              <span className="text-xs text-gray-400 ml-auto">{t('datasetDetail.records', { count: changelogTotal })}</span>
            </div>
            <p className="text-xs text-gray-400">{t('datasetDetail.historyDesc')}</p>
          </div>

          {changelogLoading ? (
            <LoadingState message={t('datasetDetail.loadingHistory')} />
          ) : changelog.length === 0 ? (
            <EmptyState
              icon={<Clock className="w-8 h-8 text-gray-300 mb-3" />}
              message={t('datasetDetail.noHistory')}
            />
          ) : (
            <div className="space-y-1">
              {changelog.map((entry, i) => {
                const dt = new Date(entry.timestamp);
                const icon = entry.action === "dataset.created" ? "✦" :
                  entry.action === "version.created" ? "⊕" :
                  entry.action === "annotation.updated" ? "✎" :
                  entry.action === "build.executed" ? "▣" :
                  entry.action === "dataset.deleted" ? "✕" : "●";
                const color = entry.action === "dataset.created" ? "text-green-600" :
                  entry.action === "version.created" ? "text-blue-600" :
                  entry.action === "annotation.updated" ? "text-amber-600" :
                  entry.action === "build.executed" ? "text-purple-600" :
                  entry.action === "dataset.deleted" ? "text-red-600" : "text-gray-600";
                return (
                  <div key={i} className="bg-white rounded-lg border border-gray-200 px-4 py-3 flex items-start gap-3 hover:border-gray-300 transition-colors">
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center text-sm shrink-0 mt-0.5 ${color} bg-gray-50`}>
                      {icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-gray-400">{dt.toLocaleString()}</span>
                        {entry.version && (
                          <Link to={`/datasets/${name}/browse/${entry.version}`}
                            className="text-xs text-blue-500 hover:text-blue-700 font-mono"
                          >{entry.version}</Link>
                        )}
                        {entry.count > 0 && <span className="text-xs text-gray-400">({entry.count} {t('datasetDetail.items')})</span>}
                      </div>
                      <p className="text-sm text-gray-700 mt-0.5">{entry.detail}</p>
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider">{entry.action}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Build Tab — redirects to standalone Recipe page */}
      {activeTab === 'build' && (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center space-y-4">
          <FileOutput className="w-12 h-12 text-purple-300 mx-auto" />
          <h3 className="text-base font-semibold text-gray-700">{t('datasetDetail.recipesStandalone')}</h3>
          <p className="text-sm text-gray-500 max-w-md mx-auto">
            {t('datasetDetail.recipesMovedDescription')}
          </p>
          <div className="flex items-center justify-center gap-3 pt-2">
            <Link to="/recipes"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors">
              <BookTemplate className="w-4 h-4" /> {t('datasetDetail.goToRecipes')}
            </Link>
            <Link to={"/recipes/" + name + "_recipe"}
              className="inline-flex items-center gap-2 px-5 py-2.5 border border-gray-200 text-gray-600 rounded-lg text-sm hover:bg-gray-50 transition-colors">
              <Plus className="w-4 h-4" /> {t('datasetDetail.createRecipeFor', { name })}
            </Link>
          </div>
          <p className="text-[10px] text-gray-400">
            {t('datasetDetail.storedAt', { path: 'data/recipes/' })}
          </p>
        </div>
      )}
    </div>
  );
}





