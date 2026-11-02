import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import {
  AlertTriangle, CheckCircle, XCircle, Merge, RefreshCw,
  ThumbsUp, ThumbsDown, Image as ImageIcon, X,
} from 'lucide-react';
import { fetchFeedbackDatasets, fetchFeedback, fetchFeedbackCategories, approveFeedback, rejectFeedback, mergeFeedback } from '../api/feedback';
import type { MisdataItem } from '../types';
import { StatusBadge } from '../components/common/StatusBadge';

const CATEGORY_LABELS: Record<string, string> = {
  false_positive: 'categoryFalsePositive',
  false_negative: 'categoryFalseNegative',
  hard_case: 'categoryHardCase',
  customer_case: 'categoryCustomerCase',
};

const CATEGORY_COLORS: Record<string, string> = {
  false_positive: 'text-red-600 bg-red-50',
  false_negative: 'text-orange-600 bg-orange-50',
  hard_case: 'text-yellow-600 bg-yellow-50',
  customer_case: 'text-blue-600 bg-blue-50',
};

export default function MisdataPage() {
  const { t } = useTranslation();
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [items, setItems] = useState<MisdataItem[]>([]);
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [categories, setCategories] = useState<Record<string, { total: number; pending: number; approved: number; rejected: number }>>({});
  const [loading, setLoading] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mergeResult, setMergeResult] = useState<{ count: number; dataset: string } | null>(null);

  useEffect(() => {
    fetchFeedbackDatasets().then(setDatasets).catch((err) => {
      console.error('Failed to fetch feedback datasets:', err);
      setError(t('feedback.loadError'));
    });
  }, []);

  const loadData = async (dataset: string) => {
    setLoading(true);
    try {
      const [feedbackData, catData] = await Promise.all([
        fetchFeedback(dataset, selectedCategory || undefined),
        fetchFeedbackCategories(dataset),
      ]);
      setItems(feedbackData.items);
      setSummary(feedbackData.summary);
      setCategories(catData);
      setError(null);
    } catch (err) {
      console.error('Failed to load feedback data for dataset', dataset, err);
      setError(t('feedback.loadError'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedDataset) loadData(selectedDataset);
  }, [selectedDataset, selectedCategory]);

  const handleApprove = async (item: MisdataItem) => {
    try {
      await approveFeedback(selectedDataset, item.category, item.filename);
      setError(null);
      await loadData(selectedDataset);
    } catch (err) {
      console.error('Failed to approve feedback item', item.filename, err);
      setError(t('feedback.approveError'));
    }
  };

  const handleReject = async (item: MisdataItem) => {
    try {
      await rejectFeedback(selectedDataset, item.category, item.filename);
      setError(null);
      await loadData(selectedDataset);
    } catch (err) {
      console.error('Failed to reject feedback item', item.filename, err);
      setError(t('feedback.rejectError'));
    }
  };

  const handleMerge = async () => {
    if (!selectedDataset) return;
    setMergeResult(null);
    try {
      const result = await mergeFeedback(selectedDataset);
      setError(null);
      console.info('Merged ' + (result.count as number) + ' approved images into dataset');
      setMergeResult({ count: result.count as number, dataset: selectedDataset });
      await loadData(selectedDataset);
    } catch (err) {
      console.error('Failed to merge feedback for dataset', selectedDataset, err);
      setError(t('feedback.mergeError'));
    }
  };

  const pendingCount = Object.values(categories).reduce((sum, c) => sum + c.pending, 0);

  return (
    <div className="p-6 max-w-6xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">{t('feedback.title')}</h1>
        {selectedDataset && (
          <button
            onClick={handleMerge}
            disabled={pendingCount === 0}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            <Merge className="w-4 h-4" /> {t('feedback.mergeToDataset', { count: pendingCount })}
          </button>
        )}
      </div>

      {mergeResult && (
        <div className="mt-3 flex items-center gap-3 bg-green-50 border border-green-200 rounded-lg px-4 py-2">
          <CheckCircle className="w-4 h-4 text-green-500" />
          <span className="text-sm text-green-700">
            {t('feedback.mergeSuccess', { count: mergeResult.count })}
          </span>
          <Link to={`/datasets/${mergeResult.dataset}`} className="text-xs text-blue-600 hover:underline ml-auto">
            {t('feedback.viewDataset')} →
          </Link>
          <Link to="/recipes" className="text-xs text-blue-600 hover:underline">
            {t('feedback.rebuildWorkspace')} →
          </Link>
        </div>
      )}

      {error && (
        <div className="mb-4 flex items-start gap-3 rounded-lg border border-red-300 bg-red-50 p-3">
          <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700 flex-1">{error}</p>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-600"
            aria-label={t('feedback.dismissError')}
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {datasets.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-400">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
          <p>{t('feedback.noFeedback')}</p>
          <p className="text-sm mt-1">
            {t('feedback.collectHint')}
          </p>
        </div>
      ) : (
        <>
          <div className="flex gap-3 mb-6">
            <select
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={selectedDataset}
              onChange={(e) => { setSelectedDataset(e.target.value); setSelectedCategory(''); }}
            >
              <option value="">{t('feedback.selectDataset')}</option>
              {datasets.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
            {selectedDataset && (
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
              >
                <option value="">{t('feedback.allCategories')}</option>
                {Object.keys(CATEGORY_LABELS).map((c) => (
                  <option key={c} value={c}>{t('feedback.' + CATEGORY_LABELS[c])}</option>
                ))}
              </select>
            )}
            <button onClick={() => loadData(selectedDataset)} className="p-2 text-gray-500 hover:text-gray-700" aria-label={t('ingestionProfiles.refresh')}>
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>

          {Object.keys(categories).length > 0 && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
              {Object.entries(categories).map(([cat, stats]) => (
                <div
                  key={cat}
                  className={`rounded-lg border p-3 cursor-pointer transition-colors ${
                    selectedCategory === cat
                      ? 'border-blue-400 bg-blue-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedCategory(selectedCategory === cat ? '' : cat)}
                >
                  <div className="text-xs font-medium text-gray-500">{t('feedback.' + CATEGORY_LABELS[cat])}</div>
                  <div className="flex items-center gap-3 mt-2 text-sm">
                    <span className="font-semibold">{stats.total}</span>
                    {stats.pending > 0 && (
                      <span className="text-yellow-600 text-xs">{t('feedback.pending', { count: stats.pending })}</span>
                    )}
                  </div>
                  <div className="flex gap-2 mt-1 text-xs text-gray-400">
                    <span className="text-green-600">{t('feedback.approved', { count: stats.approved })}</span>
                    <span className="text-red-600">{t('feedback.rejected', { count: stats.rejected })}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {summary && Object.keys(summary).length > 0 && (
            <div className="flex items-center gap-3 text-sm text-gray-500 mb-4">
              <span>{t('feedback.total', { count: items.length })}</span>
              <span className="text-green-600">{t('feedback.approved', { count: summary.approved || 0 })}</span>
              <span className="text-red-600">{t('feedback.rejected', { count: summary.rejected || 0 })}</span>
              <span className="text-yellow-600">{t('feedback.pending', { count: summary.pending || 0 })}</span>
            </div>
          )}

          {loading ? (
            <div className="text-gray-400 py-12 text-center">{t('feedback.loading')}</div>
          ) : items.length === 0 ? (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-400">
              {t('feedback.noItems')}
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
              {items.map((item) => (
                <div
                  key={item.filename}
                  className={`bg-white rounded-lg border overflow-hidden group relative ${
                    item.status === 'approved'
                      ? 'border-green-300 ring-1 ring-green-200'
                      : item.status === 'rejected'
                        ? 'border-red-300 opacity-60'
                        : 'border-gray-200'
                  }`}
                >
                  <div
                    className="aspect-square bg-gray-100 cursor-pointer overflow-hidden"
                    onClick={() => setPreviewImage(item.thumbnail || null)}
                  >
                    {item.thumbnail ? (
                      <img src={item.thumbnail} alt={item.filename} className="w-full h-full object-cover" />
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-300">
                        <ImageIcon className="w-8 h-8" />
                      </div>
                    )}
                  </div>

                  <div className="p-2">
                    <div className="text-xs text-gray-500 truncate" title={item.filename}>
                      {item.filename}
                    </div>
                    <div className={`text-xs mt-0.5 ${CATEGORY_COLORS[item.category] || ''}`}>
                      {t('feedback.' + CATEGORY_LABELS[item.category])}
                    </div>
                  </div>

                  {item.status === 'pending' && (
                    <div className="absolute inset-0 bg-black/40 flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => handleApprove(item)}
                        className="p-2 bg-green-500 text-white rounded-full hover:bg-green-600"
                        title={t('feedback.approve')}
                        aria-label={t('feedback.approve')}
                      >
                        <ThumbsUp className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleReject(item)}
                        className="p-2 bg-red-500 text-white rounded-full hover:bg-red-600"
                        title={t('feedback.reject')}
                        aria-label={t('feedback.reject')}
                      >
                        <ThumbsDown className="w-4 h-4" />
                      </button>
                    </div>
                  )}

                  {item.status !== 'pending' && (
                    <div className="absolute top-1 right-1">
                      {item.status === 'approved' ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-400" />
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {previewImage && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={() => setPreviewImage(null)}>
          <div className="max-w-3xl max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <img src={previewImage} className="max-w-full max-h-[90vh] object-contain rounded-lg" alt="Preview" />
          </div>
        </div>
      )}
    </div>
  );
}
