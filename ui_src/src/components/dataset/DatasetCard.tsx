import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Database, Layers, Image, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Dataset } from '../../types';
import { deleteDataset } from '../../api/datasets';
import { ConfirmDialog } from '../ui/ConfirmDialog';

export function DatasetCard({ dataset, onDeleted }: { dataset: Dataset; onDeleted?: () => void }) {
  const { t } = useTranslation();
  const [showConfirm, setShowConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setShowConfirm(true);
  };

  const handleConfirmDelete = async () => {
    setDeleting(true);
    try {
      await deleteDataset(dataset.name);
      setShowConfirm(false);
      if (onDeleted) onDeleted();
    } catch (e) {
      console.error('Failed to delete dataset:', dataset.name, e);
      setDeleting(false);
      setShowConfirm(false);
    }
  };

  const handleCancelDelete = () => {
    if (!deleting) setShowConfirm(false);
  };

  return (
    <>
      <div className="relative group">
        <Link
          to={`/datasets/${dataset.name}`}
          className="block bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-blue-50 rounded-lg">
              <Database className="w-5 h-5 text-blue-600" />
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="font-medium text-sm truncate">{dataset.display_name || dataset.name}</h3>
              <p className="text-xs text-gray-500">{dataset.name}</p>
            </div>
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              dataset.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
            }`}>
              {dataset.task}
            </span>
          </div>

          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <Image className="w-3.5 h-3.5" />
              {dataset.total_images.toLocaleString()}
            </span>
            <span className="flex items-center gap-1">
              <Layers className="w-3.5 h-3.5" />
              {dataset.version_count} {t('common.versions')}
            </span>
          </div>

          {dataset.health_score > 0 && (
            <div className="mt-2 flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    dataset.health_score >= 70 ? 'bg-green-500' : dataset.health_score >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${dataset.health_score}%` }}
                />
              </div>
              <span className="text-xs text-gray-400">{Math.round(dataset.health_score)}%</span>
            </div>
          )}
        </Link>

        {onDeleted && (
          <button onClick={handleDeleteClick}
            className="absolute top-2 right-2 p-1.5 bg-white/80 rounded text-red-400 opacity-0 group-hover:opacity-100 hover:text-red-600 hover:bg-red-50 transition-all"
            title={`Delete ${dataset.name}`}>
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <ConfirmDialog
        open={showConfirm}
        title={t('confirm.deleteTitle')}
        message={t('confirm.deleteMessage', { name: dataset.name })}
        confirmLabel={t('confirm.confirmDelete')}
        cancelLabel={t('common.cancel')}
        onConfirm={handleConfirmDelete}
        onCancel={handleCancelDelete}
        loading={deleting}
        variant="danger"
      />
    </>
  );
}
