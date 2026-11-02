import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  Wand2, CheckCircle2, XCircle, RotateCcw, Trash2,
} from 'lucide-react';
import type {
  ReviewStatusType, ReviewStatusResult, BatchAnnotationOperation,
} from '../../types';
import { getClassColor, type YoloBox, type BrowseImage } from './shared';

type SidebarTab = 'annotations' | 'prelabel' | 'batch';

export interface AnnotationSidebarProps {
  // ── Tab state ──
  sidebarTab: SidebarTab;
  setSidebarTab: (tab: SidebarTab) => void;

  // ── Annotations tab: review status ──
  reviewData: ReviewStatusResult | null;
  currentReviewStatus: ReviewStatusType;
  handleSetReviewStatus: (status: ReviewStatusType) => void;
  handleBatchApproveAll: () => void;

  // ── Annotations tab: annotation list ──
  annotations: YoloBox[];
  selectedBoxIdx: number | null;
  setSelectedBoxIdx: (idx: number | null) => void;
  currentImage?: BrowseImage;

  // ── Annotations tab: tools panel ──
  allClasses: { id: number; name: string }[];
  handleChangeClass: (newClsId: number) => void;
  handleDelete: () => void;

  // ── Annotations tab: drawing tools ──
  currentClassId: number;
  setCurrentClassId: (id: number) => void;

  // ── Pre-labeling tab ──
  prelabelModel: string;
  setPrelabelModel: (v: string) => void;
  prelabelConf: number;
  setPrelabelConf: (v: number) => void;
  prelabelIou: number;
  setPrelabelIou: (v: number) => void;
  prelabelImgsz: number;
  setPrelabelImgsz: (v: number) => void;
  prelabelOverwrite: boolean;
  setPrelabelOverwrite: (v: boolean) => void;
  prelabelCurrentOnly: boolean;
  setPrelabelCurrentOnly: (v: boolean) => void;
  prelabelRunning: boolean;
  prelabelResult: string | null;
  prelabelError: string | null;
  handlePrelabel: () => void;

  // ── Batch operations tab ──
  batchOp: BatchAnnotationOperation;
  setBatchOp: (op: BatchAnnotationOperation) => void;
  batchClassId: number;
  setBatchClassId: (v: number) => void;
  batchNewClassId: number;
  setBatchNewClassId: (v: number) => void;
  batchMinArea: number;
  setBatchMinArea: (v: number) => void;
  batchMaxArea: number;
  setBatchMaxArea: (v: number) => void;
  batchCurrentOnly: boolean;
  setBatchCurrentOnly: (v: boolean) => void;
  batchRunning: boolean;
  batchResult: string | null;
  batchError: string | null;
  handleBatchOp: () => void;
}

/**
 * Right-hand sidebar for the dataset browse / annotation page.
 * Hosts three tabs: annotations (review status, list, tools, instructions),
 * AI pre-labeling, and batch annotation operations.
 *
 * All state and handlers are owned by the parent (DatasetBrowsePage) and
 * passed down as props; this component is purely presentational.
 */
export default function AnnotationSidebar({
  sidebarTab, setSidebarTab,
  reviewData, currentReviewStatus, handleSetReviewStatus, handleBatchApproveAll,
  annotations, selectedBoxIdx, setSelectedBoxIdx, currentImage,
  allClasses, handleChangeClass, handleDelete,
  currentClassId, setCurrentClassId,
  prelabelModel, setPrelabelModel,
  prelabelConf, setPrelabelConf,
  prelabelIou, setPrelabelIou,
  prelabelImgsz, setPrelabelImgsz,
  prelabelOverwrite, setPrelabelOverwrite,
  prelabelCurrentOnly, setPrelabelCurrentOnly,
  prelabelRunning, prelabelResult, prelabelError, handlePrelabel,
  batchOp, setBatchOp,
  batchClassId, setBatchClassId,
  batchNewClassId, setBatchNewClassId,
  batchMinArea, setBatchMinArea,
  batchMaxArea, setBatchMaxArea,
  batchCurrentOnly, setBatchCurrentOnly,
  batchRunning, batchResult, batchError, handleBatchOp,
}: AnnotationSidebarProps) {
  const { t } = useTranslation();

  return (
    <div className="w-72 bg-white border-l border-gray-200 flex flex-col shrink-0 overflow-y-auto">
      {/* Tab switcher */}
      <div className="flex border-b border-gray-200 shrink-0">
        <button
          onClick={() => setSidebarTab('annotations')}
          className={`flex-1 px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
            sidebarTab === 'annotations'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t('datasetDetail.browse')}
        </button>
        <button
          onClick={() => setSidebarTab('prelabel')}
          className={`flex-1 px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
            sidebarTab === 'prelabel'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Wand2 className="w-3 h-3 inline mr-1" />
          {t('annotationEditor.prelabel')}
        </button>
        <button
          onClick={() => setSidebarTab('batch')}
          className={`flex-1 px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
            sidebarTab === 'batch'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t('annotationEditor.batchOps')}
        </button>
      </div>

      {/* ── Annotations tab ── */}
      {sidebarTab === 'annotations' && (
        <>
          {/* Review status controls */}
          <div className="p-3 border-b border-gray-100">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              {t('annotationEditor.reviewStatus')}
            </h3>
            {reviewData && (
              <p className="text-xs text-gray-400 mb-2">
                {t('annotationEditor.reviewSummary', reviewData.summary)}
              </p>
            )}
            <div className="flex gap-1">
              <button
                onClick={() => handleSetReviewStatus('approved')}
                className={`flex items-center gap-1 px-2 py-1 rounded text-xs border ${
                  currentReviewStatus === 'approved'
                    ? 'bg-green-600 text-white border-green-600'
                    : 'bg-white text-green-600 border-green-200 hover:bg-green-50'
                }`}
              >
                <CheckCircle2 className="w-3 h-3" />
                {t('annotationEditor.reviewMarkApproved')}
              </button>
              <button
                onClick={() => handleSetReviewStatus('rejected')}
                className={`flex items-center gap-1 px-2 py-1 rounded text-xs border ${
                  currentReviewStatus === 'rejected'
                    ? 'bg-red-600 text-white border-red-600'
                    : 'bg-white text-red-600 border-red-200 hover:bg-red-50'
                }`}
              >
                <XCircle className="w-3 h-3" />
                {t('annotationEditor.reviewMarkRejected')}
              </button>
              <button
                onClick={() => handleSetReviewStatus('pending')}
                className={`flex items-center gap-1 px-2 py-1 rounded text-xs border ${
                  currentReviewStatus === 'pending'
                    ? 'bg-yellow-500 text-white border-yellow-500'
                    : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                }`}
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            </div>
            {reviewData && reviewData.summary.pending > 0 && (
              <button
                onClick={handleBatchApproveAll}
                className="mt-2 w-full px-2 py-1 text-xs text-blue-600 border border-blue-200 rounded hover:bg-blue-50"
              >
                {t('annotationEditor.reviewBatchApproveAll')}
              </button>
            )}
          </div>

          {/* Annotation list */}
          <div className="p-3 border-b border-gray-100">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              {t('datasetDetail.browse')} ({annotations.length})
            </h3>
            {annotations.length === 0 ? (
              <p className="text-xs text-gray-400 italic">{t('annotationEditor.noAnnotations')}</p>
            ) : (
              <div className="space-y-1 max-h-52 overflow-y-auto">
                {annotations.map((box, i) => (
                  <div
                    key={i}
                    onClick={() => setSelectedBoxIdx(i)}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                      i === selectedBoxIdx ? 'bg-blue-50 ring-1 ring-blue-200' : 'hover:bg-gray-50'
                    }`}
                  >
                    <span
                      className="w-3 h-3 rounded shrink-0"
                      style={{ backgroundColor: getClassColor(box.cls_id) }}
                    />
                    <span className="font-mono text-gray-700">cls_{box.cls_id}</span>
                    <span className="text-gray-400 ml-auto">
                      {Math.round(box.w * (currentImage?.width || 640))}x{Math.round(box.h * (currentImage?.height || 480))}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Tools panel */}
          {selectedBoxIdx !== null && selectedBoxIdx < annotations.length && (
            <div className="p-3 border-b border-gray-100 space-y-3">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{t('annotationEditor.selectedBox')}</h3>

              <div>
                <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.classId')}</label>
                <div className="flex gap-1 flex-wrap">
                  {allClasses.length > 0 ? allClasses.map(cls => (
                    <button
                      key={cls.id}
                      onClick={() => handleChangeClass(cls.id)}
                      className={`px-2 py-1 rounded text-xs border ${
                        annotations[selectedBoxIdx].cls_id === cls.id
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      {cls.name.replace('class_', '')}
                    </button>
                  )) : (
                    <input
                      type="number"
                      value={annotations[selectedBoxIdx].cls_id}
                      onChange={(e) => handleChangeClass(parseInt(e.target.value) || 0)}
                      className="w-20 border border-gray-300 rounded px-2 py-1 text-xs"
                      min={0}
                    />
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs text-gray-500">
                <div><span className="text-gray-400">xc:</span> {annotations[selectedBoxIdx].xc.toFixed(4)}</div>
                <div><span className="text-gray-400">yc:</span> {annotations[selectedBoxIdx].yc.toFixed(4)}</div>
                <div><span className="text-gray-400">w:</span> {annotations[selectedBoxIdx].w.toFixed(4)}</div>
                <div><span className="text-gray-400">h:</span> {annotations[selectedBoxIdx].h.toFixed(4)}</div>
              </div>

              <button
                onClick={handleDelete}
                className="flex items-center gap-1.5 w-full px-3 py-1.5 bg-red-50 text-red-600 rounded-lg text-xs hover:bg-red-100 border border-red-200"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {t('annotationEditor.deleteAnnotation')}
              </button>
            </div>
          )}

          {/* Drawing tools */}
          <div className="p-3 border-b border-gray-100">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">{t('annotationEditor.newBoxClass')}</h3>
            <div className="flex gap-1 flex-wrap">
              {allClasses.length > 0 ? allClasses.slice(0, 10).map(cls => (
                <button
                  key={cls.id}
                  onClick={() => setCurrentClassId(cls.id)}
                  className={`px-2 py-1 rounded text-xs border ${
                    currentClassId === cls.id
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300'
                  }`}
                >
                  {cls.name.replace('class_', '')}
                </button>
              )) : (
                <input
                  type="number"
                  value={currentClassId}
                  onChange={(e) => setCurrentClassId(parseInt(e.target.value) || 0)}
                  className="w-20 border border-gray-300 rounded px-2 py-1 text-xs"
                  min={0}
                />
              )}
            </div>
          </div>

          {/* Instructions */}
          <div className="p-3 text-xs text-gray-400 space-y-1">
            <p><kbd className="px-1 bg-gray-100 rounded">{t('annotationEditor.shortcutClickSelect')}</kbd></p>
            <p><kbd className="px-1 bg-gray-100 rounded">{t('annotationEditor.shortcutDragMove')}</kbd></p>
            <p><kbd className="px-1 bg-gray-100 rounded">{t('annotationEditor.shortcutClickDragNew')}</kbd></p>
            <p><kbd className="px-1 bg-gray-100 rounded">{t('annotationEditor.shortcutDelDelete')}</kbd></p>
            <p><kbd className="px-1 bg-gray-100 rounded">{t('annotationEditor.shortcutArrowNavigate')}</kbd></p>
            <p><kbd className="px-1 bg-gray-100 rounded">{t('annotationEditor.shortcutCmdSSave')}</kbd></p>
          </div>
        </>
      )}

      {/* ── Pre-label tab ── */}
      {sidebarTab === 'prelabel' && (
        <div className="p-3 space-y-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            <Wand2 className="w-3.5 h-3.5" />
            {t('annotationEditor.prelabel')}
          </h3>

          <div>
            <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.prelabelModel')}</label>
            <input
              type="text"
              value={prelabelModel}
              onChange={(e) => setPrelabelModel(e.target.value)}
              placeholder={t('annotationEditor.prelabelModelPlaceholder')}
              className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.prelabelConf')}</label>
              <input
                type="number"
                value={prelabelConf}
                onChange={(e) => setPrelabelConf(parseFloat(e.target.value) || 0.25)}
                className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                min={0}
                max={1}
                step={0.05}
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.prelabelIou')}</label>
              <input
                type="number"
                value={prelabelIou}
                onChange={(e) => setPrelabelIou(parseFloat(e.target.value) || 0.45)}
                className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                min={0}
                max={1}
                step={0.05}
              />
            </div>
          </div>

          <div>
            <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.prelabelImgsz')}</label>
            <input
              type="number"
              value={prelabelImgsz}
              onChange={(e) => setPrelabelImgsz(parseInt(e.target.value) || 640)}
              className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
              min={320}
              max={1920}
              step={32}
            />
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={prelabelOverwrite}
                onChange={(e) => setPrelabelOverwrite(e.target.checked)}
                className="rounded"
              />
              {t('annotationEditor.prelabelOverwrite')}
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={prelabelCurrentOnly}
                onChange={(e) => setPrelabelCurrentOnly(e.target.checked)}
                className="rounded"
              />
              {t('annotationEditor.prelabelCurrentOnly')}
            </label>
          </div>

          <button
            onClick={handlePrelabel}
            disabled={prelabelRunning || !prelabelModel}
            className="flex items-center gap-1.5 w-full px-3 py-2 bg-purple-600 text-white rounded-lg text-xs hover:bg-purple-700 disabled:opacity-50"
          >
            <Wand2 className="w-3.5 h-3.5" />
            {prelabelRunning ? t('annotationEditor.prelabelRunning') : t('annotationEditor.prelabelRun')}
          </button>

          {prelabelResult && (
            <div className="text-xs text-green-600 bg-green-50 border border-green-200 rounded p-2">
              {prelabelResult}
            </div>
          )}
          {prelabelError && (
            <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded p-2">
              {prelabelError}
            </div>
          )}
        </div>
      )}

      {/* ── Batch operations tab ── */}
      {sidebarTab === 'batch' && (
        <div className="p-3 space-y-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
            {t('annotationEditor.batchOps')}
          </h3>

          {/* Operation selector */}
          <div className="grid grid-cols-1 gap-1">
            {([
              { op: 'delete_by_class', label: t('annotationEditor.batchDeleteByClass') },
              { op: 'delete_by_area', label: t('annotationEditor.batchDeleteByArea') },
              { op: 'reassign_class', label: t('annotationEditor.batchReassignClass') },
              { op: 'clear_images', label: t('annotationEditor.batchClearImages') },
            ] as const).map(({ op, label }) => (
              <button
                key={op}
                onClick={() => setBatchOp(op)}
                className={`px-2 py-1.5 rounded text-xs text-left border ${
                  batchOp === op
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Operation-specific params */}
          {(batchOp === 'delete_by_class' || batchOp === 'reassign_class') && (
            <div>
              <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.batchClassId')}</label>
              {allClasses.length > 0 ? (
                <select
                  value={batchClassId}
                  onChange={(e) => setBatchClassId(parseInt(e.target.value))}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                >
                  {allClasses.map(cls => (
                    <option key={cls.id} value={cls.id}>{cls.name}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={batchClassId}
                  onChange={(e) => setBatchClassId(parseInt(e.target.value) || 0)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                  min={0}
                />
              )}
            </div>
          )}

          {batchOp === 'delete_by_area' && (
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.batchMinArea')}</label>
                <input
                  type="number"
                  value={batchMinArea}
                  onChange={(e) => setBatchMinArea(parseFloat(e.target.value) || 0)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                  min={0}
                  max={1}
                  step={0.001}
                />
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.batchMaxArea')}</label>
                <input
                  type="number"
                  value={batchMaxArea}
                  onChange={(e) => setBatchMaxArea(parseFloat(e.target.value) || 0)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                  min={0}
                  max={1}
                  step={0.001}
                />
              </div>
            </div>
          )}

          {batchOp === 'reassign_class' && (
            <div>
              <label className="text-xs text-gray-500 block mb-1">{t('annotationEditor.batchNewClassId')}</label>
              {allClasses.length > 0 ? (
                <select
                  value={batchNewClassId}
                  onChange={(e) => setBatchNewClassId(parseInt(e.target.value))}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                >
                  {allClasses.map(cls => (
                    <option key={cls.id} value={cls.id}>{cls.name}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={batchNewClassId}
                  onChange={(e) => setBatchNewClassId(parseInt(e.target.value) || 0)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                  min={0}
                />
              )}
            </div>
          )}

          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={batchCurrentOnly}
              onChange={(e) => setBatchCurrentOnly(e.target.checked)}
              className="rounded"
            />
            {t('annotationEditor.batchCurrentOnly')}
          </label>

          <button
            onClick={handleBatchOp}
            disabled={batchRunning}
            className="flex items-center gap-1.5 w-full px-3 py-2 bg-orange-600 text-white rounded-lg text-xs hover:bg-orange-700 disabled:opacity-50"
          >
            {batchRunning ? t('annotationEditor.batchApplying') : t('annotationEditor.batchApply')}
          </button>

          {batchResult && (
            <div className="text-xs text-green-600 bg-green-50 border border-green-200 rounded p-2">
              {batchResult}
            </div>
          )}
          {batchError && (
            <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded p-2">
              {batchError}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
