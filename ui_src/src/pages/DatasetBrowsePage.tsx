import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  ChevronLeft, ChevronRight,
  ZoomIn, ZoomOut, Save, Info,
  CheckCircle2, XCircle, Clock,
} from 'lucide-react';
import {
  fetchDataset, fetchVersions,
  prelabelImages, fetchReviewStatus, setReviewStatus,
  batchSetReviewStatus, batchAnnotationOperation,
} from '../api/datasets';
import { api } from '../api/client';
import type {
  Dataset, Version,
  ReviewStatusType, ReviewStatusResult,
  BatchAnnotationOperation,
} from '../types';
import { LoadingState, EmptyState } from '../components/common/StateViews';
import AnnotationSidebar from '../components/annotation/AnnotationSidebar';
import ImageInfoOverlay from '../components/annotation/ImageInfoOverlay';
import ThumbnailStrip from '../components/annotation/ThumbnailStrip';
import {
  API_BASE,
  getClassColor,
  type BrowseImage,
  type BrowseResult,
  type YoloBox,
} from '../components/annotation/shared';

export default function DatasetBrowsePage() {
  const { t } = useTranslation();
  const { name, version } = useParams<{ name: string; version: string }>();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [browseData, setBrowseData] = useState<BrowseResult | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [annotations, setAnnotations] = useState<YoloBox[]>([]);
  const [selectedBoxIdx, setSelectedBoxIdx] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawEnd, setDrawEnd] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [dragMode, setDragMode] = useState<'none' | 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br' | 'resize-t' | 'resize-b' | 'resize-l' | 'resize-r'>('none');
  const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null);
  const [dragStartBox, setDragStartBox] = useState<YoloBox | null>(null);
  const [currentClassId, setCurrentClassId] = useState(0);
  const [allClasses, setAllClasses] = useState<{ id: number; name: string }[]>([]);
  const [showInfo, setShowInfo] = useState(false);

  // ── Pre-labeling state ──
  const [prelabelModel, setPrelabelModel] = useState('');
  const [prelabelConf, setPrelabelConf] = useState(0.25);
  const [prelabelIou, setPrelabelIou] = useState(0.45);
  const [prelabelImgsz, setPrelabelImgsz] = useState(640);
  const [prelabelOverwrite, setPrelabelOverwrite] = useState(false);
  const [prelabelCurrentOnly, setPrelabelCurrentOnly] = useState(false);
  const [prelabelRunning, setPrelabelRunning] = useState(false);
  const [prelabelResult, setPrelabelResult] = useState<string | null>(null);
  const [prelabelError, setPrelabelError] = useState<string | null>(null);

  // ── Review status state ──
  const [reviewData, setReviewData] = useState<ReviewStatusResult | null>(null);
  const [currentReviewStatus, setCurrentReviewStatus] = useState<ReviewStatusType>('pending');

  // ── Batch operations state ──
  const [batchOp, setBatchOp] = useState<BatchAnnotationOperation>('delete_by_class');
  const [batchClassId, setBatchClassId] = useState(0);
  const [batchNewClassId, setBatchNewClassId] = useState(0);
  const [batchMinArea, setBatchMinArea] = useState(0);
  const [batchMaxArea, setBatchMaxArea] = useState(0.1);
  const [batchCurrentOnly, setBatchCurrentOnly] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResult, setBatchResult] = useState<string | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  // ── Sidebar tab state ──
  const [sidebarTab, setSidebarTab] = useState<'annotations' | 'prelabel' | 'batch'>('annotations');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageUrlRef = useRef('');
  const topRulerRef = useRef<HTMLCanvasElement>(null);
  const leftRulerRef = useRef<HTMLCanvasElement>(null);
  const rulerSize = 26;
  const displaySizeRef = useRef({ w: 0, h: 0 });
  const imageInfoRef = useRef({ width: 0, height: 0, annotation_count: 0, image_id: '', storage_path: '', camera: '', capture_time: '', source: '' });

  // Load dataset info
  useEffect(() => {
    if (!name) return;
    fetchDataset(name).then(setDataset).catch((e) => console.error('Failed to fetch dataset info:', e));
    loadBrowseData();
    loadClasses();
    loadReviewStatus();
  }, [name, version]);

  const loadBrowseData = async () => {
    if (!name || !version) return;
    try {
      const data = await api.get<BrowseResult>(`/datasets/${name}/versions/${version}/browse`);
      setBrowseData(data);
      setLoading(false);
      if (data.images.length > 0) {
        loadAnnotations(0);
      }
    } catch (e) {
      console.error('Failed to load browse data', e);
      setLoading(false);
    }
  };

  const loadAnnotations = async (idx: number) => {
    if (!name || !version) return;
    try {
      const data = await api.get<{ boxes: YoloBox[] }>(`/datasets/${name}/versions/${version}/annotations/${idx}`);
      setAnnotations(data.boxes || []);
      setSelectedBoxIdx(null);
      setImageLoaded(false);
      setDirty(false);
    } catch (e) {
      console.error('Failed to load annotations', e);
      setAnnotations([]);
    }
  };

  const loadClasses = async () => {
    if (!name || !version) return;
    try {
      const data = await api.get<{ classes: { id: number; name: string }[] }>(`/datasets/${name}/versions/${version}/classes`);
      setAllClasses(data.classes || []);
    } catch (e) {
      console.error('Failed to load classes', e);
    }
  };

  const loadReviewStatus = async () => {
    if (!name || !version) return;
    try {
      const data = await api.get<ReviewStatusResult>(`/datasets/${name}/versions/${version}/review-status`);
      setReviewData(data);
    } catch (e) {
      console.error('Failed to load review status', e);
    }
  };

  const updateCurrentReviewStatus = useCallback(() => {
    if (reviewData && currentIdx < reviewData.images.length) {
      setCurrentReviewStatus(reviewData.images[currentIdx]?.status || 'pending');
    }
  }, [reviewData, currentIdx]);

  useEffect(() => {
    updateCurrentReviewStatus();
  }, [updateCurrentReviewStatus]);

  // When index changes, reload annotations
  const goToImage = useCallback((idx: number) => {
    if (!browseData) return;
    // Intercept if there are unsaved changes
    if (dirty && !window.confirm(t('annotationEditor.unsavedConfirm'))) return;
    const clamped = Math.max(0, Math.min(idx, browseData.images.length - 1));
    setCurrentIdx(clamped);
    loadAnnotations(clamped);
    setZoom(1);
    setImageLoaded(false);
    setShowInfo(false);
  }, [browseData, name, version, dirty, t]);

  // Draw on canvas
  // ── Draw ruler bars with tick marks ──
  const drawRulers = useCallback((dispW: number, dispH: number) => {
    const topCanvas = topRulerRef.current;
    const leftCanvas = leftRulerRef.current;
    if (!topCanvas || !leftCanvas) return;

    const imgInfo = imageInfoRef.current;
    if (!imgInfo.width) return;
    const origW = imgInfo.width || 1;
    const origH = imgInfo.height || 1;
    const scaleX = dispW / origW;
    const scaleY = dispH / origH;

    // Determine tick interval (responsive to zoom)
    const pixelPerTick = 50; // base interval in original pixels
    const rawPx = pixelPerTick * scaleX;
    let nicePx = 50 * scaleX;
    if (rawPx < 30) nicePx = 25 * scaleX;
    else if (rawPx < 60) nicePx = 50 * scaleX;
    else if (rawPx < 120) nicePx = 100 * scaleX;
    else if (rawPx < 250) nicePx = 200 * scaleX;
    else nicePx = 500 * scaleX;
    nicePx = Math.max(15, nicePx);

    // ── Draw top ruler ──
    const rs = rulerSize;
    topCanvas.width = dispW;
    topCanvas.height = rs;
    const tCtx = topCanvas.getContext('2d');
    if (!tCtx) return;
    tCtx.fillStyle = '#2d2d2d';
    tCtx.fillRect(0, 0, dispW, rs);
    tCtx.strokeStyle = '#555';
    tCtx.lineWidth = 0.5;
    // Bottom border
    tCtx.beginPath(); tCtx.moveTo(0, rs - 0.5); tCtx.lineTo(dispW, rs - 0.5); tCtx.stroke();

    let tickPx = 0;
    while (tickPx < dispW) {
      const isMajor = Math.round(tickPx / nicePx) % 5 === 0;
      const tickH = isMajor ? rs - 4 : rs - 10;
      tCtx.strokeStyle = isMajor ? '#aaa' : '#666';
      tCtx.lineWidth = 1;
      tCtx.beginPath(); tCtx.moveTo(tickPx, rs); tCtx.lineTo(tickPx, tickH); tCtx.stroke();
      if (isMajor) {
        const label = Math.round(tickPx / scaleX);
        tCtx.fillStyle = '#aaa';
        tCtx.font = '9px monospace';
        tCtx.fillText(String(label), tickPx + 2, rs - 6);
      }
      tickPx += nicePx;
    }

    // ── Draw left ruler ──
    leftCanvas.width = rs;
    leftCanvas.height = dispH;
    const lCtx = leftCanvas.getContext('2d');
    if (!lCtx) return;
    lCtx.fillStyle = '#2d2d2d';
    lCtx.fillRect(0, 0, rs, dispH);
    lCtx.strokeStyle = '#555';
    lCtx.lineWidth = 0.5;
    lCtx.beginPath(); lCtx.moveTo(rs - 0.5, 0); lCtx.lineTo(rs - 0.5, dispH); lCtx.stroke();

    tickPx = 0;
    while (tickPx < dispH) {
      const isMajor = Math.round(tickPx / nicePx) % 5 === 0;
      const tickW = isMajor ? rs - 4 : rs - 10;
      lCtx.strokeStyle = isMajor ? '#aaa' : '#666';
      lCtx.lineWidth = 1;
      lCtx.beginPath(); lCtx.moveTo(rs, tickPx); lCtx.lineTo(tickW, tickPx); lCtx.stroke();
      if (isMajor) {
        const label = Math.round(tickPx / scaleY);
        lCtx.fillStyle = '#aaa';
        lCtx.font = '9px monospace';
        lCtx.save();
        lCtx.translate(rs - 4, tickPx + 3);
        lCtx.textAlign = 'right';
        lCtx.fillText(String(label), 0, 0);
        lCtx.restore();
      }
      tickPx += nicePx;
    }

    // ── Ruler indicators for selected/drawn box ──
    let guideBox: { x: number; y: number; w: number; h: number } | null = null;
    if (selectedBoxIdx !== null && selectedBoxIdx < annotations.length) {
      const box = annotations[selectedBoxIdx];
      const px = box.xc * dispW; const py = box.yc * dispH;
      const pw = box.w * dispW; const ph = box.h * dispH;
      guideBox = { x: px - pw / 2, y: py - ph / 2, w: pw, h: ph };
    } else if (isDrawing && drawStart && drawEnd) {
      guideBox = {
        x: Math.min(drawStart.x, drawEnd.x),
        y: Math.min(drawStart.y, drawEnd.y),
        w: Math.abs(drawEnd.x - drawStart.x),
        h: Math.abs(drawEnd.y - drawStart.y),
      };
    }

    if (guideBox) {
      const { x, y, w, h } = guideBox;
      const gx = Math.max(0, Math.min(x, dispW));
      const gy = Math.max(0, Math.min(y, dispH));
      const gx2 = Math.max(0, Math.min(x + w, dispW));
      const gy2 = Math.max(0, Math.min(y + h, dispH));

      // Top ruler indicators
      tCtx.fillStyle = 'rgba(255,220,50,0.25)';
      tCtx.fillRect(gx, 0, gx2 - gx, rs);
      tCtx.strokeStyle = '#ffdc32';
      tCtx.lineWidth = 2;
      tCtx.beginPath(); tCtx.moveTo(gx, 0); tCtx.lineTo(gx, rs); tCtx.stroke();
      tCtx.beginPath(); tCtx.moveTo(gx2, 0); tCtx.lineTo(gx2, rs); tCtx.stroke();
      // Labels on top ruler
      const px1 = Math.round(gx / scaleX);
      const px2 = Math.round(gx2 / scaleX);
      tCtx.fillStyle = '#ffdc32';
      tCtx.font = 'bold 9px monospace';
      const lbl1 = String(px1);
      const lbl2 = String(px2);
      const tw1 = tCtx.measureText(lbl1).width;
      const tw2 = tCtx.measureText(lbl2).width;
      tCtx.fillText(lbl1, Math.max(2, Math.min(gx - tw1 - 2, dispW - tw1 - 2)), rs - 6);
      tCtx.fillText(lbl2, Math.max(2, Math.min(gx2 - tw2 - 2, dispW - tw2 - 2)), rs - 6);

      // Left ruler indicators
      lCtx.fillStyle = 'rgba(255,220,50,0.25)';
      lCtx.fillRect(0, gy, rs, gy2 - gy);
      lCtx.strokeStyle = '#ffdc32';
      lCtx.lineWidth = 2;
      lCtx.beginPath(); lCtx.moveTo(0, gy); lCtx.lineTo(rs, gy); lCtx.stroke();
      lCtx.beginPath(); lCtx.moveTo(0, gy2); lCtx.lineTo(rs, gy2); lCtx.stroke();
      // Labels on left ruler
      const py1 = Math.round(gy / scaleY);
      const py2 = Math.round(gy2 / scaleY);
      lCtx.fillStyle = '#ffdc32';
      lCtx.font = 'bold 9px monospace';
      lCtx.save();
      lCtx.translate(rs - 4, gy);
      lCtx.textAlign = 'right';
      lCtx.fillText(String(py1), 0, -2);
      lCtx.translate(0, gy2 - gy);
      lCtx.fillText(String(py2), 0, -2);
      lCtx.restore();
    }
  }, [annotations, selectedBoxIdx, isDrawing, drawStart, drawEnd, rulerSize]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.complete) return;

    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const maxW = rect.width - 0;
    const maxH = rect.height - 0;

    // Calculate display size maintaining aspect ratio
    const imgAspect = img.naturalWidth / img.naturalHeight;
    let displayW: number, displayH: number;
    if (imgAspect > maxW / maxH) {
      displayW = maxW;
      displayH = maxW / imgAspect;
    } else {
      displayH = maxH;
      displayW = maxH * imgAspect;
    }
    displayW *= zoom;
    displayH *= zoom;

    canvas.width = displayW;
    canvas.height = displayH;
    displaySizeRef.current = { w: displayW, h: displayH };
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image on canvas
    ctx.drawImage(img, 0, 0, displayW, displayH);
    drawRulers(displayW, displayH);

    // Draw existing boxes
    for (let i = 0; i < annotations.length; i++) {
      const box = annotations[i];
      const px = box.xc * displayW;
      const py = box.yc * displayH;
      const pw = box.w * displayW;
      const ph = box.h * displayH;
      const x = px - pw / 2;
      const y = py - ph / 2;

      const isSelected = i === selectedBoxIdx;
      ctx.strokeStyle = getClassColor(box.cls_id);
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.setLineDash(isSelected ? [6, 3] : []);
      ctx.strokeRect(x, y, pw, ph);
      ctx.setLineDash([]);

      // Label background
      const label = `cls_${box.cls_id}`;
      ctx.fillStyle = getClassColor(box.cls_id);
      const fm = ctx.measureText(label);
      ctx.fillRect(x, y - 18, fm.width + 6, 18);
      ctx.fillStyle = '#fff';
      ctx.font = '12px monospace';
      ctx.fillText(label, x + 3, y - 5);

      // Corner handles for selected box
      if (isSelected) {
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = getClassColor(box.cls_id);
        ctx.lineWidth = 2;
        const handleSize = 8;
        const corners = [
          [x, y], [x + pw, y],
          [x, y + ph], [x + pw, y + ph],
        ];
        for (const [cx, cy] of corners) {
          ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
          ctx.strokeRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
        }
      }
    }

    // Draw ruler guide lines for selected or in-progress box
    let guideBox: { x: number; y: number; w: number; h: number } | null = null;

    if (selectedBoxIdx !== null && selectedBoxIdx < annotations.length) {
      const box = annotations[selectedBoxIdx];
      const px = box.xc * displayW;
      const py = box.yc * displayH;
      const pw = box.w * displayW;
      const ph = box.h * displayH;
      guideBox = { x: px - pw / 2, y: py - ph / 2, w: pw, h: ph };
    } else if (isDrawing && drawStart && drawEnd) {
      const x = Math.min(drawStart.x, drawEnd.x);
      const y = Math.min(drawStart.y, drawEnd.y);
      const w = Math.abs(drawEnd.x - drawStart.x);
      const h = Math.abs(drawEnd.y - drawStart.y);
      guideBox = { x, y, w, h };
    }

    if (guideBox) {
      const { x, y, w, h } = guideBox;
      const gx = Math.max(0, x);
      const gy = Math.max(0, y);
      const gx2 = Math.min(displayW, x + w);
      const gy2 = Math.min(displayH, y + h);

      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,100,0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);

      // Horizontal guide lines → top ruler
      ctx.beginPath();
      ctx.moveTo(gx, 0); ctx.lineTo(gx, gy);
      ctx.moveTo(gx2, 0); ctx.lineTo(gx2, gy);
      ctx.stroke();

      // Vertical guide lines → left ruler
      ctx.beginPath();
      ctx.moveTo(0, gy); ctx.moveTo(gx, gy);
      ctx.moveTo(0, gy2); ctx.moveTo(gx, gy2);
      ctx.stroke();

      ctx.setLineDash([]);

      // Dimension label (W × H in original pixels)
      if (currentImage) {
        const pixelW = Math.round(w / displayW * currentImage.width);
        const pixelH = Math.round(h / displayH * currentImage.height);
        const dimLabel = `${pixelW} × ${pixelH}`;
        const labelX = Math.min(gx + 6, displayW - 80);
        const labelY = Math.max(gy - 8, 20);
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        const tw = ctx.measureText(dimLabel).width;
        ctx.fillRect(labelX - 3, labelY - 12, tw + 8, 18);
        ctx.fillStyle = '#fff';
        ctx.font = '11px monospace';
        ctx.fillText(dimLabel, labelX + 1, labelY + 1);
      }
      ctx.restore();
    }

    // Draw in-progress box
    if (isDrawing && drawStart && drawEnd) {
      ctx.strokeStyle = getClassColor(currentClassId);
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      const x = Math.min(drawStart.x, drawEnd.x);
      const y = Math.min(drawStart.y, drawEnd.y);
      const w = Math.abs(drawEnd.x - drawStart.x);
      const h = Math.abs(drawEnd.y - drawStart.y);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }, [annotations, selectedBoxIdx, isDrawing, drawStart, drawEnd, zoom, currentClassId]);

  // Redraw when state changes
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas, drawRulers]);

  // Keep imageInfoRef in sync
  useEffect(() => {
    if (browseData && browseData.images[currentIdx]) imageInfoRef.current = browseData.images[currentIdx];
  }, [currentIdx, browseData]);

  // ── Pointer handlers for canvas (unified mouse/touch/pen via Pointer Events API) ──

  const getCanvasPos = (e: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number } => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    // Capture pointer for smooth dragging even outside canvas bounds
    e.currentTarget.setPointerCapture(e.pointerId);

    const pos = getCanvasPos(e);

    // Check if clicking on a selected box handle
    if (selectedBoxIdx !== null && selectedBoxIdx < annotations.length) {
      const box = annotations[selectedBoxIdx];
      const pw = box.w * canvasRef.current!.width;
      const ph = box.h * canvasRef.current!.height;
      const bx = box.xc * canvasRef.current!.width - pw / 2;
      const by = box.yc * canvasRef.current!.height - ph / 2;

      const handleSize = 10;
      const resizeMode = getResizeMode(pos, bx, by, pw, ph, handleSize);
      if (resizeMode !== 'none') {
        setDragMode(resizeMode as any);
        setDragStartPos(pos);
        setDragStartBox({ ...box });
        return;
      }

      // Check if clicking inside selected box (move)
      if (pos.x >= bx && pos.x <= bx + pw && pos.y >= by && pos.y <= by + ph) {
        setDragMode('move');
        setDragStartPos(pos);
        setDragStartBox({ ...box });
        return;
      }
    }

    // Check if clicking on any box (select it)
    for (let i = annotations.length - 1; i >= 0; i--) {
      const box = annotations[i];
      const pw = box.w * canvasRef.current!.width;
      const ph = box.h * canvasRef.current!.height;
      const bx = box.xc * canvasRef.current!.width - pw / 2;
      const by = box.yc * canvasRef.current!.height - ph / 2;
      if (pos.x >= bx && pos.x <= bx + pw && pos.y >= by && pos.y <= by + ph) {
        setSelectedBoxIdx(i);
        return;
      }
    }

    // Start drawing a new box
    setSelectedBoxIdx(null);
    setIsDrawing(true);
    setDrawStart(pos);
    setDrawEnd(pos);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const pos = getCanvasPos(e);

    if (isDrawing) {
      setDrawEnd(pos);
      return;
    }

    if (dragMode !== 'none' && dragStartPos && dragStartBox && selectedBoxIdx !== null) {
      const dx = (pos.x - dragStartPos.x) / canvasRef.current!.width;
      const dy = (pos.y - dragStartPos.y) / canvasRef.current!.height;

      const newBoxes = [...annotations];
      const box = { ...dragStartBox };

      if (dragMode === 'move') {
        box.xc = dragStartBox.xc + dx;
        box.yc = dragStartBox.yc + dy;
        // Clamp to [0,1]
        box.xc = Math.max(box.w / 2, Math.min(1 - box.w / 2, box.xc));
        box.yc = Math.max(box.h / 2, Math.min(1 - box.h / 2, box.yc));
      } else {
        // Resize
        const x1 = dragStartBox.xc - dragStartBox.w / 2;
        const y1 = dragStartBox.yc - dragStartBox.h / 2;
        const x2 = dragStartBox.xc + dragStartBox.w / 2;
        const y2 = dragStartBox.yc + dragStartBox.h / 2;

        let nx1 = x1, ny1 = y1, nx2 = x2, ny2 = y2;

        if (dragMode.includes('l')) nx1 = x1 + dx;
        if (dragMode.includes('r')) nx2 = x2 + dx;
        if (dragMode.includes('t')) ny1 = y1 + dy;
        if (dragMode.includes('b')) ny2 = y2 + dy;

        // Ensure minimum size
        if (nx2 - nx1 < 0.01) return;
        if (ny2 - ny1 < 0.01) return;

        box.xc = (nx1 + nx2) / 2;
        box.yc = (ny1 + ny2) / 2;
        box.w = nx2 - nx1;
        box.h = ny2 - ny1;

        // Clamp
        box.xc = Math.max(box.w / 2, Math.min(1 - box.w / 2, box.xc));
        box.yc = Math.max(box.h / 2, Math.min(1 - box.h / 2, box.yc));
        box.w = Math.max(0.01, Math.min(1, box.w));
        box.h = Math.max(0.01, Math.min(1, box.h));
      }

      newBoxes[selectedBoxIdx!] = box;
      setAnnotations(newBoxes);
      setDirty(true);
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    // Release pointer capture
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    if (isDrawing && drawStart && drawEnd) {
      const canvas = canvasRef.current!;
      const x = Math.min(drawStart.x, drawEnd.x) / canvas.width;
      const y = Math.min(drawStart.y, drawEnd.y) / canvas.height;
      const w = Math.abs(drawEnd.x - drawStart.x) / canvas.width;
      const h = Math.abs(drawEnd.y - drawStart.y) / canvas.height;

      if (w > 0.01 && h > 0.01) {
        const newBox: YoloBox = {
          cls_id: currentClassId,
          xc: x + w / 2,
          yc: y + h / 2,
          w,
          h,
        };
        const newAnnotations = [...annotations, newBox];
        setAnnotations(newAnnotations);
        setSelectedBoxIdx(newAnnotations.length - 1);
        setDirty(true);
      }
    }

    setIsDrawing(false);
    setDrawStart(null);
    setDrawEnd(null);
    setDragMode('none');
    setDragStartPos(null);
    setDragStartBox(null);
  };

  // Get resize mode based on cursor position near box corners/edges
  const getResizeMode = (pos: { x: number; y: number }, bx: number, by: number, bw: number, bh: number, handleSize: number): string => {
    const near = (p: number, t: number) => Math.abs(p - t) < handleSize;

    const onLeft = near(pos.x, bx);
    const onRight = near(pos.x, bx + bw);
    const onTop = near(pos.y, by);
    const onBottom = near(pos.y, by + bh);

    if (onLeft && onTop) return 'resize-tl';
    if (onRight && onTop) return 'resize-tr';
    if (onLeft && onBottom) return 'resize-bl';
    if (onRight && onBottom) return 'resize-br';
    if (onTop) return 'resize-t';
    if (onBottom) return 'resize-b';
    if (onLeft) return 'resize-l';
    if (onRight) return 'resize-r';
    return 'none';
  };

  // ── Keyboard handler ──

  // Check if the current focus is in an input/textarea/contenteditable
  const isInputFocused = (target: EventTarget | null): boolean => {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName.toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (isInputFocused(e.target)) {
        // Allow Ctrl+S even in inputs
        if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          handleSave();
        }
        return;
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedBoxIdx !== null) {
          e.preventDefault();
          const newAnnotations = annotations.filter((_, i) => i !== selectedBoxIdx);
          setAnnotations(newAnnotations);
          setSelectedBoxIdx(null);
          setDirty(true);
        }
      }
      if (e.key === 'ArrowLeft') goToImage(currentIdx - 1);
      if (e.key === 'ArrowRight') goToImage(currentIdx + 1);
      if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSave();
      }
      if (e.key === 'Escape') setSelectedBoxIdx(null);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedBoxIdx, annotations, currentIdx, dirty]);

  // Intercept page refresh/close when there are unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [dirty]);

  // ── Save ──

  const handleSave = async () => {
    if (!name || !version) return;
    setSaving(true);
    try {
      await api.put(`/datasets/${name}/versions/${version}/annotations/${currentIdx}`, { boxes: annotations });
      setDirty(false);
    } catch (e) {
      console.error('Failed to save annotations', e);
    }
    setSaving(false);
  };

  // ── Delete selected ──

  const handleDelete = () => {
    if (selectedBoxIdx === null) return;
    const newAnnotations = annotations.filter((_, i) => i !== selectedBoxIdx);
    setAnnotations(newAnnotations);
    setSelectedBoxIdx(null);
    setDirty(true);
  };

  // ── Change class for selected box ──

  const handleChangeClass = (newClsId: number) => {
    if (selectedBoxIdx === null) return;
    const newAnnotations = [...annotations];
    newAnnotations[selectedBoxIdx] = { ...newAnnotations[selectedBoxIdx], cls_id: newClsId };
    setAnnotations(newAnnotations);
    setDirty(true);
  };

  // ── AI Pre-labeling ──

  const handlePrelabel = async () => {
    if (!name || !version || !prelabelModel) return;
    setPrelabelRunning(true);
    setPrelabelError(null);
    setPrelabelResult(null);
    try {
      const indices = prelabelCurrentOnly ? [currentIdx] : undefined;
      const result = await prelabelImages(name, version, {
        model_path: prelabelModel,
        conf: prelabelConf,
        iou: prelabelIou,
        imgsz: prelabelImgsz,
        indices,
        overwrite: prelabelOverwrite,
      });
      setPrelabelResult(
        t('annotationEditor.prelabelResult', {
          labeled: result.labeled, skipped: result.skipped, totalBoxes: result.total_boxes,
        }),
      );
      // Reload annotations for current image if it was labeled
      if (!prelabelCurrentOnly || result.labeled > 0) {
        loadAnnotations(currentIdx);
        loadBrowseData();
      }
    } catch (e) {
      setPrelabelError(t('annotationEditor.prelabelError', { error: String(e).slice(0, 100) }));
    }
    setPrelabelRunning(false);
  };

  // ── Review status ──

  const handleSetReviewStatus = async (status: ReviewStatusType) => {
    if (!name || !version) return;
    try {
      const result = await setReviewStatus(name, version, currentIdx, status);
      setCurrentReviewStatus(status);
      // Update reviewData summary
      if (reviewData) {
        const newImages = [...reviewData.images];
        if (newImages[currentIdx]) {
          newImages[currentIdx] = { ...newImages[currentIdx], status };
        }
        setReviewData({ ...reviewData, summary: result.summary, images: newImages });
      }
    } catch (e) {
      console.error('Failed to set review status', e);
    }
  };

  const handleBatchApproveAll = async () => {
    if (!name || !version || !reviewData) return;
    try {
      const allIndices = reviewData.images.map(img => img.index);
      const result = await batchSetReviewStatus(name, version, allIndices, 'approved');
      setReviewData({ ...reviewData, summary: result.summary });
      setCurrentReviewStatus('approved');
    } catch (e) {
      console.error('Failed to batch approve', e);
    }
  };

  // ── Batch annotation operations ──

  const handleBatchOp = async () => {
    if (!name || !version) return;
    if (!window.confirm(t('annotationEditor.batchConfirm'))) return;
    setBatchRunning(true);
    setBatchError(null);
    setBatchResult(null);
    try {
      const indices = batchCurrentOnly ? [currentIdx] : undefined;
      const params: Record<string, unknown> = {
        operation: batchOp,
        indices,
      };
      if (batchOp === 'delete_by_class' || batchOp === 'reassign_class') {
        params.class_id = batchClassId;
      }
      if (batchOp === 'delete_by_area') {
        params.min_area = batchMinArea || undefined;
        params.max_area = batchMaxArea || undefined;
      }
      if (batchOp === 'reassign_class') {
        params.new_class_id = batchNewClassId;
      }
      const result = await batchAnnotationOperation(name, version, params as any);
      setBatchResult(
        t('annotationEditor.batchResult', {
          processed: result.images_processed,
          removed: result.boxes_removed,
          modified: result.boxes_modified,
        }),
      );
      // Reload current image annotations and browse data
      loadAnnotations(currentIdx);
      loadBrowseData();
    } catch (e) {
      setBatchError(t('annotationEditor.batchError', { error: String(e).slice(0, 100) }));
    }
    setBatchRunning(false);
  };

  // ── Render ──

  const currentImage = browseData?.images[currentIdx];

  if (loading) {
    return (
      <div className="p-6">
        <LoadingState />
      </div>
    );
  }

  if (!browseData || browseData.images.length === 0) {
    return (
      <div className="p-6">
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
          <Link to="/datasets" className="hover:text-blue-600">{t('datasets.title')}</Link>
          <span>/</span>
          <span className="text-gray-900">{version}</span>
        </div>
        <EmptyState message={t('annotationEditor.noImages')} />
      </div>
    );
  }

  const imgUrl = `${API_BASE}/datasets/${name}/versions/${version}/image/${currentIdx}`;

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2 text-sm">
          <Link to="/datasets" className="text-gray-500 hover:text-blue-600">{t('datasets.title')}</Link>
          <span className="text-gray-300">/</span>
          <Link to={`/datasets/${name}`} className="text-gray-500 hover:text-blue-600">{dataset?.display_name || name}</Link>
          <span className="text-gray-300">/</span>
          <span className="text-gray-900 font-medium">{t('datasetDetail.browse')}</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(z => Math.max(0.25, z - 0.25))}
            className="p-1.5 text-gray-500 hover:bg-gray-100 rounded"
            title={t('annotationEditor.zoomOut')}
            aria-label={t('annotationEditor.zoomOut')}
          ><ZoomOut className="w-4 h-4" /></button>
          <span className="text-xs text-gray-500 w-10 text-center">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoom(z => Math.min(4, z + 0.25))}
            className="p-1.5 text-gray-500 hover:bg-gray-100 rounded"
            title={t('annotationEditor.zoomIn')}
            aria-label={t('annotationEditor.zoomIn')}
          ><ZoomIn className="w-4 h-4" /></button>
          <div className="w-px h-5 bg-gray-200 mx-1" />
          <button
            onClick={() => setShowInfo(!showInfo)}
            className={`p-1.5 rounded ${showInfo ? 'bg-blue-100 text-blue-600' : 'text-gray-500 hover:bg-gray-100'}`}
            title={t('datasetDetail.imageInfo')}
            aria-label={t('datasetDetail.imageInfo')}
            aria-expanded={showInfo}
          ><Info className="w-4 h-4" /></button>
          <div className="w-px h-5 bg-gray-200 mx-1" />
          {dirty && <span className="text-xs text-amber-600 font-medium">{t('annotationEditor.unsaved')}</span>}
          <button
            onClick={handleSave}
            disabled={saving || !dirty}
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            <Save className="w-4 h-4" />
            {saving ? t('annotationEditor.saving') : t('common.save')}
          </button>
        </div>
      </div>

      {/* Main content area */}
      <div className="flex flex-1 min-h-0">
        {/* Image & canvas container */}
        <div className="flex-1 flex flex-col min-w-0 bg-gray-900 relative">
          {/* Navigation bar on top of image */}
          <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-3 py-2 bg-gradient-to-b from-black/60 to-transparent">
            <div className="flex items-center gap-2 text-white text-sm">
              <button
                onClick={() => goToImage(currentIdx - 1)}
                disabled={currentIdx <= 0}
                className="p-1 hover:bg-white/20 rounded disabled:opacity-30"
                aria-label={t('common.previousImage')}
              ><ChevronLeft className="w-5 h-5" /></button>
              <span className="font-mono tabular-nums">
                {currentIdx + 1} / {browseData.total}
              </span>
              <button
                onClick={() => goToImage(currentIdx + 1)}
                disabled={currentIdx >= browseData.total - 1}
                className="p-1 hover:bg-white/20 rounded disabled:opacity-30"
                aria-label={t('common.nextImage')}
              ><ChevronRight className="w-5 h-5" /></button>
            </div>
            {currentImage && (
              <div className="flex items-center gap-3 text-white/70 text-xs">
                <span>{currentImage.annotation_count} {t('annotationEditor.annotations')}</span>
                {currentImage.camera && <span>&middot; {currentImage.camera}</span>}
                {currentReviewStatus === 'approved' && (
                  <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/20 text-green-300">
                    <CheckCircle2 className="w-3 h-3" /> {t('annotationEditor.reviewApproved')}
                  </span>
                )}
                {currentReviewStatus === 'rejected' && (
                  <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/20 text-red-300">
                    <XCircle className="w-3 h-3" /> {t('annotationEditor.reviewRejected')}
                  </span>
                )}
                {currentReviewStatus === 'pending' && (
                  <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-300">
                    <Clock className="w-3 h-3" /> {t('annotationEditor.reviewPending')}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Image info overlay */}
          <ImageInfoOverlay show={showInfo} image={currentImage} />

          {/* Image + canvas */}
          <div ref={containerRef} className="flex-1 flex items-center justify-center overflow-hidden">
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              {/* Corner + top ruler row */}
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <div style={{ width: rulerSize, height: rulerSize, backgroundColor: '#2d2d2d', borderBottom: '0.5px solid #555', borderRight: '0.5px solid #555', flexShrink: 0 }} />
                <canvas ref={topRulerRef} style={{ display: 'block', flexShrink: 0 }} />
              </div>
              {/* Left ruler + canvas row */}
              <div style={{ display: 'flex' }}>
                <canvas ref={leftRulerRef} style={{ display: 'block', flexShrink: 0 }} />
                <div className="relative" style={{ maxWidth: '100%', maxHeight: '100%' }}>
                  <img
                    ref={imageRef}
                    src={imgUrl}
                    alt={t('annotationEditor.imageAlt', { index: currentIdx })}
                    onLoad={() => { setImageLoaded(true); }}
                    style={{
                      display: 'block',
                      maxWidth: `${100 / zoom}%`,
                      maxHeight: `${100 / zoom}%`,
                      visibility: 'hidden',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                    }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="cursor-crosshair"
                    style={{ touchAction: 'none' }}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerLeave={() => {
                      if (isDrawing) {
                        setIsDrawing(false);
                        setDrawStart(null);
                        setDrawEnd(null);
                      }
                      if (dragMode !== 'none') {
                        setDragMode('none');
                        setDragStartPos(null);
                        setDragStartBox(null);
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Thumbnail strip */}
          <ThumbnailStrip
            images={browseData.images}
            currentIdx={currentIdx}
            onSelect={goToImage}
            name={name || ''}
            version={version || ''}
          />
        </div>

        {/* Sidebar */}
        <AnnotationSidebar
          sidebarTab={sidebarTab}
          setSidebarTab={setSidebarTab}
          reviewData={reviewData}
          currentReviewStatus={currentReviewStatus}
          handleSetReviewStatus={handleSetReviewStatus}
          handleBatchApproveAll={handleBatchApproveAll}
          annotations={annotations}
          selectedBoxIdx={selectedBoxIdx}
          setSelectedBoxIdx={setSelectedBoxIdx}
          currentImage={currentImage}
          allClasses={allClasses}
          handleChangeClass={handleChangeClass}
          handleDelete={handleDelete}
          currentClassId={currentClassId}
          setCurrentClassId={setCurrentClassId}
          prelabelModel={prelabelModel}
          setPrelabelModel={setPrelabelModel}
          prelabelConf={prelabelConf}
          setPrelabelConf={setPrelabelConf}
          prelabelIou={prelabelIou}
          setPrelabelIou={setPrelabelIou}
          prelabelImgsz={prelabelImgsz}
          setPrelabelImgsz={setPrelabelImgsz}
          prelabelOverwrite={prelabelOverwrite}
          setPrelabelOverwrite={setPrelabelOverwrite}
          prelabelCurrentOnly={prelabelCurrentOnly}
          setPrelabelCurrentOnly={setPrelabelCurrentOnly}
          prelabelRunning={prelabelRunning}
          prelabelResult={prelabelResult}
          prelabelError={prelabelError}
          handlePrelabel={handlePrelabel}
          batchOp={batchOp}
          setBatchOp={setBatchOp}
          batchClassId={batchClassId}
          setBatchClassId={setBatchClassId}
          batchNewClassId={batchNewClassId}
          setBatchNewClassId={setBatchNewClassId}
          batchMinArea={batchMinArea}
          setBatchMinArea={setBatchMinArea}
          batchMaxArea={batchMaxArea}
          setBatchMaxArea={setBatchMaxArea}
          batchCurrentOnly={batchCurrentOnly}
          setBatchCurrentOnly={setBatchCurrentOnly}
          batchRunning={batchRunning}
          batchResult={batchResult}
          batchError={batchError}
          handleBatchOp={handleBatchOp}
        />
      </div>
    </div>
  );
}
