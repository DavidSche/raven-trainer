import React from 'react';
import { useTranslation } from 'react-i18next';
import type { BrowseImage } from './shared';

interface ImageInfoOverlayProps {
  show: boolean;
  image: BrowseImage | undefined;
}

/**
 * Floating overlay panel that displays metadata for the currently selected
 * image (id, source, camera, capture time, dimensions, annotation count).
 */
export default function ImageInfoOverlay({ show, image }: ImageInfoOverlayProps) {
  const { t } = useTranslation();

  if (!show || !image) return null;

  return (
    <div className="absolute top-12 right-3 z-10 bg-black/80 text-white text-xs p-3 rounded-lg space-y-1 min-w-[200px]">
      <div className="font-medium mb-1">{t('annotationEditor.imageInfo')}</div>
      <div><span className="text-gray-400">{t('annotationEditor.fieldId')}:</span> {image.image_id}</div>
      <div><span className="text-gray-400">{t('annotationEditor.fieldSource')}:</span> {image.source}</div>
      <div><span className="text-gray-400">{t('annotationEditor.fieldCamera')}:</span> {image.camera}</div>
      <div><span className="text-gray-400">{t('annotationEditor.fieldTime')}:</span> {image.capture_time}</div>
      <div><span className="text-gray-400">{t('annotationEditor.fieldSize')}:</span> {image.width} x {image.height}</div>
      <div><span className="text-gray-400">{t('annotationEditor.fieldAnnotations')}:</span> {image.annotation_count}</div>
    </div>
  );
}
