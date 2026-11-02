import React from 'react';
import { API_BASE, type BrowseImage } from './shared';

interface ThumbnailStripProps {
  images: BrowseImage[];
  currentIdx: number;
  onSelect: (idx: number) => void;
  name: string;
  version: string;
}

/**
 * Horizontal scrollable strip of image thumbnails shown at the bottom of the
 * annotation canvas. Clicking a thumbnail navigates to that image.
 * Renders nothing when there is only a single image.
 */
export default function ThumbnailStrip({ images, currentIdx, onSelect, name, version }: ThumbnailStripProps) {
  if (images.length <= 1) return null;

  return (
    <div className="h-20 bg-gray-950 border-t border-gray-800 flex items-center gap-1 px-3 overflow-x-auto shrink-0">
      {images.map((img, i) => (
        <button
          key={i}
          onClick={() => onSelect(i)}
          className={`shrink-0 w-16 h-14 rounded overflow-hidden border-2 transition-all ${
            i === currentIdx ? 'border-blue-500 opacity-100' : 'border-transparent opacity-60 hover:opacity-90'
          }`}
        >
          <img
            src={`${API_BASE}/datasets/${name}/versions/${version}/image/${i}`}
            alt={`Thumb ${i}`}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        </button>
      ))}
    </div>
  );
}
