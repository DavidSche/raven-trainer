import React from 'react';
import { CheckCircle, Circle, GitBranch } from 'lucide-react';
import type { Version } from '../../types';

export function VersionTimeline({ versions }: { versions: Version[] }) {
  if (versions.length === 0) return <div className="text-gray-400 text-sm py-4">No versions yet</div>;

  return (
    <div className="space-y-0">
      {versions.map((v, idx) => (
        <div key={v.version} className="flex gap-3">
          <div className="flex flex-col items-center">
            <div className="p-1 rounded-full bg-blue-100 text-blue-600">
              {idx === versions.length - 1 ? (
                <CheckCircle className="w-4 h-4" />
              ) : (
                <Circle className="w-4 h-4" />
              )}
            </div>
            {idx < versions.length - 1 && <div className="w-px flex-1 bg-gray-200 my-1" />}
          </div>
          <div className="pb-4 flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium text-sm">{v.version}</span>
              <span className="text-xs text-gray-400">{v.created_at.slice(0, 10)}</span>
              {v.parent_version && (
                <span className="text-xs text-gray-400 flex items-center gap-1">
                  <GitBranch className="w-3 h-3" /> from {v.parent_version}
                </span>
              )}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {v.image_count} images &middot; {v.annotation_count} annotations &middot; source: {v.source}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
