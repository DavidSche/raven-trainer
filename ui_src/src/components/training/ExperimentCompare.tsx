import React from 'react';
import type { Experiment } from '../../types';

interface Props {
  experiments: Experiment[];
  selected: string[];
  onToggle: (id: string) => void;
}

export function ExperimentCompare({ experiments, selected, onToggle }: Props) {
  const selectedExps = experiments.filter((e) => selected.includes(e.id));

  return (
    <div>
      <div className="mb-4">
        <h3 className="text-sm font-medium mb-2">Select experiments to compare</h3>
        <div className="flex flex-wrap gap-2">
          {experiments.slice(0, 10).map((exp) => (
            <button
              key={exp.id}
              onClick={() => onToggle(exp.id)}
              className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                selected.includes(exp.id)
                  ? 'bg-blue-100 border-blue-300 text-blue-700'
                  : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300'
              }`}
            >
              {exp.id}
            </button>
          ))}
        </div>
      </div>

      {selectedExps.length >= 2 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-50">
                <th className="text-left px-3 py-2 font-medium border">Metric</th>
                {selectedExps.map((exp) => (
                  <th key={exp.id} className="text-center px-3 py-2 font-medium border">{exp.id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {['model', 'epochs', 'batch', 'imgsz', 'status'].map((field) => (
                <tr key={field}>
                  <td className="px-3 py-2 border font-medium text-gray-600 capitalize">{field}</td>
                  {selectedExps.map((exp) => (
                    <td key={exp.id} className="px-3 py-2 border text-center">
                      {String((exp as unknown as Record<string, unknown>)[field] ?? '-')}
                    </td>
                  ))}
                </tr>
              ))}
              {selectedExps.length > 0 && Object.keys(selectedExps[0].metrics || {}).length > 0 && (
                <>
                  <tr className="bg-gray-50">
                    <td colSpan={selectedExps.length + 1} className="px-3 py-2 border font-semibold text-sm">
                      Metrics
                    </td>
                  </tr>
                  {Object.keys(selectedExps[0].metrics).map((metric) => (
                    <tr key={metric}>
                      <td className="px-3 py-2 border font-medium text-gray-600">{metric}</td>
                      {selectedExps.map((exp) => (
                        <td key={exp.id} className="px-3 py-2 border text-center font-mono">
                          {exp.metrics?.[metric]?.toFixed?.(4) ?? exp.metrics?.[metric] ?? '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      )}

      {selectedExps.length < 2 && (
        <div className="text-gray-400 text-sm py-6 text-center">Select at least 2 experiments to compare</div>
      )}
    </div>
  );
}

