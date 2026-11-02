import React, { useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { fetchReports } from '../api';
import { api } from '../api/client';
import type { ReportItem } from '../types';
import {
  FileText, ExternalLink, Download, X, Search,
  File as FileIcon, BarChart3, Activity, FileSpreadsheet,
} from 'lucide-react';

const REPORT_ICONS: Record<string, React.ReactNode> = {
  train: <BarChart3 className="w-4 h-4" />,
  eval: <Activity className="w-4 h-4" />,
  intelligence: <FileSpreadsheet className="w-4 h-4" />,
  ingestion: <FileText className="w-4 h-4" />,
  health: <Activity className="w-4 h-4" />,
  export: <Download className="w-4 h-4" />,
};

const GROUP_LABELS: Record<string, string> = {
  train: 'reports.groupTraining',
  eval: 'reports.groupEvaluation',
  intelligence: 'reports.groupIntelligence',
  ingestion: 'reports.groupIngestion',
  export: 'reports.groupExport',
  health: 'reports.groupHealth',
};

export default function ReportsPage() {
  const { t } = useTranslation();
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [filter, setFilter] = useState('');
  const [selectedReport, setSelectedReport] = useState<ReportItem | null>(null);
  const [htmlContent, setHtmlContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    fetchReports().then(setReports).catch((e) => console.error('Failed to load reports:', e));
  }, []);

  const openReport = async (report: ReportItem) => {
    setSelectedReport(report);
    setLoading(true);
    setHtmlContent(null);
    try {
      const data = await api.get<{ html?: string }>('/reports/' + report.id);
      if (data.html) {
        setHtmlContent(data.html);
      }
    } catch (e) {
      console.error('Failed to load report content for', report.id, e);
    }
    setLoading(false);
  };

  const openInNewTab = (report: ReportItem) => {
    const w = window.open('', '_blank');
    if (w && htmlContent && selectedReport?.id === report.id) {
      w.document.write(htmlContent);
      w.document.close();
    } else {
      window.open('/api/v1/reports/' + report.id, '_blank');
    }
  };

  const filtered = reports.filter((r) =>
    r.name.toLowerCase().includes(filter.toLowerCase()) ||
    r.type.toLowerCase().includes(filter.toLowerCase())
  );

  const groupByType = (items: ReportItem[]): Record<string, ReportItem[]> => {
    const groups: Record<string, ReportItem[]> = {};
    for (const item of items) {
      const key = item.type || 'other';
      if (!groups[key]) groups[key] = [];
      groups[key].push(item);
    }
    return groups;
  };

  const grouped = groupByType(filtered);

  return (
    <div className="p-6 max-w-6xl h-full flex flex-col">
      <div className="flex items-center justify-between mb-6 shrink-0">
        <h1 className="text-xl font-semibold">{t('reports.title')}</h1>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder={t('reports.searchPlaceholder')}
            className="pl-9 pr-3 py-1.5 border border-gray-300 rounded-lg text-sm"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
      </div>

      {selectedReport && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden mb-4 shrink-0">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b">
            <div className="flex items-center gap-2 text-sm">
              <FileIcon className="w-4 h-4 text-gray-400" />
              <span className="font-medium">{selectedReport.name}</span>
              <span className="text-gray-400 text-xs">{selectedReport.type}</span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => openInNewTab(selectedReport)}
                className="p-1.5 text-gray-400 hover:text-gray-600 rounded"
                title={t('reports.openInNewTab')}
              >
                <ExternalLink className="w-4 h-4" />
              </button>
              <button
                onClick={() => { setSelectedReport(null); setHtmlContent(null); }}
                className="p-1.5 text-gray-400 hover:text-gray-600 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 flex gap-6 min-h-0">
        <div className="w-72 shrink-0 overflow-y-auto">
          {reports.length === 0 ? (
            <div className="text-gray-400 text-sm py-6 text-center">
              <FileText className="w-8 h-8 mx-auto mb-2" />
              {t('reports.noReports')}
            </div>
          ) : (
            <div className="space-y-3">
              {Object.entries(grouped).map(([type, items]) => (
                <div key={type}>
                  <div className="text-xs font-medium text-gray-400 uppercase mb-1 px-1">
                    {t(GROUP_LABELS[type] || type)}
                  </div>
                  {items.map((r) => (
                    <button
                      key={r.id}
                      onClick={() => openReport(r)}
                      className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
                        selectedReport?.id === r.id
                          ? 'bg-blue-50 text-blue-700'
                          : 'hover:bg-gray-50 text-gray-700'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {REPORT_ICONS[r.type] || <FileText className="w-3.5 h-3.5 text-gray-400" />}
                        <span className="truncate">{r.name}</span>
                      </div>
                      <div className="text-xs text-gray-400 mt-0.5 pl-5">
                        {new Date(r.created_at * 1000).toLocaleDateString()}
                      </div>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full text-gray-400">{t('reports.loadingReport')}</div>
          ) : htmlContent ? (
            <iframe
              ref={iframeRef}
              srcDoc={htmlContent}
              className="w-full h-full border-0"
              title="Report Viewer"
              sandbox="allow-scripts"
            />
          ) : selectedReport ? (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm">
              {t('reports.contentNotAvailable')}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm">
              <div className="text-center">
                <FileText className="w-12 h-12 mx-auto mb-2 text-gray-200" />
                {t('reports.selectToView')}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
