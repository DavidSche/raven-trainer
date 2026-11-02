import React, { useEffect, useState, useRef } from 'react';
import { Loader2, X, Wifi, WifiOff } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { fetchExperimentLog, connectTrainingWebSocket } from '../../api/training';

interface Props {
  experimentId: string;
  isRunning: boolean;
  onClose?: () => void;
}

export function TrainingProgress({ experimentId, isRunning, onClose }: Props) {
  const { t } = useTranslation();
  const [logs, setLogs] = useState<string[]>([]);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connected'>('disconnected');
  const logEndRef = useRef<HTMLDivElement>(null);

  // WebSocket connection for real-time logs when running
  useEffect(() => {
    if (!experimentId) return;

    if (isRunning) {
      const cleanup = connectTrainingWebSocket(
        experimentId,
        (message) => {
          setLogs((prev) => [...prev, message]);
        },
        (status) => {
          setWsStatus(status as 'disconnected' | 'connected');
        },
      );
      return cleanup;
    }
    return;
  }, [experimentId, isRunning]);

  // Poll for logs when not running (completed experiments)
  useEffect(() => {
    if (!experimentId || isRunning) return;

    fetchExperimentLog(experimentId).then((data) => {
      if (data.log) {
        setLogs(data.log.trim().split("\n"));
      }
    }).catch((e) => console.error('Failed to fetch experiment log:', e));
  }, [experimentId, isRunning]);

  // Auto-scroll to bottom when logs update
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 font-mono">
            {t('trainingDetail.trainingLog', { name: experimentId })}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {isRunning && (
            <span className={`flex items-center gap-1 text-xs ${wsStatus === 'connected' ? 'text-green-400' : 'text-yellow-400'}`}>
              {wsStatus === 'connected' ? (
                <><Wifi className="w-3 h-3" /> {t('trainingDetail.live')}</>
              ) : (
                <><Loader2 className="w-3 h-3 animate-spin" /> {t('trainingDetail.connecting')}</>
              )}
            </span>
          )}
          {!isRunning && (
            <span className="text-xs text-gray-500">
              <WifiOff className="w-3 h-3 inline mr-1" />
              {t('trainingDetail.lines', { count: logs.length })}
            </span>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-300 transition-colors"
              title={t('common.close')}
              aria-label={t('common.close')}
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      <div className="h-80 overflow-y-auto p-3">
        {logs.length === 0 ? (
          <div className="text-gray-500 text-xs font-mono">
            {isRunning ? t('trainingDetail.waitingForLogs') : t('trainingDetail.noLogData')}
          </div>
        ) : (
          <pre className="text-green-400 text-xs font-mono leading-relaxed">
            {logs.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            <div ref={logEndRef} />
          </pre>
        )}
      </div>
    </div>
  );
}
