import React from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, Loader2, Inbox, RefreshCw } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  className?: string;
}

export function LoadingState({ message, className = '' }: LoadingStateProps) {
  const { t } = useTranslation();
  return (
    <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
      <Loader2 className="w-6 h-6 text-blue-500 animate-spin mb-3" />
      <p className="text-sm text-gray-400">{message || t('common.loading')}</p>
    </div>
  );
}

interface ErrorStateProps {
  error?: string | null;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({ error, message, onRetry, className = '' }: ErrorStateProps) {
  const { t } = useTranslation();
  return (
    <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
      <AlertTriangle className="w-8 h-8 text-red-400 mb-3" />
      <p className="text-sm font-medium text-gray-600 mb-1">{message || t('common.loadError')}</p>
      {error && <p className="text-xs text-gray-400 mb-3 max-w-md text-center">{error}</p>}
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="w-3 h-3" /> {t('common.retry')}
        </button>
      )}
    </div>
  );
}

interface EmptyStateProps {
  message?: string;
  hint?: string;
  icon?: React.ReactNode;
  className?: string;
}

export function EmptyState({ message, hint, icon, className = '' }: EmptyStateProps) {
  const { t } = useTranslation();
  return (
    <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
      {icon || <Inbox className="w-8 h-8 text-gray-300 mb-3" />}
      <p className="text-sm font-medium text-gray-500">{message || t('common.noData')}</p>
      {hint && <p className="text-xs text-gray-400 mt-1">{hint}</p>}
    </div>
  );
}
